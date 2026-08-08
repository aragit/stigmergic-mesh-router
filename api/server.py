"""FastAPI ingress server with OpenAI-compatible REST endpoints.

Exposes ``POST /v1/completions``, ``POST /v1/chat/completions``, and
``GET /v1/models`` — all routed through the stigmergic mesh router so
that traffic is dynamically load-balanced based on real-time pheromone
feedback rather than static round-robin or least-connections.
"""

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import uvicorn
import yaml
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import make_asgi_app
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

from api.auth import (
    TenantContext,
    get_current_tenant,
)
from api.governance import TenantPolicyEngine
from api.rate_limiter import RedisRateLimiter
from api.metrics import (
    stigmergic_requests_total,
    stigmergic_request_duration_seconds,
    update_prometheus_metrics,
)
from api.streaming import stream_chat_completion, stream_completion
from core.decay_engine import start_decay_engine
from core.memory_field import (
    BasePheromoneMemoryField,
    get_memory_field,
)
from core.router_agent import StigmergicRouterAgent
from core.checkpointing import CheckpointManager
from core.worker_node import (
    BaseWorkerNode,
    CPUMockWorkerNode,
    GPUvLLMWorkerNode,
    OllamaWorkerNode,
)

logger = logging.getLogger("stigmergic.api")

CONFIG_PATH = "config.yaml"


# ── Pydantic request / response models ──────────────────────────────────

class ChatMessage(BaseModel):
    """A single chat message in the OpenAI chat format."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible request for ``POST /v1/chat/completions``."""

    model: str = Field(default="stigmergic-mesh")
    messages: List[ChatMessage] = Field(default_factory=list)
    max_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    stop: Optional[List[str]] = None


class CompletionRequest(BaseModel):
    """OpenAI-compatible request for ``POST /v1/completions``."""

    model: str = Field(default="stigmergic-mesh")
    prompt: str
    max_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    stop: Optional[List[str]] = None


class ModelInfo(BaseModel):
    """Model information returned by ``GET /v1/models``."""

    id: str
    object: str = "model"
    owned_by: str = "stigmergic"


class ModelList(BaseModel):
    """Response body for ``GET /v1/models``."""

    object: str = "list"
    data: List[ModelInfo]


# ── Ingress governance helpers (Phase 11) ─────────────────────────────────

def _estimate_prompt_tokens(prompt: str) -> int:
    """Estimate input-token count for *prompt*.

    Mirrors the heuristic already used by the usage reporting in the
    completion handlers (``len(prompt.split())``) so the TPM quota is
    charged consistently with what clients see reported.
    """
    return len(prompt.split())


def _rate_limit_headers(info: Dict[str, Any]) -> Dict[str, str]:
    """Derive ``X-RateLimit-*`` headers from a rate-limit *info* dict."""
    return {
        "X-RateLimit-Limit": str(int(info["limit"])),
        "X-RateLimit-Remaining": str(max(0, int(info["remaining"]))),
        "X-RateLimit-Reset": str(int(info["reset"])),
    }


def _governed_weights(tenant: TenantContext) -> Optional[Dict[str, float]]:
    """Return governance-biased matrix weights for *tenant*, or ``None``.

    When no policy engine is configured (security disabled) this returns
    ``None`` so the router falls back to its own defaults.
    """
    if _state.policy_engine is None or _state.router is None:
        return None
    base = {
        "alpha": _state.router.alpha,
        "beta": _state.router.beta,
        "gamma": _state.router.gamma,
        "delta": _state.router.delta,
    }
    return _state.policy_engine.apply_governance_bias(tenant, base)


async def _check_rate_and_headers(
    tenant: TenantContext,
    prompt: str,
    max_tokens: int,
) -> Tuple[Dict[str, str], bool]:
    """Evaluate RPM/TPM quotas and build ``X-RateLimit-*`` headers.

    Returns ``(headers, allowed)``.  When rate limiting is unconfigured
    (no Redis / security disabled) the headers dict is empty and
    ``allowed`` is ``True``.
    """
    if _state.rate_limiter is None:
        return {}, True
    prompt_tokens = _estimate_prompt_tokens(prompt)
    cost = prompt_tokens + max_tokens
    allowed, info = await _state.rate_limiter.check_rate_limits(tenant, cost)
    headers = _rate_limit_headers(info)
    if not allowed:
        headers["Retry-After"] = str(int(info["reset"]))
    return headers, allowed


class MeshState:
    """Holds router, workers, and background tasks for the API lifecycle."""

    def __init__(self) -> None:
        self.config: Dict[str, Any] = {}
        self.workers: Dict[str, BaseWorkerNode] = {}
        self.memory_field: Optional[BasePheromoneMemoryField] = None
        self.router: Optional[StigmergicRouterAgent] = None
        self.decay_task: Optional[asyncio.Task] = None
        self.metrics_task: Optional[asyncio.Task] = None
        # Phase 11: ingress security / governance components.
        self.authenticator: Optional[Any] = None
        self.rate_limiter: Optional[RedisRateLimiter] = None
        self.policy_engine: Optional[TenantPolicyEngine] = None
        self.redis_client: Optional[Any] = None
        # Phase 12: pheromone state checkpointing.
        self.checkpoint_manager: Optional[Any] = None
        self.checkpoint_task: Optional[asyncio.Task] = None

    @property
    def ready(self) -> bool:
        return self.router is not None and self.memory_field is not None


_state = MeshState()


async def _metrics_loop() -> None:
    """Background task that syncs memory field state into Prometheus Gauges."""
    while True:
        if _state.memory_field is not None and _state.router is not None:
            await update_prometheus_metrics(_state.memory_field, _state.router)
        await asyncio.sleep(1.0)


def _create_worker(spec: Dict[str, Any]) -> BaseWorkerNode:
    """Instantiate a worker node from a config dict."""
    wtype = spec.get("type", "mock")
    node_id = spec["node_id"]
    if wtype == "mock":
        return CPUMockWorkerNode(
            node_id=node_id,
            base_delay_sec=spec.get("base_delay_sec", 0.05),
            load_factor=spec.get("load_factor", 0.5),
            success_rate=spec.get("success_rate", 1.0),
            capability_tags=spec.get("capability_tags"),
        )
    elif wtype == "vllm":
        return GPUvLLMWorkerNode(
            node_id=node_id,
            base_url=spec["base_url"],
            api_key=spec.get("api_key", ""),
            model=spec.get("model", "default"),
            timeout_sec=spec.get("timeout_sec", 30.0),
            capability_tags=spec.get("capability_tags"),
        )
    elif wtype == "ollama":
        return OllamaWorkerNode(
            node_id=node_id,
            base_url=spec["base_url"],
            model=spec.get("model", "phi3"),
            timeout_sec=spec.get("timeout_sec", 60.0),
            capability_tags=spec.get("capability_tags"),
        )
    else:
        raise ValueError(f"Unknown worker type: {wtype!r}")


# ── Prompt Capability Analysis ─────────────────────────────────────────

_THINKING_PATTERNS = [
    "think", "reason", "step by step", "explain",
    "why", "because", "analyze", "analysis",
    "chain of thought", "let's think", "</thinking>",
]
_LONG_PROMPT_THRESHOLD = 500
_SHORT_PROMPT_THRESHOLD = 100


def analyze_prompt(prompt: str) -> Dict[str, float]:
    """Analyze a prompt and return capability context multipliers.

    Inspects prompt length and content to compute a mapping of
    capability tags to importance weights.  Short, simple prompts bias
    toward ``"slm"`` and ``"low-latency"`` tags; long prompts with
    thinking/reasoning keywords bias toward ``"llm"`` and ``"reasoning"``
    tags.
    """
    context: Dict[str, float] = {}
    lowered = prompt.lower()
    has_thinking = any(p in lowered for p in _THINKING_PATTERNS)

    # Thinking/reasoning patterns override length-based routing — a
    # short prompt that says "let's think step by step" still needs
    # an LLM-capable node.
    if has_thinking:
        context["llm"] = 1.5
        context["reasoning"] = 1.5
    elif len(prompt) < _SHORT_PROMPT_THRESHOLD:
        context["slm"] = 1.5
        context["fast"] = 1.3
        context["low-latency"] = 1.2

    if len(prompt) > _LONG_PROMPT_THRESHOLD:
        context["llm"] = context.get("llm", 1.0) * 1.5
        context["balanced"] = 1.2

    return context


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise mesh state on startup, tear down on shutdown."""
    with open(CONFIG_PATH, "r") as f:
        _state.config = yaml.safe_load(f)

     # Allow environment variable overrides for container deployments
    env_backend = os.environ.get("STIGMERGIC_STORAGE_BACKEND")
    if env_backend:
        _state.config["storage_backend"] = env_backend
    env_host = os.environ.get("STIGMERGIC_REDIS_HOST")
    if env_host:
        redis_cfg = _state.config.setdefault("redis", {})
        redis_cfg["host"] = env_host

    # Phase 11: ingress security overrides for container deployments.
    sec_cfg = _state.config.setdefault("security", {})
    env_security = os.environ.get("STIGMERGIC_SECURITY_ENABLED")
    if env_security is not None:
        sec_cfg["enabled"] = env_security.lower() in ("1", "true", "yes")
    env_tenant_keys = os.environ.get("STIGMERGIC_DEFAULT_TENANT_KEYS")
    if env_tenant_keys:
        try:
            import json as _json

            sec_cfg["defaultTenantKeys"] = _json.loads(env_tenant_keys)
        except (ValueError, TypeError):
            logger.warning("STIGMERGIC_DEFAULT_TENANT_KEYS contained invalid JSON; ignoring")

    # Phase 12: checkpointing overrides for container deployments.  Gated
    # by ``enabled`` (mirrors the Phase 11 ``security.enabled`` convention)
    # so local-dev / unit tests cold-start without writing to disk.
    ckpt_cfg = _state.config.setdefault("checkpoints", {})
    env_ckpt = os.environ.get("STIGMERGIC_CHECKPOINTS_ENABLED")
    if env_ckpt is not None:
        ckpt_cfg["enabled"] = env_ckpt.lower() in ("1", "true", "yes")
    env_interval = os.environ.get("STIGMERGIC_CHECKPOINT_INTERVAL")
    if env_interval is not None:
        try:
            ckpt_cfg["interval"] = int(env_interval)
        except ValueError:
            pass
    env_storage = os.environ.get("STIGMERGIC_CHECKPOINT_STORAGE_PATH")
    if env_storage:
        ckpt_cfg["storage_path"] = env_storage

    worker_specs = _state.config.get("server", {}).get("workers", [])
    if not worker_specs:
        # Fallback: create three mock workers
        worker_specs = [
            {"node_id": "mock_node_0", "type": "mock", "base_delay_sec": 0.05},
            {"node_id": "mock_node_1", "type": "mock", "base_delay_sec": 0.05},
            {"node_id": "mock_node_2", "type": "mock", "base_delay_sec": 0.10},
        ]

    _state.workers = {
        spec["node_id"]: _create_worker(spec) for spec in worker_specs
    }
    node_ids = list(_state.workers.keys())

    _state.memory_field = get_memory_field(_state.config, node_ids)
    _state.router = StigmergicRouterAgent(
        workers=_state.workers,
        memory_field=_state.memory_field,
        weights=_state.config.get("weights", {}),
        temperature=_state.config.get("temperature", 0.5),
        delta=_state.config.get("weights", {}).get("delta", 1.5),
    )

    _state.decay_task = asyncio.create_task(
        start_decay_engine(
            memory_field=_state.memory_field,
            decay_rate=_state.config.get("decay_rate", 0.05),
            interval_sec=_state.config.get("decay_interval_sec", 0.5),
        )
    )

    _state.metrics_task = asyncio.create_task(
        _metrics_loop()
    )

    # Phase 11: configure tenant authentication, rate limiting, and
    # governance when security is enabled.  When disabled (local-dev /
    # test default) the components stay ``None`` and requests flow
    # through using a permissive default tenant.
    await _configure_security()

    # Phase 12: hydrate pheromone state from the latest checkpoint (if any)
    # so freshly-booted / recovered replicas skip cold-start exploration.
    await _warmup_checkpoint()

    logger.info("Mesh router started with %d workers: %s", len(node_ids), ", ".join(node_ids))
    yield

    if _state.metrics_task:
        _state.metrics_task.cancel()
        try:
            await _state.metrics_task
        except asyncio.CancelledError:
            pass
    if _state.decay_task:
        _state.decay_task.cancel()
        try:
            await _state.decay_task
        except asyncio.CancelledError:
            pass
    # Phase 12: capture a final shutdown snapshot before exiting.
    await _shutdown_checkpoint()
    if _state.checkpoint_task:
        _state.checkpoint_task.cancel()
        try:
            await _state.checkpoint_task
        except asyncio.CancelledError:
            pass
    if _state.redis_client is not None:
        try:
            await _state.redis_client.aclose()
        except Exception:
            logger.debug("Redis client close skipped", exc_info=True)
    logger.info("Mesh router shutting down")


async def _warmup_checkpoint() -> None:
    """Hydrate the memory field from the latest checkpoint, if enabled.

    When checkpointing is disabled (the local-dev / test default) this is a
    no-op that reports a cold start, so existing flows are unaffected.
    """
    global _state

    from api.metrics import stigmergic_checkpoint_restore_status

    ckpt_cfg: Dict[str, Any] = _state.config.get("checkpoints", {})
    if not ckpt_cfg.get("enabled", False):
        stigmergic_checkpoint_restore_status.set(0)
        logger.info("Checkpointing disabled — cold-starting with balanced seed weights")
        return

    cm = CheckpointManager(
        memory_field=_state.memory_field,
        redis_client=_state.redis_client,
        storage_path=ckpt_cfg.get("storage_path", "./data/checkpoints"),
        router_agent=_state.router,
    )
    _state.checkpoint_manager = cm

    snapshot = await cm.load_latest_checkpoint()
    if snapshot is not None:
        try:
            updated = await _state.memory_field.hydrate_from_snapshot(snapshot)
            stigmergic_checkpoint_restore_status.set(1)
            logger.info(
                "Warm-started from checkpoint: hydrated %d nodes (t=%.0f, %d requests)",
                updated, snapshot.timestamp, snapshot.total_routed_requests,
            )
        except Exception as exc:
            stigmergic_checkpoint_restore_status.set(0)
            logger.warning("Checkpoint restore failed (cold-starting): %s", exc)
    else:
        stigmergic_checkpoint_restore_status.set(0)
        logger.info("No checkpoint found — cold-starting with balanced seed weights")

    interval = int(ckpt_cfg.get("interval", 60))
    _state.checkpoint_task = asyncio.create_task(
        cm.start_periodic_checkpointing(interval_seconds=interval)
    )


async def _shutdown_checkpoint() -> None:
    """Persist a final snapshot before the process exits."""
    global _state

    if _state.checkpoint_manager is None or _state.memory_field is None:
        return
    try:
        snapshot = await _state.checkpoint_manager.create_snapshot()
        await _state.checkpoint_manager.save_checkpoint(snapshot)
        logger.info("Final checkpoint saved (timestamp=%.0f)", snapshot.timestamp)
    except Exception as exc:
        logger.warning("Final checkpoint save failed: %s", exc)


def _build_secret_keys_map(
    raw_keys: list,
) -> Dict[str, "TenantContext"]:
    """Translate ``security.defaultTenantKeys`` into a hash -> context map."""
    mapping: Dict[str, "TenantContext"] = {}
    for entry in raw_keys or []:
        key_hash = entry.get("key_hash")
        if not key_hash:
            continue
        mapping[key_hash] = _entry_to_tenant(entry)
    return mapping


def _entry_to_tenant(entry: Dict[str, Any]) -> "TenantContext":
    return TenantContext(
        tenant_id=entry.get("tenant_id", "unknown"),
        tier=entry.get("tier", "pro"),
        rpm_limit=int(entry.get("rpm_limit", 600)),
        tpm_limit=int(entry.get("tpm_limit", 100_000)),
        priority_weight=float(entry.get("priority_weight", 1.0)),
    )


async def _configure_security() -> None:
    """Initialise authenticator, rate limiter, and policy engine from config."""
    global _state

    sec_cfg: Dict[str, Any] = _state.config.get("security", {})
    if not sec_cfg.get("enabled", False):
        return

    redis_cfg = _state.config.get("redis", {})
    redis_host = redis_cfg.get("host", "localhost")
    redis_port = int(redis_cfg.get("port", 6379))
    redis_db = int(redis_cfg.get("db", 0))
    redis_password = redis_cfg.get("password")

    try:
        import redis.asyncio as aioredis

        _state.redis_client = aioredis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=False,
        )
    except ImportError:  # pragma: no cover
        logger.warning("redis-py not installed; auth will use fallback map only")
        _state.redis_client = None

    secret_keys_map = _build_secret_keys_map(
        sec_cfg.get("defaultTenantKeys", [])
    )

    from api.auth import APIKeyAuthenticator

    _state.authenticator = APIKeyAuthenticator(
        redis_client=_state.redis_client,
        secret_keys_map=secret_keys_map,
    )
    _state.rate_limiter = RedisRateLimiter(redis_client=_state.redis_client)
    _state.policy_engine = TenantPolicyEngine()
    logger.info("Security enabled: %d default tenant keys configured", len(secret_keys_map))


# ── FastAPI application ─────────────────────────────────────────────────

app = FastAPI(
    title="Stigmergic Mesh Router",
    description="Decentralised LLM load router using pheromone feedback.",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Expose Prometheus metrics at /metrics
app.mount("/metrics", make_asgi_app())


@app.get("/v1/models", response_model=ModelList,
         dependencies=[Depends(get_current_tenant)])
async def list_models():
    """Return available models (the mesh and individual nodes)."""
    data: List[ModelInfo] = [
        ModelInfo(id="stigmergic-mesh", owned_by="stigmergic"),
    ]
    if _state.ready:
        for nid in _state.memory_field.node_ids:
            data.append(ModelInfo(id=nid, owned_by="stigmergic"))
    return ModelList(data=data)


@app.post("/v1/completions")
async def completions(
    request: CompletionRequest,
    tenant: TenantContext = Depends(get_current_tenant),
):
    """Route a completion-style request through the mesh."""
    if not _state.ready:
        raise HTTPException(status_code=503, detail="Router not initialised")

    base_weights = _governed_weights(tenant)
    cap_context = analyze_prompt(request.prompt)

    if request.stream:
        rate_headers, allowed = await _check_rate_and_headers(
            tenant, request.prompt, request.max_tokens
        )
        if not allowed:
            return JSONResponse(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded"},
                headers=rate_headers,
            )
        return StreamingResponse(
            media_type="text/event-stream",
            content=stream_completion(
                router=_state.router,
                memory_field=_state.memory_field,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                capability_context=cap_context,
                capability_match=0.5,
                weights_override=base_weights,
            ),
            headers=rate_headers,
        )

    start = time.monotonic()
    rate_headers, allowed = await _check_rate_and_headers(
        tenant, request.prompt, request.max_tokens
    )
    if not allowed:
        return JSONResponse(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Rate limit exceeded"},
            headers=rate_headers,
        )
    result = await _state.router.route_and_execute(
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        capability_context=cap_context,
        weights_override=base_weights,
    )
    duration = time.monotonic() - start

    status = "success" if result.get("success", True) else "error"
    node_id = result.get("node_id", "unknown")
    stigmergic_requests_total.labels(node_id=node_id, status=status).inc()
    stigmergic_request_duration_seconds.labels(node_id=node_id).observe(duration)

    now = int(time.time())
    completion_text = result.get("text", f"[Response from {result['node_id']}]")

    return JSONResponse(
        {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": now,
            "model": result.get("node_id", request.model),
            "choices": [
                {
                    "index": 0,
                    "text": completion_text,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(request.prompt.split()),
                "completion_tokens": result.get("tokens", 0),
                "total_tokens": len(request.prompt.split()) + result.get("tokens", 0),
            },
        },
        headers=rate_headers,
    )


def _messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Convert a list of chat messages into a single prompt string."""
    parts: List[str] = []
    for msg in messages:
        prefix = "User" if msg.role == "user" else "Assistant"
        parts.append(f"{prefix}: {msg.content}")
    return "\n".join(parts) + "\nAssistant:"


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    tenant: TenantContext = Depends(get_current_tenant),
):
    """Route a chat-style request through the mesh."""
    if not _state.ready:
        raise HTTPException(status_code=503, detail="Router not initialised")

    base_weights = _governed_weights(tenant)

    prompt = _messages_to_prompt(request.messages)
    cap_context = analyze_prompt(prompt)

    rate_headers, allowed = await _check_rate_and_headers(
        tenant, prompt, request.max_tokens
    )
    if not allowed:
        return JSONResponse(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Rate limit exceeded"},
            headers=rate_headers,
        )

    if request.stream:
        stream_kwargs: Dict[str, Any] = dict(
            router=_state.router,
            memory_field=_state.memory_field,
            prompt=prompt,
            max_tokens=request.max_tokens,
            capability_context=cap_context,
            capability_match=0.5,
        )
        if base_weights is not None:
            stream_kwargs["weights_override"] = base_weights
        return StreamingResponse(
            media_type="text/event-stream",
            content=stream_chat_completion(**stream_kwargs),
            headers=rate_headers,
        )

    start = time.monotonic()
    result = await _state.router.route_and_execute(
        prompt=prompt,
        max_tokens=request.max_tokens,
        capability_context=cap_context,
        weights_override=base_weights,
    )
    duration = time.monotonic() - start

    status = "success" if result.get("success", True) else "error"
    node_id = result.get("node_id", "unknown")
    stigmergic_requests_total.labels(node_id=node_id, status=status).inc()
    stigmergic_request_duration_seconds.labels(node_id=node_id).observe(duration)

    now = int(time.time())
    content = result.get("text", f"[Response from {result['node_id']}]")

    return JSONResponse(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": now,
            "model": result.get("node_id", request.model),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": result.get("tokens", 0),
                "total_tokens": len(prompt.split()) + result.get("tokens", 0),
            },
        },
        headers=rate_headers,
    )


@app.get("/health")
async def health():
    """Simple health-check endpoint."""
    if not _state.ready:
        return JSONResponse({"status": "starting"}, status_code=206)
    return JSONResponse(
        {
            "status": "ok",
            "nodes": _state.memory_field.node_ids,
            "node_count": len(_state.memory_field.node_ids),
        }
    )


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for running the API server standalone."""
    config = uvicorn.Config(
        "api.server:app",
        host=host,
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    asyncio.run(server.serve())


if __name__ == "__main__":
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    srv_cfg = cfg.get("server", {})
    run_server(
        host=srv_cfg.get("host", "0.0.0.0"),
        port=srv_cfg.get("port", 8000),
    )
