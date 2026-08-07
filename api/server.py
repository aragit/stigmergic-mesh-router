"""FastAPI ingress server with OpenAI-compatible REST endpoints.

Exposes ``POST /v1/completions``, ``POST /v1/chat/completions``, and
``GET /v1/models`` — all routed through the stigmergic mesh router so
that traffic is dynamically load-balanced based on real-time pheromone
feedback rather than static round-robin or least-connections.
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from core.decay_engine import start_decay_engine
from core.memory_field import PheromoneMemoryField
from core.router_agent import StigmergicRouterAgent
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


# ── Mesh state container ────────────────────────────────────────────────

class MeshState:
    """Holds router, workers, and background tasks for the API lifecycle."""

    def __init__(self) -> None:
        self.config: Dict[str, Any] = {}
        self.workers: Dict[str, BaseWorkerNode] = {}
        self.memory_field: Optional[PheromoneMemoryField] = None
        self.router: Optional[StigmergicRouterAgent] = None
        self.decay_task: Optional[asyncio.Task] = None

    @property
    def ready(self) -> bool:
        return self.router is not None and self.memory_field is not None


_state = MeshState()


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

    _state.memory_field = PheromoneMemoryField(node_ids=node_ids)
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

    logger.info("Mesh router started with %d workers: %s", len(node_ids), ", ".join(node_ids))
    yield

    if _state.decay_task:
        _state.decay_task.cancel()
        try:
            await _state.decay_task
        except asyncio.CancelledError:
            pass
    logger.info("Mesh router shutting down")


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


@app.get("/v1/models", response_model=ModelList)
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
async def completions(request: CompletionRequest):
    """Route a completion-style request through the mesh."""
    if not _state.ready:
        raise HTTPException(status_code=503, detail="Router not initialised")

    cap_context = analyze_prompt(request.prompt)
    result = await _state.router.route_and_execute(
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        capability_context=cap_context,
    )

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
        }
    )


def _messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Convert a list of chat messages into a single prompt string."""
    parts: List[str] = []
    for msg in messages:
        prefix = "User" if msg.role == "user" else "Assistant"
        parts.append(f"{prefix}: {msg.content}")
    return "\n".join(parts) + "\nAssistant:"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Route a chat-style request through the mesh."""
    if not _state.ready:
        raise HTTPException(status_code=503, detail="Router not initialised")

    prompt = _messages_to_prompt(request.messages)
    cap_context = analyze_prompt(prompt)

    result = await _state.router.route_and_execute(
        prompt=prompt,
        max_tokens=request.max_tokens,
        capability_context=cap_context,
    )

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
        }
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
