"""SSE EventStream response handler for streaming token delivery.

Converts async generator chunks from ``worker.execute_streaming()``
into Server-Sent Events (SSE) compatible with the OpenAI streaming
format.  Interleaves mid-stream trace deposits into the memory field
so that latency and token-velocity are continuously updated while
the response is still being generated.
"""

import json
import time
from typing import Any, AsyncGenerator, Dict, Optional

from core.memory_field import BasePheromoneMemoryField
from core.router_agent import StigmergicRouterAgent


def _sse_line(data: Dict[str, Any]) -> str:
    """Format a dict as an SSE ``data:`` line."""
    return f"data: {json.dumps(data)}\n\n"


async def stream_chat_completion(
    router: StigmergicRouterAgent,
    memory_field: BasePheromoneMemoryField,
    prompt: str,
    max_tokens: int,
    capability_context: Optional[Dict[str, float]] = None,
    capability_match: float = 0.5,
    weights_override: Optional[Dict[str, float]] = None,
) -> AsyncGenerator[str, None]:
    """Stream a chat completion response as SSE chunks.

    Yields ``data: {...}\\n\\n`` strings suitable for direct output
    as an SSE response body.  After the first token arrives, deposits
    TTFT into the node's Latency Trace (L).  After each batch of tokens,
    deposits token velocity into the Success Trace (V).
    """
    worker = await router.sample_worker(capability_context, weights_override)

    try:
        stream = worker.execute_streaming(prompt, max_tokens)
        chunk_count = 0
        ttft_deposited = False
        first_token_time: Optional[float] = None

        async for chunk in stream:
            yield _sse_line({
                "id": f"chatcmpl-{chunk_count}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": worker.node_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": chunk.get("chunk_text", "")
                        },
                        "finish_reason": "stop" if chunk.get("done") else None,
                    }
                ],
            })

            elapsed = chunk.get("latency_sec", 0.0)
            is_first = chunk.get("is_first_token", False)
            tokens = chunk.get("token_count", 0)

            if is_first and not ttft_deposited:
                first_token_time = elapsed
                await memory_field.deposit_trace(
                    node_id=worker.node_id,
                    latency_sec=elapsed,
                    tokens=tokens,
                    success=chunk.get("success", True),
                    active_load=chunk.get("active_load", 0),
                    capability_match=capability_match,
                )
                ttft_deposited = True

            if elapsed > 0 and tokens > 0:
                velocity = tokens / elapsed
                v_boost = min(1.0, velocity / 50.0)
                await memory_field.deposit_trace(
                    node_id=worker.node_id,
                    latency_sec=elapsed,
                    tokens=tokens,
                    success=True,
                    active_load=chunk.get("active_load", 0),
                    capability_match=capability_match,
                )

            chunk_count += 1

    except Exception as exc:
        yield _sse_line({
            "error": {
                "message": str(exc),
                "type": "streaming_error",
            }
        })
    finally:
        yield "data: [DONE]\n\n"


async def stream_completion(
    router: StigmergicRouterAgent,
    memory_field: BasePheromoneMemoryField,
    prompt: str,
    max_tokens: int,
    capability_context: Optional[Dict[str, float]] = None,
    capability_match: float = 0.5,
    weights_override: Optional[Dict[str, float]] = None,
) -> AsyncGenerator[str, None]:
    """Stream a completion response as SSE chunks.

    Same as :func:`stream_chat_completion` but uses the OpenAI
    completion SSE format (``text`` field instead of ``content``).
    """
    worker = await router.sample_worker(capability_context, weights_override)

    try:
        stream = worker.execute_streaming(prompt, max_tokens)
        chunk_count = 0
        ttft_deposited = False

        async for chunk in stream:
            yield _sse_line({
                "id": f"cmpl-{chunk_count}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": worker.node_id,
                "choices": [
                    {
                        "index": 0,
                        "text": chunk.get("chunk_text", ""),
                        "finish_reason": "stop" if chunk.get("done") else None,
                    }
                ],
            })

            elapsed = chunk.get("latency_sec", 0.0)
            is_first = chunk.get("is_first_token", False)
            tokens = chunk.get("token_count", 0)

            if is_first and not ttft_deposited:
                await memory_field.deposit_trace(
                    node_id=worker.node_id,
                    latency_sec=elapsed,
                    tokens=tokens,
                    success=chunk.get("success", True),
                    active_load=chunk.get("active_load", 0),
                    capability_match=capability_match,
                )
                ttft_deposited = True

            if elapsed > 0 and tokens > 0:
                await memory_field.deposit_trace(
                    node_id=worker.node_id,
                    latency_sec=elapsed,
                    tokens=tokens,
                    success=True,
                    active_load=chunk.get("active_load", 0),
                    capability_match=capability_match,
                )

            chunk_count += 1

    except Exception as exc:
        yield _sse_line({
            "error": {
                "message": str(exc),
                "type": "streaming_error",
            }
        })
    finally:
        yield "data: [DONE]\n\n"
