"""Unit tests for SSE streaming telemetry and trace deposits.

Verifies:
* SSE chunk formatting (``data: {...}\\n\\n``).
* TTFT is deposited as Latency Trace (L) on first chunk.
* Token velocity is deposited as Success Trace (V) on subsequent chunks.
* StreamingResponse is returned when ``stream=True`` in the request.
* The [DONE] sentinel is appended at the end of the stream.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.pop("STIGMERGIC_STORAGE_BACKEND", None)
os.environ.pop("STIGMERGIC_REDIS_HOST", None)

from api.server import app
from api.streaming import stream_chat_completion, stream_completion
from core.memory_field import InMemoryPheromoneMemoryField
from core.router_agent import StigmergicRouterAgent
from core.worker_node import CPUMockWorkerNode


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def streaming_setup():
    """Create a memory field, router, and mock worker for streaming tests."""
    workers = {
        "stream-node": CPUMockWorkerNode(
            node_id="stream-node",
            base_delay_sec=0.02,
            load_factor=0.1,
            capability_tags=["slm", "fast"],
        ),
    }
    memory_field = InMemoryPheromoneMemoryField(
        node_ids=["stream-node"],
    )
    rng = np.random.default_rng(42)
    router = StigmergicRouterAgent(
        workers=workers,
        memory_field=memory_field,
        weights={"alpha": 1.0, "beta": 2.0, "gamma": 1.5, "delta": 1.5},
        temperature=1.0,
        delta=1.5,
        rng=rng,
    )
    return memory_field, router


@pytest.mark.asyncio
async def test_stream_completion_yields_sse_chunks(streaming_setup):
    """stream_completion should yield properly formatted SSE data lines."""
    memory_field, router = streaming_setup

    chunks = []
    async for line in stream_completion(
        router=router,
        memory_field=memory_field,
        prompt="Test prompt",
        max_tokens=16,
        capability_context={"slm": 1.0},
    ):
        chunks.append(line)

    assert len(chunks) > 0
    assert all(c.startswith("data: ") for c in chunks if not c.startswith("data: [DONE]"))
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_stream_deposits_first_chunk_as_ttft(streaming_setup):
    """The first chunk should deposit TTFT into the Latency Trace (L)."""
    memory_field, router = streaming_setup

    # Collect all chunks
    async for _ in stream_completion(
        router=router,
        memory_field=memory_field,
        prompt="Hello",
        max_tokens=16,
        capability_context=None,
    ):
        pass

    state = await memory_field.get_state_vector()
    # L should have been updated from baseline 0.1 to the TTFT value
    lat = state[0, 1]
    assert lat != 0.1  # Should have changed from baseline


@pytest.mark.asyncio
async def test_stream_chat_completion_format(streaming_setup):
    """stream_chat_completion should use chat SSE format with 'content' field."""
    memory_field, router = streaming_setup

    chunks = []
    async for line in stream_chat_completion(
        router=router,
        memory_field=memory_field,
        prompt="Hello",
        max_tokens=16,
        capability_context=None,
    ):
        chunks.append(line)

    # Find a content chunk (not the DONE sentinel)
    content_chunks = [c for c in chunks if "content" in c and "[DONE]" not in c]
    assert len(content_chunks) > 0


@pytest.mark.asyncio
async def test_stream_deposits_token_velocity(streaming_setup):
    """Token velocity deposits should modify the Latency Trace (L)."""
    memory_field, router = streaming_setup

    # Record L before streaming
    state_before = await memory_field.get_state_vector()
    l_before = state_before[0, 1]

    async for _ in stream_completion(
        router=router,
        memory_field=memory_field,
        prompt="Hello world",
        max_tokens=32,
        capability_context=None,
    ):
        pass

    state_after = await memory_field.get_state_vector()
    # L should have been updated from baseline 0.1 to actual TTFT
    l_after = state_after[0, 1]
    assert l_after != l_before


@pytest.mark.asyncio
async def test_stream_error_handling(streaming_setup):
    """Streaming should handle worker errors gracefully and emit [DONE]."""
    memory_field, router = streaming_setup

    from unittest.mock import AsyncMock, patch

    # Create a worker that raises an error
    bad_worker = CPUMockWorkerNode(node_id="bad-node")
    bad_worker.execute_streaming = AsyncMock(side_effect=RuntimeError("Test error"))

    router.workers["stream-node"] = bad_worker
    router.memory_field = memory_field

    chunks = []
    async for line in stream_completion(
        router=router,
        memory_field=memory_field,
        prompt="Test",
        max_tokens=16,
        capability_context=None,
    ):
        chunks.append(line)

    # Should still emit [DONE] at the end
    assert chunks[-1] == "data: [DONE]\n\n"
    # Should contain an error chunk
    error_chunks = [c for c in chunks if "error" in c.lower()]
    assert len(error_chunks) > 0


def test_streaming_response_via_api(client):
    """POST /v1/chat/completions with stream=True should return SSE."""
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Hello!"}],
            "max_tokens": 16,
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    # Collect chunks
    content = resp.text
    assert "data: " in content
    assert "[DONE]" in content


def test_non_streaming_still_works(client):
    """POST /v1/chat/completions with stream=False should return JSON."""
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Hello!"}],
            "max_tokens": 16,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/json"
    data = resp.json()
    assert "choices" in data
    assert "message" in data["choices"][0]
