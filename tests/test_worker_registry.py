"""Unit tests for dynamic worker auto-registration via Redis Pub/Sub.

Tests use ``fakeredis.aioredis`` to verify:
* Worker registration on the discovery channel.
* Heartbeat TTL-key expiration and pruning.
* Dynamic state matrix expansion (add_node) and pruning (remove_node).
"""

import asyncio
import time

import fakeredis.aioredis
import numpy as np
import pytest

from core.memory_field import InMemoryPheromoneMemoryField
from core.worker_registry import WorkerRegistry


@pytest.fixture
def fake_redis():
    return fakeredis.aioredis.FakeRedis()


@pytest.fixture
def memory_field():
    return InMemoryPheromoneMemoryField(
        node_ids=["node-0"],
    )


@pytest.fixture
def registry(fake_redis, memory_field):
    return WorkerRegistry(
        redis_client=fake_redis,
        memory_field=memory_field,
        heartbeat_ttl=2,
        heartbeat_interval=0.5,
        prune_interval=0.5,
    )


@pytest.mark.asyncio
async def test_register_worker_adds_to_registry(registry):
    """Registering a worker should add it to the known nodes and expand the matrix."""
    await registry.register_worker({
        "node_id": "worker-a",
        "capability_tags": ["slm", "fast"],
    })

    assert "worker-a" in registry.known_nodes
    assert "worker-a" in registry._memory_field.node_ids
    # Matrix should now have 2 rows (original + new)
    state = await registry._memory_field.get_state_vector()
    assert state.shape == (2, 4)


@pytest.mark.asyncio
async def test_register_worker_already_present_no_duplicate(registry):
    """Registering the same worker twice should not duplicate the matrix row."""
    config = {"node_id": "worker-a", "capability_tags": ["slm"]}
    await registry.register_worker(config)
    await registry.register_worker(config)

    assert registry._memory_field.node_ids.count("worker-a") == 1
    state = await registry._memory_field.get_state_vector()
    assert state.shape == (2, 4)  # still 2, not 3


@pytest.mark.asyncio
async def test_heartbeat_sets_ttl_key(fake_redis, registry):
    """send_heartbeat should refresh the TTL key in Redis."""
    await registry.register_worker({
        "node_id": "worker-b",
        "capability_tags": [],
    })

    heartbeat_key = registry._heartbeat_key("worker-b")

    # Key should exist after registration
    exists = await fake_redis.exists(heartbeat_key)
    assert exists == 1

    # After heartbeat, TTL should be refreshed (still exists)
    await asyncio.sleep(0.1)
    await registry.send_heartbeat("worker-b")
    ttl = await fake_redis.ttl(heartbeat_key)
    assert ttl > 0  # key still has time-to-live


@pytest.mark.asyncio
async def test_heartbeat_key_expires_and_prune_removes_worker(fake_redis, registry):
    """When a heartbeat key expires, prune_unhealthy_workers should detect it."""
    # Use a very short TTL for testing
    registry._heartbeat_ttl = 1

    await registry.register_worker({
        "node_id": "worker-x",
        "capability_tags": ["test"],
    })

    assert "worker-x" in registry.known_nodes

    # Wait for TTL to expire
    await asyncio.sleep(1.5)

    # TTL key should no longer exist
    exists = await fake_redis.exists(registry._heartbeat_key("worker-x"))
    assert exists == 0

    # Prune should detect and remove the worker
    await registry.prune_unhealthy_workers()
    assert "worker-x" not in registry.known_nodes


@pytest.mark.asyncio
async def test_prune_sets_latency_to_999(fake_redis, memory_field):
    """Pruned workers should have their latency trace set to 999.0."""
    registry = WorkerRegistry(
        redis_client=fake_redis,
        memory_field=memory_field,
        heartbeat_ttl=1,          # 1 second TTL
        prune_interval=0.5,
    )

    await registry.register_worker({
        "node_id": "worker-y",
        "capability_tags": [],
    })

    await asyncio.sleep(1.5)  # Exceed TTL of 1 second
    await registry.prune_unhealthy_workers()

    # The node should still exist in the memory field (not removed),
    # but its L trace should have been set to 999.0 via deposit_trace
    state = await registry._memory_field.get_state_vector()
    node_idx = registry._memory_field.node_ids.index("worker-y")
    # V should be degraded (EWMA of 1.0 with success=0 → 0.5)
    assert state[node_idx, 0] < 1.0
    # L should be high (999.0 merged via EWMA → at least > 1.0)
    assert state[node_idx, 1] > 1.0


@pytest.mark.asyncio
async def test_dynamic_add_node_expands_matrix(memory_field):
    """add_node should append a new row to the state matrix."""
    original_count = len(memory_field.node_ids)
    await memory_field.add_node("new-node", ["slm"])

    assert len(memory_field.node_ids) == original_count + 1
    state = await memory_field.get_state_vector()
    new_idx = memory_field._node_index["new-node"]
    np.testing.assert_allclose(
        state[new_idx], [1.0, 0.1, 0.0, 1.0]
    )


@pytest.mark.asyncio
async def test_dynamic_remove_node_shrinks_matrix(memory_field):
    """remove_node should remove a row and reindex."""
    await memory_field.add_node("temp-node", [])
    assert len(memory_field.node_ids) == 2

    await memory_field.remove_node("temp-node")
    assert len(memory_field.node_ids) == 1
    assert "temp-node" not in memory_field.node_ids

    # Reindex should be correct
    idx = memory_field._node_index.get("temp-node")
    assert idx is None


@pytest.mark.asyncio
async def test_start_and_stop_background_tasks(registry):
    """Background discovery and prune tasks should start and stop cleanly."""
    await registry.start_background_tasks()
    assert registry._running is True
    assert registry._discovery_task is not None
    assert registry._prune_task is not None

    await asyncio.sleep(0.3)
    await registry.stop_background_tasks()
    assert registry._running is False


@pytest.mark.asyncio
async def test_discovery_channel_publishes_json(fake_redis, registry):
    """register_worker should call redis.publish with a JSON message.

    Uses unittest.mock to spy on the publish call since fakeredis
    async pubsub support is limited.
    """
    import json as _json
    from unittest.mock import patch, AsyncMock

    publish_spy = AsyncMock(wraps=fake_redis.publish)
    with patch.object(fake_redis, "publish", publish_spy):
        await registry.register_worker({
            "node_id": "pub-test",
            "capability_tags": ["test"],
        })

        # Verify publish was called with the discovery channel
        publish_spy.assert_called()
        call_args = publish_spy.call_args
        channel = call_args[0][0]  # positional args
        message = call_args[0][1]

        assert channel == "stigmergic:worker:discovery"
        data = _json.loads(message)
        assert data["node_id"] == "pub-test"
        assert data["action"] == "register"
        assert data["capability_tags"] == ["test"]
