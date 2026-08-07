"""Unit tests for RedisPheromoneMemoryField using fakeredis.

These tests verify atomic trace reads, writes, and decay mechanics
without requiring a live Redis server, using ``fakeredis.aioredis``.
"""

import asyncio
import math
import time

import fakeredis.aioredis
import numpy as np
import pytest

from core.memory_field import RedisPheromoneMemoryField


@pytest.fixture
def fake_redis():
    """Provide a fresh fakeredis instance for each test."""
    return fakeredis.aioredis.FakeRedis()


@pytest.fixture
async def redis_field(fake_redis):
    """Create a RedisPheromoneMemoryField backed by fakeredis.

    Uses ``decay_rate=0`` so lazy decay does not introduce timing
    drift in tests that assert exact EWMA values.
    """
    field = RedisPheromoneMemoryField(
        node_ids=["node-a", "node-b", "node-c"],
        redis_client=fake_redis,
        decay_rate=0.0,
        decay_interval_sec=0.5,
        saturation_scale=0.1,
        decay_success=True,
        decay_capability=True,
    )
    return field


@pytest.mark.asyncio
async def test_redis_initial_state_is_baseline(redis_field):
    """Newly initialised field should start from the baseline state."""
    state = await redis_field.get_state_vector()
    assert state.shape == (3, 4)

    # V=1.0, L=0.1, S=0.0, C=1.0
    np.testing.assert_allclose(state[0], [1.0, 0.1, 0.0, 1.0])
    np.testing.assert_allclose(state[1], [1.0, 0.1, 0.0, 1.0])
    np.testing.assert_allclose(state[2], [1.0, 0.1, 0.0, 1.0])


@pytest.mark.asyncio
async def test_redis_deposit_updates_state(redis_field):
    """Depositing a trace should update the corresponding node's row."""
    await redis_field.deposit_trace(
        node_id="node-b",
        latency_sec=0.05,
        tokens=128,
        success=True,
        active_load=2,
        capability_match=0.8,
    )

    state = await redis_field.get_state_vector()

    # node-b: V should go from 1.0 → EWMA(1.0, success=1.0) = 1.0
    #         L should go from 0.1 → EWMA(0.1, 0.05) = 0.075
    #         S should go from 0.0 → 2 * 0.1 = 0.2
    #         C should go from 1.0 → EWMA(1.0, 0.8) = 0.9
    np.testing.assert_allclose(
        state[1], [1.0, 0.075, 0.2, 0.9], rtol=1e-5
    )

    # node-a and node-c should be unchanged
    np.testing.assert_allclose(state[0], [1.0, 0.1, 0.0, 1.0], rtol=1e-5)
    np.testing.assert_allclose(state[2], [1.0, 0.1, 0.0, 1.0], rtol=1e-5)


@pytest.mark.asyncio
async def test_redis_failure_deposits_zero_success(redis_field):
    """A failed inference should pull V toward 0 via EWMA."""
    await redis_field.deposit_trace(
        node_id="node-a",
        latency_sec=0.2,
        tokens=50,
        success=False,
        active_load=1,
        capability_match=0.3,
    )

    state = await redis_field.get_state_vector()

    # V: EWMA(1.0, 0.0) = 0.5
    # L: EWMA(0.1, 0.2) = 0.15
    # S: 1 * 0.1 = 0.1
    # C: EWMA(1.0, 0.3) = 0.65
    np.testing.assert_allclose(
        state[0], [0.5, 0.15, 0.1, 0.65], rtol=1e-5
    )


@pytest.mark.asyncio
async def test_redis_apply_evaporation(redis_field):
    """Bulk evaporation should scale V, L, S, C by (1 - decay_rate).

    With decay_rate=0, lazy decay is a no-op, so the expected result
    after apply_evaporation(0.1) is simply state_before * 0.9.
    """
    # Set non-baseline values first
    await redis_field.deposit_trace(
        node_id="node-a",
        latency_sec=0.5,
        tokens=10,
        success=True,
        active_load=5,
        capability_match=0.5,
    )

    state_before = await redis_field.get_state_vector()

    await redis_field.apply_evaporation(decay_rate=0.1)

    state_after = await redis_field.get_state_vector()

    factor = 0.9
    for idx in range(3):
        expected = state_before[idx] * factor
        np.testing.assert_allclose(state_after[idx], expected, rtol=1e-5)


@pytest.mark.asyncio
async def test_redis_evaporation_skips_success_when_disabled(fake_redis):
    """When decay_success=False, V should NOT decay during evaporation."""
    field = RedisPheromoneMemoryField(
        node_ids=["node-x", "node-y"],
        redis_client=fake_redis,
        decay_rate=0.0,
        decay_success=False,
        decay_capability=True,
    )

    await field.deposit_trace(
        node_id="node-x",
        latency_sec=0.3,
        tokens=10,
        success=False,
        active_load=2,
        capability_match=0.5,
    )

    state_before = await field.get_state_vector()
    v_before, l_before, s_before, c_before = state_before[0]

    await field.apply_evaporation(decay_rate=0.1)

    state_after = await field.get_state_vector()
    v_after, l_after, s_after, c_after = state_after[0]

    # V should NOT have changed (decay_success=False means V is skipped)
    assert math.isclose(v_after, v_before, rel_tol=1e-5)

    # L and S should have decayed
    assert l_after < l_before
    assert s_after < s_before

    # C should have decayed
    assert c_after < c_before


@pytest.mark.asyncio
async def test_redis_concurrent_deposits_no_lost_writes(redis_field):
    """Concurrent trace deposits to the same node must not lose writes.

    Runs 50 concurrent deposit_trace coroutines hitting node-a.  After
    all complete, V should be exactly 1.0 (EWMA of 50 successes on a
    baseline of 1.0 stays at 1.0), and L should be the EWMA of 50
    identical latency values — also deterministic.
    """
    async def deposit_one():
        await redis_field.deposit_trace(
            node_id="node-a",
            latency_sec=0.02,
            tokens=64,
            success=True,
            active_load=0,
            capability_match=0.5,
        )

    await asyncio.gather(*[deposit_one() for _ in range(50)])

    state = await redis_field.get_state_vector()
    # V stays 1.0 regardless of update order
    assert math.isclose(state[0, 0], 1.0, abs_tol=1e-6)
    # L converges to 0.02 (all deposits use latency 0.02)
    assert math.isclose(state[0, 1], 0.02, abs_tol=1e-3)
    # S stays 0.0 (all deposits use active_load=0)
    assert math.isclose(state[0, 2], 0.0, abs_tol=1e-6)


@pytest.mark.asyncio
async def test_redis_deposit_unknown_node_raises(redis_field):
    """Depositing to a node not in the field should raise KeyError."""
    with pytest.raises(KeyError):
        await redis_field.deposit_trace(
            node_id="nonexistent",
            latency_sec=0.1,
            tokens=1,
            success=True,
            active_load=1,
        )


@pytest.mark.asyncio
async def test_redis_close(redis_field, fake_redis):
    """close() should gracefully shut down the Redis connection."""
    await redis_field.close()
    # No exception means we're good
