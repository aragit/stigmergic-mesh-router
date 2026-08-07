"""Unit tests for PheromoneMemoryField."""

import asyncio

import numpy as np
import pytest

from core.memory_field import PheromoneMemoryField


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def node_ids():
    return ["node_alpha", "node_beta", "node_gamma"]


@pytest.fixture
async def field(node_ids):
    f = PheromoneMemoryField(node_ids=node_ids)
    return f


@pytest.fixture
async def field_with_state(node_ids):
    """Field seeded with a custom initial state for deterministic tests."""
    initial = np.array(
        [[0.5, 0.1, 0.0, 0.8],
         [0.3, 0.2, 0.1, 0.6],
         [0.8, 0.05, 0.0, 0.9]],
        dtype=np.float64,
    )
    return PheromoneMemoryField(node_ids=node_ids, initial_state=initial)


# ── Initialization ────────────────────────────────────────────────────

class TestInitialization:
    def test_matrix_dimensions(self, field, node_ids):
        """State matrix must be (N_nodes, 4)."""
        assert field._state.shape == (len(node_ids), 4)

    def test_baseline_state(self, field):
        """Default initialization must be [V=1.0, L=0.1, S=0.0, C=1.0] for all nodes."""
        expected = np.array(
            [[1.0, 0.1, 0.0, 1.0]] * 3, dtype=np.float64
        )
        np.testing.assert_allclose(field._state, expected)

    def test_custom_initial_state(self, field_with_state):
        """initial_state must be accepted and stored correctly."""
        expected = np.array(
            [[0.5, 0.1, 0.0, 0.8],
             [0.3, 0.2, 0.1, 0.6],
             [0.8, 0.05, 0.0, 0.9]],
            dtype=np.float64,
        )
        np.testing.assert_allclose(field_with_state._state, expected)

    def test_custom_initial_state_defensive_copy(self, node_ids):
        """Mutating the input array must not affect the field's internal state."""
        initial = np.array(
            [[1.0, 0.1, 0.0, 1.0],
             [0.5, 0.2, 0.0, 0.5],
             [0.8, 0.05, 0.0, 0.9]],
            dtype=np.float64,
        )
        f = PheromoneMemoryField(node_ids=node_ids, initial_state=initial)
        initial[0, 0] = 999.0
        assert f._state[0, 0] == 1.0

    def test_shape_validation(self, node_ids):
        """Wrong-shaped initial_state must raise ValueError."""
        bad_state = np.zeros((2, 4))  # only 2 rows, but 3 nodes
        with pytest.raises(ValueError, match="does not match"):
            PheromoneMemoryField(node_ids=node_ids, initial_state=bad_state)

    def test_node_index_mapping(self, field, node_ids):
        """Node IDs must map to correct row indices."""
        for i, nid in enumerate(node_ids):
            assert field._node_index[nid] == i

    def test_saturation_scale_default(self, field):
        assert field.saturation_scale == 0.1

    def test_saturation_scale_custom(self, node_ids):
        f = PheromoneMemoryField(node_ids, saturation_scale=0.25)
        assert f.saturation_scale == 0.25

    def test_decay_success_default(self, field):
        assert field.decay_success is True

    def test_decay_success_disabled(self, node_ids):
        f = PheromoneMemoryField(node_ids, decay_success=False)
        assert f.decay_success is False

    def test_decay_capability_default(self, field):
        assert field.decay_capability is True

    def test_decay_capability_disabled(self, node_ids):
        f = PheromoneMemoryField(node_ids, decay_capability=False)
        assert f.decay_capability is False


# ── Trace Deposition ───────────────────────────────────────────────────

class TestDepositTrace:
    @pytest.mark.asyncio
    async def test_success_updates_v(self, field, node_ids):
        """A successful inference must set V toward 1.0 via EWMA."""
        await field.deposit_trace(
            node_id=node_ids[0],
            latency_sec=0.05,
            tokens=10,
            success=True,
            active_load=0,
        )
        state = await field.get_state_vector()
        expected_v = 0.5 * 1.0 + 0.5 * 1.0  # alpha=0.5, success=1, old=1.0 (baseline)
        np.testing.assert_allclose(state[0, 0], expected_v)

    @pytest.mark.asyncio
    async def test_failure_updates_v(self, field, node_ids):
        """A failed inference must set V toward 0.0 via EWMA."""
        # Baseline V is 1.0; after failure it should drop
        await field.deposit_trace(
            node_id=node_ids[0], latency_sec=0.05, tokens=10,
            success=False, active_load=0,
        )
        state = await field.get_state_vector()
        expected_v = 0.5 * 0.0 + 0.5 * 1.0  # 0.5
        np.testing.assert_allclose(state[0, 0], expected_v)

    @pytest.mark.asyncio
    async def test_latency_ewma(self, field, node_ids):
        """L must update as EWMA toward observed latency."""
        await field.deposit_trace(
            node_id=node_ids[0], latency_sec=0.1, tokens=10,
            success=True, active_load=0,
        )
        state = await field.get_state_vector()
        expected_l = 0.5 * 0.1 + 0.5 * 0.1  # baseline L=0.1
        np.testing.assert_allclose(state[0, 1], expected_l)

    @pytest.mark.asyncio
    async def test_saturation_from_active_load(self, field, node_ids):
        """S must be active_load * saturation_scale."""
        await field.deposit_trace(
            node_id=node_ids[0], latency_sec=0.05, tokens=10,
            success=True, active_load=3,
        )
        state = await field.get_state_vector()
        np.testing.assert_allclose(state[0, 2], 3 * 0.1)

    @pytest.mark.asyncio
    async def test_zero_active_load_zero_saturation(self, field, node_ids):
        """Idle node (active_load=0) must have S=0."""
        await field.deposit_trace(
            node_id=node_ids[0], latency_sec=0.05, tokens=10,
            success=True, active_load=0,
        )
        state = await field.get_state_vector()
        assert state[0, 2] == 0.0

    @pytest.mark.asyncio
    async def test_capability_fit_ewma(self, field, node_ids):
        """C must update as EWMA toward capability_match."""
        await field.deposit_trace(
            node_id=node_ids[0], latency_sec=0.05, tokens=10,
            success=True, active_load=0, capability_match=0.8,
        )
        state = await field.get_state_vector()
        # Baseline C=1.0; EWMA: 0.5*0.8 + 0.5*1.0 = 0.9
        np.testing.assert_allclose(state[0, 3], 0.9)

    @pytest.mark.asyncio
    async def test_capability_match_clamped(self, field, node_ids):
        """capability_match outside [0, 1] must be clamped."""
        await field.deposit_trace(
            node_id=node_ids[0], latency_sec=0.05, tokens=10,
            success=True, active_load=0, capability_match=2.5,
        )
        state = await field.get_state_vector()
        np.testing.assert_allclose(state[0, 3], 0.5 * 1.0 + 0.5 * 1.0)

        await field.deposit_trace(
            node_id=node_ids[0], latency_sec=0.05, tokens=10,
            success=True, active_load=0, capability_match=-0.5,
        )
        state = await field.get_state_vector()
        np.testing.assert_allclose(state[0, 3], 0.5 * 0.0 + 0.5 * 1.0)

    @pytest.mark.asyncio
    async def test_default_capability_match_neutral(self, field, node_ids):
        """Default capability_match=0.5 with baseline C=1.0 gives C=0.75."""
        await field.deposit_trace(
            node_id=node_ids[0], latency_sec=0.05, tokens=10,
            success=True, active_load=0,
        )
        state = await field.get_state_vector()
        np.testing.assert_allclose(state[0, 3], 0.75)

    @pytest.mark.asyncio
    async def test_deposit_affects_only_target_node(self, field, node_ids):
        """Deposit for one node must not alter other nodes' traces."""
        await field.deposit_trace(
            node_id=node_ids[0], latency_sec=0.05, tokens=10,
            success=True, active_load=0,
        )
        state = await field.get_state_vector()
        # Other nodes still at baseline
        assert state[1, 0] == 1.0
        assert state[2, 1] == 0.1

    @pytest.mark.asyncio
    async def test_ewma_convergence(self, node_ids):
        """Repeated identical deposits must converge toward observed values."""
        f = PheromoneMemoryField(node_ids=node_ids)
        for _ in range(20):
            await f.deposit_trace(
                node_id=node_ids[0], latency_sec=0.075, tokens=10,
                success=True, active_load=2, capability_match=0.9,
            )
        state = await f.get_state_vector()
        np.testing.assert_allclose(state[0, 0], 1.0, atol=1e-3)
        np.testing.assert_allclose(state[0, 1], 0.075, atol=1e-3)
        np.testing.assert_allclose(state[0, 2], 0.2, atol=1e-3)
        np.testing.assert_allclose(state[0, 3], 0.9, atol=1e-3)

    @pytest.mark.asyncio
    async def test_invalid_node_id(self, field):
        """Depositing for an unknown node must raise KeyError."""
        with pytest.raises(KeyError):
            await field.deposit_trace(
                node_id="nonexistent", latency_sec=0.05,
                tokens=10, success=True, active_load=0,
            )


# ── Evaporation ────────────────────────────────────────────────────────

class TestEvaporation:
    @pytest.mark.asyncio
    async def test_latency_and_saturation_decay(self, field_with_state):
        """Evaporation must multiply L and S by (1 - decay_rate)."""
        # field_with_state: L=[0.1, 0.2, 0.05], S=[0.0, 0.1, 0.0], C=[0.8, 0.6, 0.9]
        await field_with_state.apply_evaporation(0.1)
        state = await field_with_state.get_state_vector()
        np.testing.assert_allclose(state[:, 1], [0.09, 0.18, 0.045])
        np.testing.assert_allclose(state[:, 2], [0.0, 0.09, 0.0])

    @pytest.mark.asyncio
    async def test_capability_decay_when_enabled(self, field_with_state):
        """C must decay when decay_capability=True (default)."""
        # C=[0.8, 0.6, 0.9]
        await field_with_state.apply_evaporation(0.1)
        state = await field_with_state.get_state_vector()
        np.testing.assert_allclose(state[:, 3], [0.72, 0.54, 0.81])

    @pytest.mark.asyncio
    async def test_success_decay_when_enabled(self, field_with_state):
        """V must decay when decay_success=True (default)."""
        # V=[0.5, 0.3, 0.8]
        await field_with_state.apply_evaporation(0.2)
        state = await field_with_state.get_state_vector()
        np.testing.assert_allclose(state[:, 0], [0.4, 0.24, 0.64])

    @pytest.mark.asyncio
    async def test_success_persists_when_disabled(self, field_with_state):
        """V must NOT decay when decay_success=False."""
        f = PheromoneMemoryField(
            node_ids=["a", "b", "c"],
            initial_state=field_with_state._state.copy(),
            decay_success=False,
        )
        await f.apply_evaporation(0.5)
        state = await f.get_state_vector()
        np.testing.assert_allclose(state[:, 0], [0.5, 0.3, 0.8])

    @pytest.mark.asyncio
    async def test_capability_persists_when_disabled(self, field_with_state):
        """C must NOT decay when decay_capability=False."""
        f = PheromoneMemoryField(
            node_ids=["a", "b", "c"],
            initial_state=field_with_state._state.copy(),
            decay_capability=False,
        )
        await f.apply_evaporation(0.5)
        state = await f.get_state_vector()
        np.testing.assert_allclose(state[:, 3], [0.8, 0.6, 0.9])

    @pytest.mark.asyncio
    async def test_zero_decay_no_change(self, field_with_state):
        """decay_rate=0 must leave the field unchanged."""
        before = (await field_with_state.get_state_vector()).copy()
        await field_with_state.apply_evaporation(0.0)
        after = await field_with_state.get_state_vector()
        np.testing.assert_array_equal(before, after)

    @pytest.mark.asyncio
    async def test_invalid_decay_rate_negative(self, field):
        with pytest.raises(ValueError, match="decay_rate must be"):
            await field.apply_evaporation(-0.1)

    @pytest.mark.asyncio
    async def test_invalid_decay_rate_one(self, field):
        with pytest.raises(ValueError, match="decay_rate must be"):
            await field.apply_evaporation(1.0)

    @pytest.mark.asyncio
    async def test_evaporation_applies_to_all_nodes(self, field_with_state):
        """Evaporation must scale ALL nodes, not just one."""
        await field_with_state.apply_evaporation(0.5)
        state = await field_with_state.get_state_vector()
        np.testing.assert_allclose(state[:, 1], [0.05, 0.1, 0.025])
        np.testing.assert_allclose(state[:, 2], [0.0, 0.05, 0.0])
        np.testing.assert_allclose(state[:, 3], [0.4, 0.3, 0.45])


# ── Async Lock / Thread-Safety ────────────────────────────────────────

class TestConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_deposits(self, node_ids):
        """Many concurrent deposits must not corrupt the state matrix."""
        f = PheromoneMemoryField(node_ids=node_ids)

        async def deposit(idx, success):
            await f.deposit_trace(
                node_id=node_ids[idx],
                latency_sec=0.05,
                tokens=10,
                success=success,
                active_load=0,
                capability_match=0.8,
            )

        tasks = []
        for i in range(100):
            idx = i % 3
            success = i % 4 != 0
            tasks.append(deposit(idx, success))
        await asyncio.gather(*tasks)

        state = await f.get_state_vector()
        assert state.shape == (3, 4)
        assert np.all(state[:, 0] >= 0.0)
        assert np.all(state[:, 0] <= 1.0)
        assert np.all(state[:, 3] >= 0.0)
        assert np.all(state[:, 3] <= 1.0)
        assert not np.any(np.isnan(state))
        assert not np.any(np.isinf(state))

    @pytest.mark.asyncio
    async def test_concurrent_deposit_and_evaporate(self, node_ids):
        """Concurrent evaporation and deposits must keep state consistent."""
        f = PheromoneMemoryField(node_ids=node_ids, decay_success=True)

        async def deposit_and_evaporate():
            for i in range(50):
                await f.deposit_trace(
                    node_id=node_ids[i % 3],
                    latency_sec=0.05,
                    tokens=10,
                    success=True,
                    active_load=0,
                    capability_match=0.7,
                )
                await f.apply_evaporation(0.1)

        async def evaporate_only():
            for _ in range(50):
                await f.apply_evaporation(0.1)
                await asyncio.sleep(0)

        await asyncio.gather(deposit_and_evaporate(), evaporate_only())
        state = await f.get_state_vector()
        assert not np.any(np.isnan(state))
        assert not np.any(np.isinf(state))

    @pytest.mark.asyncio
    async def test_concurrent_get_state(self, field_with_state):
        """Multiple concurrent get_state_vector calls must return consistent data."""
        results = await asyncio.gather(
            *[field_with_state.get_state_vector() for _ in range(20)]
        )
        for r in results:
            np.testing.assert_array_equal(r, results[0])
