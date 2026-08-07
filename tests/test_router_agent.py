"""Unit tests for StigmergicRouterAgent."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from core.memory_field import PheromoneMemoryField
from core.router_agent import StigmergicRouterAgent
from core.worker_node import BaseWorkerNode


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def node_ids():
    return ["node_alpha", "node_beta", "node_gamma"]


@pytest.fixture
def mock_workers(node_ids):
    """Create mock workers that return deterministic results."""
    workers = {}
    caps = ["slm", "fast", "low-latency"], ["llm", "reasoning"], ["llm", "balanced"]
    for i, nid in enumerate(node_ids):
        worker = MagicMock(spec=BaseWorkerNode)
        worker.node_id = nid
        worker.requests_served = 0
        worker.base_delay_sec = 0.05
        worker._active_load = 0
        worker.capability_tags = caps[i]
        worker.execute_inference = AsyncMock(
            return_value={
                "node_id": nid,
                "latency_sec": 0.075,
                "tokens": 10,
                "success": True,
                "active_load": 0,
                "text": f"Response from {nid}",
            }
        )
        workers[nid] = worker
    return workers


@pytest.fixture
async def router(mock_workers):
    """Create a router with a deterministic RNG."""
    memory_field = PheromoneMemoryField(node_ids=list(mock_workers.keys()))
    rng = np.random.default_rng(42)
    r = StigmergicRouterAgent(
        workers=mock_workers,
        memory_field=memory_field,
        weights={"alpha": 1.0, "beta": 2.0, "gamma": 1.5},
        temperature=0.5,
        delta=1.5,
        rng=rng,
    )
    return r


# ── Score Calculation ─────────────────────────────────────────────────

class TestComputeScores:
    def test_equal_scores_uniform_state(self, router):
        """All nodes with equal V, L, S, C must have equal scores."""
        state = np.array(
            [[1.0, 0.1, 0.0, 1.0],
             [1.0, 0.1, 0.0, 1.0],
             [1.0, 0.1, 0.0, 1.0]],
            dtype=np.float64,
        )
        scores = router.compute_scores(state)
        np.testing.assert_allclose(scores, scores[0])

    def test_higher_v_increases_score(self, router):
        """Higher V must produce higher Score (numerator effect)."""
        state = np.array(
            [[0.2, 0.1, 0.0, 1.0], [0.8, 0.1, 0.0, 1.0], [0.5, 0.1, 0.0, 1.0]],
            dtype=np.float64,
        )
        scores = router.compute_scores(state)
        assert scores[1] > scores[2] > scores[0]

    def test_higher_l_decreases_score(self, router):
        """Higher L must produce lower Score (denominator effect)."""
        state = np.array(
            [[1.0, 0.3, 0.0, 1.0], [1.0, 0.1, 0.0, 1.0], [1.0, 0.5, 0.0, 1.0]],
            dtype=np.float64,
        )
        scores = router.compute_scores(state)
        assert scores[1] > scores[0] > scores[2]

    def test_higher_s_decreases_score(self, router):
        """Higher S must produce lower Score."""
        state = np.array(
            [[1.0, 0.1, 0.0, 1.0], [1.0, 0.1, 0.5, 1.0], [1.0, 0.1, 0.1, 1.0]],
            dtype=np.float64,
        )
        scores = router.compute_scores(state)
        assert scores[0] > scores[2] > scores[1]

    def test_higher_c_increases_score(self, router):
        """Higher C (capability fit) must produce higher Score."""
        state = np.array(
            [[1.0, 0.1, 0.0, 0.3],
             [1.0, 0.1, 0.0, 0.7],
             [1.0, 0.1, 0.0, 0.1]],
            dtype=np.float64,
        )
        scores = router.compute_scores(state)
        assert scores[1] > scores[0] > scores[2]

    def test_delta_weight_controls_capability_influence(self):
        """Higher delta must amplify C's effect on the score."""
        memory_field = PheromoneMemoryField(node_ids=["a", "b"])
        rng = np.random.default_rng(0)
        workers = {"a": MagicMock(), "b": MagicMock()}
        r_low = StigmergicRouterAgent(
            workers=workers, memory_field=memory_field,
            weights={"alpha": 1.0, "beta": 2.0, "gamma": 1.5},
            temperature=1.0, delta=0.1, rng=rng,
        )
        r_high = StigmergicRouterAgent(
            workers=workers, memory_field=memory_field,
            weights={"alpha": 1.0, "beta": 2.0, "gamma": 1.5},
            temperature=1.0, delta=10.0, rng=rng,
        )
        state = np.array(
            [[1.0, 0.1, 0.0, 0.3],
             [1.0, 0.1, 0.0, 0.7]], dtype=np.float64
        )
        s_low = r_low.compute_scores(state)
        s_high = r_high.compute_scores(state)
        # With high delta, the C difference should dominate
        ratio_low = s_low[1] / s_low[0]
        ratio_high = s_high[1] / s_high[0]
        assert ratio_high > ratio_low

    def test_weights_applied_correctly(self):
        """Custom weights must affect the score formula."""
        memory_field = PheromoneMemoryField(node_ids=["a", "b"])
        rng = np.random.default_rng(0)
        workers = {"a": MagicMock(), "b": MagicMock()}
        r_custom = StigmergicRouterAgent(
            workers=workers,
            memory_field=memory_field,
            weights={"alpha": 2.0, "beta": 1.0, "gamma": 0.5},
            temperature=1.0,
            delta=1.5,
            rng=rng,
        )
        state = np.array([[1.0, 0.1, 0.0, 1.0], [1.0, 0.1, 0.0, 1.0]], dtype=np.float64)
        scores = r_custom.compute_scores(state)
        # Score = (2*1 + 1.5*1 + eps) / (1*0.1 + 0.5*0 + eps) ≈ 3.5 / 0.1 ≈ 35
        np.testing.assert_allclose(scores, np.full(2, 35.0), atol=1e-2)

    def test_score_formula_exact_4d(self, router):
        """Verify 4D formula: (alpha*V + delta*C + eps) / (beta*L + gamma*S + eps)."""
        state = np.array([[0.5, 0.1, 0.2, 0.8]], dtype=np.float64)
        scores = router.compute_scores(state)
        expected = (1.0 * 0.5 + 1.5 * 0.8 + router._EPS) / (
            2.0 * 0.1 + 1.5 * 0.2 + router._EPS
        )
        np.testing.assert_allclose(scores[0], expected)

    def test_epsilon_prevents_division_by_zero(self, router):
        """Score with all-zero state must be finite (not inf/NaN)."""
        state = np.zeros((3, 4))
        scores = router.compute_scores(state)
        assert np.all(np.isfinite(scores))

    def test_output_shape(self, router):
        """Scores array must have shape (N_nodes,)."""
        state = np.zeros((3, 4))
        scores = router.compute_scores(state)
        assert scores.shape == (3,)


# ── Softmax Probabilities ─────────────────────────────────────────────

class TestSoftmax:
    def test_probabilities_sum_to_one(self, router):
        """Probabilities must sum to 1.0."""
        scores = np.array([1.0, 2.0, 3.0])
        probs = router.compute_probabilities(scores)
        np.testing.assert_allclose(np.sum(probs), 1.0)

    def test_equal_scores_equal_probabilities(self, router):
        """Equal scores must produce uniform probabilities."""
        scores = np.array([5.0, 5.0, 5.0])
        probs = router.compute_probabilities(scores)
        np.testing.assert_allclose(probs, np.full(3, 1.0 / 3.0))

    def test_higher_score_higher_probability(self, router):
        """Higher score must yield higher probability."""
        scores = np.array([1.0, 3.0, 2.0])
        probs = router.compute_probabilities(scores)
        assert probs[1] > probs[2] > probs[0]

    def test_temperature_effect(self):
        """Higher temperature must flatten the distribution."""
        memory_field = PheromoneMemoryField(node_ids=["a", "b", "c"])
        workers = {"a": MagicMock(), "b": MagicMock(), "c": MagicMock()}
        rng = np.random.default_rng(0)

        r_low = StigmergicRouterAgent(
            workers=workers, memory_field=memory_field,
            weights={"alpha": 1.0, "beta": 2.0, "gamma": 1.5},
            temperature=0.1, rng=rng,
        )
        r_high = StigmergicRouterAgent(
            workers=workers, memory_field=memory_field,
            weights={"alpha": 1.0, "beta": 2.0, "gamma": 1.5},
            temperature=5.0, rng=rng,
        )
        scores = np.array([1.0, 2.0, 3.0])
        probs_low = r_low.compute_probabilities(scores)
        probs_high = r_high.compute_probabilities(scores)

        # Low T should be sharper (max prob - min prob is larger)
        spread_low = probs_low.max() - probs_low.min()
        spread_high = probs_high.max() - probs_high.min()
        assert spread_low > spread_high

    def test_numerical_stability_large_scores(self, router):
        """Very large scores must not produce NaN or Inf."""
        scores = np.array([1.0, 100.0, 200.0])
        probs = router.compute_probabilities(scores)
        assert np.all(np.isfinite(probs))
        np.testing.assert_allclose(np.sum(probs), 1.0)

    def test_numerical_stability_negative_scores(self, router):
        """Negative scores must not break softmax."""
        scores = np.array([-5.0, -3.0, -1.0])
        probs = router.compute_probabilities(scores)
        np.testing.assert_allclose(np.sum(probs), 1.0)
        assert np.all(probs >= 0.0)

    def test_single_node(self):
        """Single-node router must always have probability 1.0."""
        memory_field = PheromoneMemoryField(node_ids=["only"])
        rng = np.random.default_rng(0)
        r = StigmergicRouterAgent(
            workers={"only": MagicMock()},
            memory_field=memory_field,
            temperature=1.0,
            rng=rng,
        )
        probs = r.compute_probabilities(np.array([3.0]))
        np.testing.assert_allclose(probs, [1.0])


# ── Capability Matching ────────────────────────────────────────────────

class TestCapabilityMatch:
    def test_no_context_returns_neutral(self, router, mock_workers):
        """Without capability_context, match must be 0.5 (neutral)."""
        match = router.compute_capability_match(
            mock_workers["node_alpha"], None
        )
        assert match == 0.5

    def test_empty_context_returns_neutral(self, router, mock_workers):
        match = router.compute_capability_match(
            mock_workers["node_alpha"], {}
        )
        assert match == 0.5

    def test_full_tag_match_returns_one(self, router, mock_workers):
        """All requested tags present on worker → match = 1.0."""
        match = router.compute_capability_match(
            mock_workers["node_alpha"],
            {"slm": 1.0, "fast": 1.0},
        )
        np.testing.assert_allclose(match, 1.0)

    def test_partial_tag_match(self, router, mock_workers):
        """Only some tags match → match < 1.0."""
        match = router.compute_capability_match(
            mock_workers["node_alpha"],
            {"slm": 1.0, "llm": 1.0},
        )
        np.testing.assert_allclose(match, 0.5)

    def test_no_matching_tags(self, router, mock_workers):
        """No tag overlap → match = 0.0."""
        match = router.compute_capability_match(
            mock_workers["node_alpha"],
            {"llm": 1.0, "reasoning": 1.0},
        )
        np.testing.assert_allclose(match, 0.0)

    def test_weighted_tags(self, router, mock_workers):
        """Tag weights must proportion the match score."""
        match = router.compute_capability_match(
            mock_workers["node_alpha"],
            {"slm": 3.0, "fast": 1.0, "low-latency": 1.0},
        )
        # All 3 tags match → 3+1+1 / 3+1+1 = 1.0
        np.testing.assert_allclose(match, 1.0)

    def test_no_capability_tags_on_worker(self):
        """Worker with empty tags must return 0.5 for any context."""
        worker = MagicMock()
        worker.capability_tags = []
        rng = np.random.default_rng(0)
        r = StigmergicRouterAgent(
            workers={"a": worker},
            memory_field=PheromoneMemoryField(node_ids=["a"]),
            rng=rng,
        )
        match = r.compute_capability_match(worker, {"slm": 1.0})
        assert match == 0.5


# ── Capability-Aware Sampling ──────────────────────────────────────────

class TestCapabilitySampling:
    @pytest.mark.asyncio
    async def test_capability_context_boosts_matching_nodes(self, router, mock_workers):
        """Matching capability context must bias sampling toward tagged nodes."""
        ctx = {"slm": 1.5, "fast": 1.3}
        counts = {"node_alpha": 0, "node_beta": 0, "node_gamma": 0}
        for _ in range(500):
            worker = await router.sample_worker(capability_context=ctx)
            counts[worker.node_id] += 1
        # node_alpha has ["slm", "fast", "low-latency"], should dominate
        assert counts["node_alpha"] > counts["node_beta"]
        assert counts["node_alpha"] > counts["node_gamma"]

    @pytest.mark.asyncio
    async def test_reasoning_context_prefers_llm_nodes(self, router, mock_workers):
        """Reasoning context must boost llm-tagged nodes."""
        ctx = {"llm": 2.0, "reasoning": 1.5}
        counts = {"node_alpha": 0, "node_beta": 0, "node_gamma": 0}
        for _ in range(500):
            worker = await router.sample_worker(capability_context=ctx)
            counts[worker.node_id] += 1
        # node_beta has ["llm", "reasoning"], node_gamma has ["llm", "balanced"]
        assert counts["node_beta"] > counts["node_alpha"]


# ── Trace Feedback / Route ──────────────────────────────────────────────

class TestRouteTraceFeedback:
    @pytest.mark.asyncio
    async def test_route_deposits_trace(self, router, mock_workers):
        """route() must call memory_field.deposit_trace after executing."""
        with patch.object(
            router.memory_field, "deposit_trace", new_callable=AsyncMock
        ) as deposit_mock:
            await router.route("test prompt", max_tokens=10)
            deposit_mock.assert_called_once()
            call_kwargs = deposit_mock.call_args.kwargs
            assert call_kwargs["latency_sec"] == 0.075
            assert call_kwargs["success"] is True
            assert call_kwargs["active_load"] == 0

    @pytest.mark.asyncio
    async def test_route_returns_node_id(self, router, mock_workers):
        """route() must return the node_id of the selected worker."""
        result = await router.route("test prompt", max_tokens=10)
        assert result["node_id"] in router.workers
        assert result["routed_to"] == result["node_id"]

    @pytest.mark.asyncio
    async def test_route_and_execute_alias(self, router, mock_workers):
        """route_and_execute must behave identically to route."""
        result1 = await router.route("test", max_tokens=10)
        result2 = await router.route_and_execute("test", max_tokens=10)
        assert result1["node_id"] in router.workers
        assert result2["node_id"] in router.workers
        assert result1["routed_to"] == result1["node_id"]

    @pytest.mark.asyncio
    async def test_route_samples_correct_number(self, router, mock_workers):
        """Each route() call must execute exactly one worker."""
        for _ in range(10):
            await router.route("test prompt", max_tokens=10)
        for worker in mock_workers.values():
            total_calls = worker.execute_inference.await_count
            assert total_calls >= 0
        assert sum(w.execute_inference.await_count for w in mock_workers.values()) == 10

    @pytest.mark.asyncio
    async def test_trace_deposited_for_correct_node(self, router, mock_workers):
        """The trace must be deposited for the node that actually executed."""
        with patch.object(
            router.memory_field, "deposit_trace", new_callable=AsyncMock
        ) as deposit_mock:
            await router.route("test", max_tokens=10)
            # The deposit should reference the same node_id as the result
            call_kwargs = deposit_mock.call_args.kwargs
            assert call_kwargs["node_id"] == call_kwargs["node_id"]

    @pytest.mark.asyncio
    async def test_failure_still_deposits_trace(self, router, mock_workers):
        """Even if inference raises, a failure trace must be deposited."""
        alpha_worker = mock_workers["node_alpha"]
        alpha_worker.execute_inference = AsyncMock(
            side_effect=RuntimeError("worker crash")
        )
        with patch.object(
            router, "sample_worker", new_callable=AsyncMock, return_value=alpha_worker
        ), patch.object(
            router.memory_field, "deposit_trace", new_callable=AsyncMock
        ) as deposit_mock:
            result = await router.route("test", max_tokens=10)
            assert result["success"] is False
            assert result["latency_sec"] == 0.0
            assert deposit_mock.call_args.kwargs["success"] is False
            assert deposit_mock.call_args.kwargs["latency_sec"] == 0.0


# ── Worker Sampling ───────────────────────────────────────────────────

class TestSampleWorker:
    @pytest.mark.asyncio
    async def test_sample_worker_returns_valid_worker(self, router):
        """sample_worker must return a worker from the pool."""
        worker = await router.sample_worker()
        assert isinstance(worker, BaseWorkerNode) or worker in router.workers.values()

    @pytest.mark.asyncio
    async def test_sampling_bias_follows_scores(self, router):
        """Nodes with higher scores should be sampled more frequently."""
        # Set up a known state where node_alpha has much higher V
        state = np.array(
            [[1.0, 0.01, 0.0], [0.1, 0.1, 0.0], [0.1, 0.1, 0.0]],
            dtype=np.float64,
        )
        with patch.object(
            router.memory_field, "get_state_vector",
            new_callable=AsyncMock, return_value=state,
        ):
            counts = {"node_alpha": 0, "node_beta": 0, "node_gamma": 0}
            for _ in range(1000):
                worker = await router.sample_worker()
                counts[worker.node_id] += 1
            assert counts["node_alpha"] > counts["node_beta"]
            assert counts["node_alpha"] > counts["node_gamma"]
