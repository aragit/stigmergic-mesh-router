"""Stigmergic router agent — samples workers via Boltzmann softmax over pheromone scores.

Extends the classic 3-trace scoring model (V, L, S) with a **capability-fit**
trace (C) that biases routing toward nodes whose declared capability tags
match the semantics of the incoming request:

    Score_i = (alpha * V_i + delta * C_i + eps) / (beta * L_i + gamma * S_i + eps)

The router accepts an optional *capability_context* dictionary on each
``route()`` call.  This context maps capability tags (e.g. ``"slm"``,
``"reasoning"``, ``"low-latency"``) to importance weights.  The router
computes a *capability_match* for each candidate node by intersecting
its declared tags with the requested context, then deposits that match
into the shared memory field so that future routing decisions are
informed by both performance and capability alignment.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from .memory_field import PheromoneMemoryField
from .worker_node import BaseWorkerNode


class StigmergicRouterAgent:
    """Decentralised load router using stigmergic pheromone feedback.

    The router periodically consults the shared ``PheromoneMemoryField`` to
    compute an *attraction score* for every node, converts scores to
    selection probabilities via a Boltzmann (softmax) distribution, and
    dispatches the request to the sampled worker.

    **Attraction score (extended with capability)**

    For node *i* with state ``[V_i, L_i, S_i, C_i]``::

        Score_i = (alpha * V_i + delta * C_i + eps) / (beta * L_i + gamma * S_i + eps)

    where ``eps = 1e-5`` prevents division by zero and ``delta`` controls
    the weight of the capability-fit trace relative to success.

    **Boltzmann softmax**

    Probabilities are derived from scores using temperature *T*::

        P_i = exp(Score_i / T) / sum_j exp(Score_j / T)

    A lower temperature yields sharper (more greedy) selection, while a
    higher temperature encourages exploration.
    """

    _EPS: float = 1e-5

    def __init__(
        self,
        workers: Dict[str, BaseWorkerNode],
        memory_field: PheromoneMemoryField,
        weights: Optional[Dict[str, float]] = None,
        temperature: float = 0.5,
        delta: float = 1.5,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.workers: Dict[str, BaseWorkerNode] = workers
        self.memory_field: PheromoneMemoryField = memory_field
        self.alpha: float = weights.get("alpha", 1.0) if weights else 1.0
        self.beta: float = weights.get("beta", 2.0) if weights else 2.0
        self.gamma: float = weights.get("gamma", 1.5) if weights else 1.5
        self.delta: float = delta
        self.temperature: float = temperature
        self.rng: np.random.Generator = rng or np.random.default_rng()

    def compute_scores(self, state: np.ndarray) -> np.ndarray:
        """Compute attraction scores for all nodes from the 4D state matrix.

        Parameters
        ----------
        state
            ``(N_nodes, 4)`` matrix with columns ``[V, L, S, C]``.

        Returns
        -------
        np.ndarray
            1-D array of shape ``(N_nodes,)`` with attraction scores.
        """
        v = state[:, 0]
        l = state[:, 1]
        s = state[:, 2]
        c = state[:, 3] if state.shape[1] >= 4 else np.zeros_like(v)
        return (self.alpha * v + self.delta * c + self._EPS) / (
            self.beta * l + self.gamma * s + self._EPS
        )

    def compute_capability_match(
        self,
        worker: BaseWorkerNode,
        capability_context: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute a normalised capability-fit score in [0, 1].

        The match is the weighted fraction of requested capability tags
        that the worker actually supports:

            match = sum(adjustment[t] for t in context if t in worker.tags)
                    / sum(adjustment[t] for t in context)

        Returns 0.5 (neutral) when no capability context is provided.
        """
        if not capability_context:
            return 0.5

        tags = getattr(worker, "capability_tags", [])
        if not tags:
            return 0.5

        total_weight = sum(capability_context.values())
        if total_weight <= 0.0:
            return 0.5

        matched = sum(
            w for t, w in capability_context.items() if t in tags
        )
        return min(1.0, matched / total_weight)

    def compute_probabilities(self, scores: np.ndarray) -> np.ndarray:
        """Apply Boltzmann softmax with the configured temperature.

        Numerically stable implementation: the maximum score is subtracted
        before exponentiation to prevent overflow.

        Parameters
        ----------
        scores
            1-D array of raw attraction scores.

        Returns
        -------
        np.ndarray
            1-D array of probabilities summing to 1.0.
        """
        scaled = scores / self.temperature
        shifted = scaled - np.max(scaled)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores)

    async def sample_worker(
        self,
        capability_context: Optional[Dict[str, float]] = None,
    ) -> BaseWorkerNode:
        """Sample a worker node according to stigmergic pheromone probabilities.

        Reads the current state vector, computes scores and probabilities,
        and performs a weighted random draw.

        Parameters
        ----------
        capability_context
            Optional mapping of capability tags to importance weights.
            When provided, the scores are adjusted on-the-fly by boosting
            nodes whose tags match the context.

        Returns
        -------
        BaseWorkerNode
            The selected worker node.
        """
        state = await self.memory_field.get_state_vector()
        scores = self.compute_scores(state)

        if capability_context:
            boosts = np.array(
                [
                    self.compute_capability_match(w, capability_context)
                    for w in self.workers.values()
                ]
            )
            scores = scores * boosts

        probs = self.compute_probabilities(scores)
        node_ids: List[str] = self.memory_field.node_ids
        chosen_idx = int(self.rng.choice(len(node_ids), p=probs))
        chosen_id = node_ids[chosen_idx]
        return self.workers[chosen_id]

    async def route(
        self,
        prompt: str,
        max_tokens: int = 128,
        capability_context: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Route a single inference request through the stigmergic mechanism.

        1. Sample a worker based on current pheromone state and optional
           capability context.
        2. Execute inference on the selected worker.
        3. Compute capability match and deposit a trace into the memory
           field so that future routing decisions are influenced by
           this outcome.

        Parameters
        ----------
        prompt
            Input prompt for the inference.
        max_tokens
            Maximum tokens to generate.
        capability_context
            Optional mapping of capability tags to importance weights.

        Returns
        -------
        Dict[str, Any]
            The inference result enriched with ``routed_to`` (the node_id
            that handled the request).
        """
        worker = await self.sample_worker(capability_context)
        try:
            result: Dict[str, Any] = await worker.execute_inference(
                prompt, max_tokens
            )
        except Exception:
            result = {
                "node_id": worker.node_id,
                "latency_sec": 0.0,
                "tokens": 0,
                "success": False,
                "active_load": 0,
            }

        cap_match = self.compute_capability_match(worker, capability_context)
        await self.memory_field.deposit_trace(
            node_id=result["node_id"],
            latency_sec=result["latency_sec"],
            tokens=result["tokens"],
            success=result["success"],
            active_load=result["active_load"],
            capability_match=cap_match,
        )
        result["routed_to"] = result["node_id"]
        return result

    async def route_and_execute(
        self,
        prompt: str,
        max_tokens: int = 128,
        capability_context: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Sample a worker, execute inference, and deposit a trace.

        Alias for :meth:`route` with capability context support.
        """
        return await self.route(prompt, max_tokens, capability_context)
