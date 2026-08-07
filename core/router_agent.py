"""Stigmergic router agent — samples workers via Boltzmann softmax over pheromone scores."""

import asyncio
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

    **Attraction score**

    For node *i* with state ``[V_i, L_i, S_i]``::

        Score_i = (alpha * V_i + eps) / (beta * L_i + gamma * S_i + eps)

    where ``eps = 1e-5`` prevents division by zero.  A node with a high
    success trace *V* and low latency *L* / saturation *S* receives a
    high score and is therefore more likely to be selected.

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
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.workers: Dict[str, BaseWorkerNode] = workers
        self.memory_field: PheromoneMemoryField = memory_field
        self.alpha: float = weights.get("alpha", 1.0) if weights else 1.0
        self.beta: float = weights.get("beta", 2.0) if weights else 2.0
        self.gamma: float = weights.get("gamma", 1.5) if weights else 1.5
        self.temperature: float = temperature
        self.rng: np.random.Generator = rng or np.random.default_rng()

    def compute_scores(self, state: np.ndarray) -> np.ndarray:
        """Compute attraction scores for all nodes from the state matrix.

        Parameters
        ----------
        state
            ``(N_nodes, 3)`` matrix with columns ``[V, L, S]``.

        Returns
        -------
        np.ndarray
            1-D array of shape ``(N_nodes,)`` with attraction scores.
        """
        v = state[:, 0]
        l = state[:, 1]
        s = state[:, 2]
        return (self.alpha * v + self._EPS) / (
            self.beta * l + self.gamma * s + self._EPS
        )

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

    async def sample_worker(self) -> BaseWorkerNode:
        """Sample a worker node according to stigmergic pheromone probabilities.

        Reads the current state vector, computes scores and probabilities,
        and performs a weighted random draw with ``np.random.choice``.

        Returns
        -------
        BaseWorkerNode
            The selected worker node.
        """
        state = await self.memory_field.get_state_vector()
        scores = self.compute_scores(state)
        probs = self.compute_probabilities(scores)
        node_ids: List[str] = self.memory_field.node_ids
        chosen_idx = int(self.rng.choice(len(node_ids), p=probs))
        chosen_id = node_ids[chosen_idx]
        return self.workers[chosen_id]

    async def route(
        self,
        prompt: str,
        max_tokens: int = 128,
    ) -> Dict[str, Any]:
        """Route a single inference request through the stigmergic mechanism.

        1. Sample a worker based on current pheromone state.
        2. Execute inference on the selected worker.
        3. Deposit a trace back into the memory field so that future
           routing decisions are influenced by this outcome.

        Parameters
        ----------
        prompt
            Input prompt for the inference.
        max_tokens
            Maximum tokens to generate.

        Returns
        -------
        Dict[str, Any]
            The inference result enriched with ``routed_to`` (the node_id
            that handled the request).
        """
        worker = await self.sample_worker()
        try:
            result: Dict[str, Any] = await worker.execute_inference(prompt, max_tokens)
        except Exception:
            result = {
                "node_id": worker.node_id,
                "latency_sec": 0.0,
                "tokens": 0,
                "success": False,
                "active_load": 0,
            }
        await self.memory_field.deposit_trace(
            node_id=result["node_id"],
            latency_sec=result["latency_sec"],
            tokens=result["tokens"],
            success=result["success"],
            active_load=result["active_load"],
        )
        result["routed_to"] = result["node_id"]
        return result

    async def route_and_execute(
        self,
        prompt: str,
        max_tokens: int = 128,
    ) -> Dict[str, Any]:
        """Sample a worker, execute inference, and deposit a trace.

        This is an alias for :meth:`route` providing a more descriptive
        name for callers that want to emphasise the full execute-then-record
        lifecycle.
        """
        return await self.route(prompt, max_tokens)
