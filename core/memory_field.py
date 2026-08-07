"""Pheromone memory field for stigmergic routing."""

import asyncio
from typing import Dict, List, Optional

import numpy as np


class PheromoneMemoryField:
    """Async-safe 2D pheromone memory field for stigmergic routing.

    Maintains a state matrix of shape (N_nodes, 3) where each row encodes
    the pheromone traces for a single node:

    Column 0 — V (Success): exponentially-weighted success signal in [0, 1].
        Higher values indicate a node with a strong record of successful
        inferences and are positively correlated with selection probability.

    Column 1 — L (Latency): exponentially-weighted moving average of observed
        latency in seconds. Lower values are better; high latency penalises
        the node's attraction score.

    Column 2 — S (Saturation): normalized residual load on the node at the
        time a trace is deposited.  Higher saturation discourages further
        traffic (load balancing via negative feedback).  Values are scaled
        by *saturation_scale* to keep them comparable with latency in the
        score formula.
    """

    def __init__(
        self,
        node_ids: List[str],
        initial_state: Optional[np.ndarray] = None,
        saturation_scale: float = 0.1,
    ) -> None:
        """Initialise the field with *node_ids* and an optional initial state.

        Parameters
        ----------
        node_ids
            Ordered list of worker node identifiers.
        initial_state
            Optional ``(N_nodes, 3)`` array to seed the field.  When
            *None* the field starts neutral (all zeros) so that the
            Boltzmann distribution is initially uniform.
        saturation_scale
            Scaling factor applied to the *active_load* parameter when
            writing to the S column.  A value of 0.1 (the default) maps
            a residual load of 1 to S = 0.1, which is comparable to
            typical latency values (~0.05-0.3 s) and prevents S from
            dominating the attraction score.
        """
        self.node_ids: List[str] = list(node_ids)
        self._node_index: Dict[str, int] = {
            nid: i for i, nid in enumerate(self.node_ids)
        }
        self.saturation_scale: float = saturation_scale
        n_nodes = len(self.node_ids)

        if initial_state is not None:
            self._state: np.ndarray = np.asarray(
                initial_state, dtype=np.float64
            ).copy()
            if self._state.shape != (n_nodes, 3):
                raise ValueError(
                    f"initial_state shape {self._state.shape} "
                    f"does not match expected ({n_nodes}, 3)"
                )
        else:
            # Neutral baseline: zero pheromone everywhere so that all
            # nodes start with equal Boltzmann probability.
            self._state = np.zeros((n_nodes, 3), dtype=np.float64)

        self._lock = asyncio.Lock()

    @property
    def n_nodes(self) -> int:
        """Number of nodes tracked by this field."""
        return len(self.node_ids)

    def _index_of(self, node_id: str) -> int:
        """Return the row index for *node_id*."""
        if node_id not in self._node_index:
            raise KeyError(f"Unknown node_id: {node_id!r}")
        return self._node_index[node_id]

    async def get_state_vector(self) -> np.ndarray:
        """Return a defensive copy of the full state matrix ``(N_nodes, 3)``."""
        async with self._lock:
            return self._state.copy()

    async def deposit_trace(
        self,
        node_id: str,
        latency_sec: float,
        tokens: int,
        success: bool,
        active_load: int,
    ) -> None:
        """Update pheromone traces for *node_id* after a completed inference.

        Updates applied to the state matrix row for *node_id*:

        * **V (Success)** — EWMA update toward the binary success indicator.
          ``V_new = 0.5 * success + 0.5 * V_old`` keeping V ∈ [0, 1].
        * **L (Latency)** — EWMA update toward the observed latency.
          ``L_new = 0.5 * latency + 0.5 * L_old``.
        * **S (Saturation)** — scaled assignment from *active_load*.
          ``S = active_load * saturation_scale``.  Using the residual load
          (reported by the worker after completion) means S = 0 when the
          node is idle, preserving its attractiveness when healthy.

        Parameters
        ----------
        node_id
            Identifier of the node that completed the request.
        latency_sec
            Wall-clock latency of the inference in seconds.
        tokens
            Number of tokens generated (accepted for future weighting;
            currently not written to the 3-column state matrix).
        success
            Whether the inference completed successfully.
        active_load
            Residual concurrent load on the node after this request
            completed (proxy for saturation).
        """
        async with self._lock:
            idx = self._index_of(node_id)

            success_val = 1.0 if success else 0.0
            alpha = 0.5  # EWMA smoothing factor

            # V: success signal EWMA
            self._state[idx, 0] = alpha * success_val + (1.0 - alpha) * self._state[idx, 0]

            # L: latency EWMA
            self._state[idx, 1] = alpha * latency_sec + (1.0 - alpha) * self._state[idx, 1]

            # S: normalized saturation — scaled to be comparable with L
            self._state[idx, 2] = float(active_load) * self.saturation_scale

    async def apply_evaporation(self, decay_rate: float) -> None:
        """Apply multiplicative evaporation to every trace.

        All pheromone values are scaled by ``(1 - decay_rate)``, causing
        stale traces to fade over time so that the field reflects recent
        performance rather than ancient history.

        Parameters
        ----------
        decay_rate
            Fraction of the current value to evaporate per step.
            Must satisfy ``0 <= decay_rate < 1``.
        """
        if not 0.0 <= decay_rate < 1.0:
            raise ValueError("decay_rate must be in [0, 1)")
        async with self._lock:
            self._state *= (1.0 - decay_rate)
