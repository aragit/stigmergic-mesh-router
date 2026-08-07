"""Pheromone memory field for stigmergic routing.

Maintains a state matrix of shape ``(N_nodes, 4)`` where each row encodes the
pheromone traces for a single worker node:

Column 0 — V (Success): exponentially-weighted success signal in [0, 1].
    Higher values indicate a node with a strong record of successful
    inferences and are positively correlated with selection probability.

Column 1 — L (Latency): exponentially-weighted moving average of observed
    latency in seconds. Lower values are better; high latency penalises
    the node's attraction score.

Column 2 — S (Saturation): normalized residual load on the node at the
    time a trace is deposited. Higher saturation discourages further
    traffic (load balancing via negative feedback). Values are scaled
    by *saturation_scale* to keep them comparable with latency in the
    score formula.

Column 3 — C (Capability Fit): exponentially-weighted record of how well
    the node's declared capability tags match the requests routed to it.
    A value near 1.0 means the node consistently handles requests that
    match its declared capabilities; lower values indicate a mismatch.
    This trace is used in the extended attraction score:

        Score = (alpha * V + delta * C + eps) / (beta * L + gamma * S + eps)
"""

import asyncio
from typing import Dict, List, Optional

import numpy as np


class PheromoneMemoryField:
    """Async-safe 2D pheromone memory field for stigmergic routing.

    Tracks four pheromone traces per node: **V** (success), **L** (latency),
    **S** (saturation), and **C** (capability fit).  All updates are guarded
    by an :class:`asyncio.Lock` to ensure correctness under concurrent
    access from multiple router agents or background tasks.
    """

    _DEFAULT_STATE = np.array([1.0, 0.1, 0.0, 1.0], dtype=np.float64)

    def __init__(
        self,
        node_ids: List[str],
        initial_state: Optional[np.ndarray] = None,
        saturation_scale: float = 0.1,
        decay_success: bool = True,
        decay_capability: bool = True,
    ) -> None:
        """Initialise the field with *node_ids* and an optional initial state.

        Parameters
        ----------
        node_ids
            Ordered list of worker node identifiers.
        initial_state
            Optional ``(N_nodes, 4)`` array to seed the field.  When
            *None* the field starts from the baseline ``[V=1.0, L=0.1,
            S=0.0, C=1.0]`` so that all nodes begin with equal attraction
            probability.
        saturation_scale
            Scaling factor applied to the *active_load* parameter when
            writing to the S column.  A value of 0.1 (the default) maps
            a residual load of 1 to S = 0.1, which is comparable to
            typical latency values (~0.05-0.3 s) and prevents S from
            dominating the attraction score.
        decay_success
            When *True* (default) the V column (success) is also scaled
            by ``(1 - decay_rate)`` during evaporation, matching the
            classic stigmergic model.  When *False*, V is treated as a
            persistent quality metric that does **not** evaporate.
        decay_capability
            When *True* (default) the C column (capability fit) also
            decays.  Set to *False* to keep capability scores persistent.
        """
        self.node_ids: List[str] = list(node_ids)
        self._node_index: Dict[str, int] = {
            nid: i for i, nid in enumerate(self.node_ids)
        }
        self.saturation_scale: float = saturation_scale
        self.decay_success: bool = decay_success
        self.decay_capability: bool = decay_capability
        n_nodes = len(self.node_ids)

        if initial_state is not None:
            self._state: np.ndarray = np.asarray(
                initial_state, dtype=np.float64
            ).copy()
            if self._state.shape != (n_nodes, 4):
                raise ValueError(
                    f"initial_state shape {self._state.shape} "
                    f"does not match expected ({n_nodes}, 4)"
                )
        else:
            self._state = np.tile(
                self._DEFAULT_STATE, (n_nodes, 1)
            ).astype(np.float64)

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
        """Return a defensive copy of the full state matrix ``(N_nodes, 4)``."""
        async with self._lock:
            return self._state.copy()

    async def deposit_trace(
        self,
        node_id: str,
        latency_sec: float,
        tokens: int,
        success: bool,
        active_load: int,
        capability_match: float = 0.5,
    ) -> None:
        """Update pheromone traces for *node_id* after a completed inference.

        Updates applied to the state matrix row for *node_id*:

        * **V (Success)** — EWMA update toward the binary success indicator.
          ``V_new = 0.5 * success + 0.5 * V_old`` keeping V ∈ [0, 1].
        * **L (Latency)** — EWMA update toward the observed latency.
          ``L_new = 0.5 * latency + 0.5 * L_old``.
        * **S (Saturation)** — scaled assignment from *active_load*.
          ``S = active_load * saturation_scale``.
        * **C (Capability Fit)** — EWMA update toward *capability_match*
          (a value in [0, 1] indicating how well the node's declared
          capability tags matched the request).

        Parameters
        ----------
        node_id
            Identifier of the node that completed the request.
        latency_sec
            Wall-clock latency of the inference in seconds.
        tokens
            Number of tokens generated (accepted for future weighting;
            currently not written to the 4-column state matrix).
        success
            Whether the inference completed successfully.
        active_load
            Residual concurrent load on the node after this request
            completed (proxy for saturation).
        capability_match
            How well the node's capability tags matched the request,
            in [0, 1].  Default 0.5 is neutral.
        """
        async with self._lock:
            idx = self._index_of(node_id)

            success_val = 1.0 if success else 0.0
            alpha = 0.5  # EWMA smoothing factor

            # V: success signal EWMA
            self._state[idx, 0] = (
                alpha * success_val + (1.0 - alpha) * self._state[idx, 0]
            )

            # L: latency EWMA
            self._state[idx, 1] = (
                alpha * latency_sec + (1.0 - alpha) * self._state[idx, 1]
            )

            # S: normalized saturation — scaled to be comparable with L
            self._state[idx, 2] = float(active_load) * self.saturation_scale

            # C: capability fit EWMA
            cap = max(0.0, min(1.0, float(capability_match)))
            self._state[idx, 3] = (
                alpha * cap + (1.0 - alpha) * self._state[idx, 3]
            )

    async def apply_evaporation(self, decay_rate: float) -> None:
        """Apply multiplicative evaporation to every trace.

        Columns L (latency), S (saturation), and — when
        ``decay_capability`` is *True* — C (capability fit) are always
        scaled by ``(1 - decay_rate)``, causing stale observations to fade.

        When ``decay_success`` is *True* (the default), the V (success)
        column is also scaled, following the classic stigmergic model.
        When *False*, V is left untouched.

        Parameters
        ----------
        decay_rate
            Fraction of the current value to evaporate per step.
            Must satisfy ``0 <= decay_rate < 1``.
        """
        if not 0.0 <= decay_rate < 1.0:
            raise ValueError("decay_rate must be in [0, 1)")
        async with self._lock:
            if self.decay_success:
                self._state[:, 0] *= (1.0 - decay_rate)
            self._state[:, 1] *= (1.0 - decay_rate)
            self._state[:, 2] *= (1.0 - decay_rate)
            if self.decay_capability:
                self._state[:, 3] *= (1.0 - decay_rate)
