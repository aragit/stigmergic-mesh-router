"""Tenant governance: map tier / SLA metadata to matrix-routing biases.

The stigmergic router scores each candidate worker with the extended 4D
attraction function::

    Score_i = (alpha * V_i + delta * C_i + eps) / (beta * L_i + gamma * S_i + eps)

where ``V`` (success), ``L`` (latency), ``S`` (saturation) and ``C``
(capability-fit / cost-efficiency) are the four pheromone traces.

:func:`TenantPolicyEngine.apply_governance_bias` returns an *override* weights
dictionary that the router consumes on a per-request basis so that free, pro,
and enterprise tenants are routed through differently-biased preference
matrices without mutating shared router state.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

#: The four canonical 4D matrix weights.
_MATRIX_WEIGHT_KEYS = ("alpha", "beta", "gamma", "delta")


class TenantPolicyEngine:
    """Map :class:`~api.auth.TenantContext` tier values to weight overrides."""

    def __init__(self) -> None:
        # Per-tier multipliers.  ``None`` means "leave unchanged".
        self._tier_multipliers: Dict[str, Dict[str, float]] = {
            # Enterprise: penalise latency (w_L up) and prioritise
            # high-throughput performers (w_V up).
            "enterprise": {"beta": 1.5, "alpha": 1.2},
            # Free: penalise cost by boosting capability/cost-efficiency
            # (w_C up) and cap concurrency harder (lowered saturation weight
            # so loaded nodes are pushed away more aggressively).
            "free": {"delta": 2.0, "gamma": 2.0},
            # Pro: balanced — no adjustment.
            "pro": {},
        }

    @staticmethod
    def _normalize_tier(tenant: Any) -> str:
        tier = getattr(tenant, "tier", None) or "pro"
        if tier not in ("free", "pro", "enterprise"):
            logger.warning("Unknown tenant tier %r; defaulting to 'pro'", tier)
            tier = "pro"
        return tier

    def apply_governance_bias(
        self,
        tenant: Any,
        matrix_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Return a *copy* of *matrix_weights* biased by the tenant tier.

        Parameters
        ----------
        tenant
            An object exposing a ``tier`` attribute (typically a
            :class:`~api.auth.TenantContext`).
        matrix_weights
            The base 4D weights, e.g. ``{"alpha": 1.0, "beta": 2.0, ...}``.

        Returns
        -------
        dict
            A new dict with the same keys, values scaled per the tier policy.
            Only ``alpha``, ``beta``, ``gamma`` and ``delta`` are honoured;
            unknown keys are passed through unchanged.
        """
        tier = self._normalize_tier(tenant)
        multipliers = self._tier_multipliers.get(tier, {})
        biased = dict(matrix_weights)
        for key in _MATRIX_WEIGHT_KEYS:
            if key in biased:
                biased[key] = biased[key] * multipliers.get(key, 1.0)
        return biased
