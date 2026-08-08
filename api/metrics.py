"""Prometheus observability metrics for the stigmergic mesh router.

Declares global Prometheus Gauges, Counters, and Histograms that mirror
the 4D pheromone trace matrix (``V, L, S, C``), per-node attraction scores,
request totals, and execution-latency distributions.

The :func:`update_prometheus_metrics` coroutine syncs the current memory
field state into the Gauges so that ``/metrics`` always reflects the latest
pheromone values when scraped.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
)

from core.memory_field import BasePheromoneMemoryField
from core.router_agent import StigmergicRouterAgent

logger = logging.getLogger(__name__)

# ── Gauge metrics — one per node, synced from the memory field ──────────────

stigmergic_node_success_trace = Gauge(
    "stigmergic_node_success_trace",
    "Success trace (V) for each node — EWMA of binary success",
    ["node_id"],
)

stigmergic_node_latency_trace = Gauge(
    "stigmergic_node_latency_trace",
    "Latency trace (L) for each node — EWMA of observed latency in seconds",
    ["node_id"],
)

stigmergic_node_saturation_trace = Gauge(
    "stigmergic_node_saturation_trace",
    "Saturation trace (S) for each node — normalised residual load",
    ["node_id"],
)

stigmergic_node_capability_trace = Gauge(
    "stigmergic_node_capability_trace",
    "Capability-fit trace (C) for each node — EWMA of capability match",
    ["node_id"],
)

stigmergic_node_attraction_score = Gauge(
    "stigmergic_node_attraction_score",
    "Calculated attraction score for each node",
    ["node_id"],
)

# ── Counter metrics ─────────────────────────────────────────────────────────

stigmergic_requests_total = Counter(
    "stigmergic_requests_total",
    "Total number of requests routed to each node, by status",
    ["node_id", "status"],
)

# ── Histogram metrics ───────────────────────────────────────────────────────

stigmergic_request_duration_seconds = Histogram(
    "stigmergic_request_duration_seconds",
    "Execution latency distribution per node in seconds",
    ["node_id"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

# ── Custom metrics for Kubernetes HPA via Prometheus Adapter ───────────────────

stigmergic_entropy_rate_total = Counter(
    "stigmergic_entropy_rate_total",
    "Cumulative stigmergic entropy decay events — drives HPA scaling",
)

stigmergic_auth_failures_total = Counter(
    "stigmergic_auth_failures_total",
    "Total count of unauthorized request attempts",
    ["reason"],
)

stigmergic_rate_limit_exceeded_total = Counter(
    "stigmergic_rate_limit_exceeded_total",
    "Total count of requests rejected due to rate limits",
    ["tenant_id", "dimension"],
)

stigmergic_checkpoints_created_total = Counter(
    "stigmergic_checkpoints_created_total",
    "Total count of matrix snapshots successfully saved",
    ["storage_type"],  # "redis" or "disk"
)

stigmergic_checkpoint_restore_status = Gauge(
    "stigmergic_checkpoint_restore_status",
    "1 if matrix successfully hydrated on boot, 0 if cold-started",
)


def get_total_routed_requests() -> int:
    """Return the cumulative number of routed requests across all labels.

    Best-effort: the underlying Counter is labeled, so we sum every child
    value.  Any access error yields ``0``.
    """
    try:
        total = 0
        for child in stigmergic_requests_total._metrics.values():
            total += int(child._value.get())
        return total
    except Exception:
        return 0

agent_active_mesh_routes = Gauge(
    "agent_active_mesh_routes",
    "Current count of active mesh routing slots per agent/pod",
)

routing_queue_latency_seconds = Histogram(
    "routing_queue_latency_seconds",
    "Total time requests spend queued before routing dispatch",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)


async def update_prometheus_metrics(
    memory_field: BasePheromoneMemoryField,
    router_agent: Optional[StigmergicRouterAgent] = None,
) -> None:
    """Sync the current memory-field state into Prometheus Gauges.

    Reads the full ``(N_nodes, 4)`` state vector, computes attraction
    scores via the router agent (if provided), and writes every value
    to the corresponding Gauge so that ``/metrics`` scrape reflects live
    trace values.

    Parameters
    ----------
    memory_field
        The shared memory field (in-memory or Redis-backed).
    router_agent
        Optional router agent used to compute attraction scores.  If
        *None*, the ``stigmergic_node_attraction_score`` gauge is left
        at its previous value.
    """
    try:
        state = await memory_field.get_state_vector()
        node_ids: List[str] = memory_field.node_ids

        for i, nid in enumerate(node_ids):
            v = float(state[i, 0])
            l = float(state[i, 1])
            s = float(state[i, 2])
            c = float(state[i, 3])

            stigmergic_node_success_trace.labels(node_id=nid).set(v)
            stigmergic_node_latency_trace.labels(node_id=nid).set(l)
            stigmergic_node_saturation_trace.labels(node_id=nid).set(s)
            stigmergic_node_capability_trace.labels(node_id=nid).set(c)

            if router_agent is not None:
                score = float(router_agent.compute_scores(state)[i])
                stigmergic_node_attraction_score.labels(node_id=nid).set(score)

        # ── Custom HPA metrics ────────────────────────────────────────────
        if router_agent is not None:
            entropy_delta = float(abs(np.diff(state[:, 0]).sum())) if len(node_ids) > 1 else 0.0
            stigmergic_entropy_rate_total.inc(entropy_delta)
            agent_active_mesh_routes.set(int(router_agent.active_swarm_size()))

        logger.debug("Prometheus metrics updated for %d nodes", len(node_ids))

    except Exception as exc:
        logger.warning("Failed to update Prometheus metrics: %s", exc)
