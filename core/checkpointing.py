"""Pheromone state checkpointing and cold-start warmup.

A :class:`CheckpointManager` periodically serialises the live 4D pheromone
matrix (``V, L, S, C`` per worker node) into a :class:`MatrixSnapshot`,
persisting it to *both* local disk and (optionally) a Redis store for
cross-replica durability.  On router boot the most recent snapshot is
loaded back into the memory field so freshly-started or recovered replicas
immediately inherit an informed routing bias instead of exploring from a
neutral baseline.

Persistence layout
------------------
* **Disk** — ``<storage_path>/snapshot_latest.json`` (rolling) and
  ``<storage_path>/snapshot_<epoch>.json`` (historical archive).
* **Redis** — ``SET stigmergic:matrix:checkpoint:latest <json>`` (rolling)
  and an ``XADD`` into the ``stigmergic:matrix:checkpoints`` stream
  (append-only history).
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

from api.metrics import get_total_routed_requests, stigmergic_checkpoints_created_total

logger = logging.getLogger(__name__)

try:
    import redis.asyncio as redis
except ImportError:  # pragma: no cover
    redis = None


class MatrixSnapshot(BaseModel):
    """Structured serialisation of the pheromone state matrix.

    Attributes
    ----------
    timestamp
        Unix epoch at which the snapshot was taken.
    version
        Schema version — incremented when the on-disk format changes.
    node_metrics
        Mapping ``node_id -> {"V", "L", "S", "C"}`` where each value is a
        floatised pheromone trace for that node.
    entropy_rate
        Boltzmann entropy of the current node-selection probability
        distribution (a proxy for routing "disorder"); ``0`` when the router
        agent is unavailable to compute it.
    total_routed_requests
        Cumulative request count at snapshot time (best-effort).
    """

    timestamp: float
    version: str = "1.0"
    node_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    entropy_rate: float = 0.0
    total_routed_requests: int = 0


def _decode_redis_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


class CheckpointManager:
    """Persist and restore :class:`BasePheromoneMemoryField` state.

    Parameters
    ----------
    memory_field
        The memory field to snapshot / hydrate.  May be ``None`` when the
        manager is used purely for checkpoint inspection/CLI operations.
    redis_client
        Optional ``redis.asyncio.Redis`` client.  When provided the
        latest snapshot is stored under
        ``stigmergic:matrix:checkpoint:latest`` and appended to the
        ``stigmergic:matrix:checkpoints`` stream.
    storage_path
        Local directory for JSON snapshot files.
    router_agent
        Optional router agent used to compute the entropy rate during
        snapshotting.  When omitted ``entropy_rate`` defaults to ``0.0``.
    """

    CHECKPOINT_REDIS_KEY = "stigmergic:matrix:checkpoint:latest"
    CHECKPOINT_STREAM_KEY = "stigmergic:matrix:checkpoints"
    CURRENT_VERSION = "1.0"

    def __init__(
        self,
        memory_field: Optional[Any] = None,
        redis_client: Optional[Any] = None,
        storage_path: str = "./data/checkpoints",
        router_agent: Optional[Any] = None,
    ) -> None:
        self._memory_field = memory_field
        self._redis: Optional[Any] = redis_client
        self._storage_path: str = storage_path
        self._router_agent = router_agent
        # Directory is created lazily on the first write so that merely
        # constructing a manager (e.g. for a read-only load) has no side
        # effects.

    # ── Snapshot creation ─────────────────────────────────────────────

    async def create_snapshot(self) -> MatrixSnapshot:
        """Capture the current pheromone matrix into a :class:`MatrixSnapshot`."""
        if self._memory_field is None:
            raise RuntimeError("CheckpointManager requires a memory_field to create snapshots")

        state = await self._memory_field.get_state_vector()
        node_ids: List[str] = list(self._memory_field.node_ids)
        node_metrics: Dict[str, Dict[str, float]] = {}
        for i, nid in enumerate(node_ids):
            if i < state.shape[0]:
                node_metrics[nid] = {
                    "V": float(state[i, 0]),
                    "L": float(state[i, 1]),
                    "S": float(state[i, 2]),
                    "C": float(state[i, 3]) if state.shape[1] >= 4 else 0.0,
                }

        entropy_rate = await self._compute_entropy_rate(state)

        return MatrixSnapshot(
            timestamp=time.time(),
            version=self.CURRENT_VERSION,
            node_metrics=node_metrics,
            entropy_rate=entropy_rate,
            total_routed_requests=get_total_routed_requests(),
        )

    async def _compute_entropy_rate(self, state: np.ndarray) -> float:
        """Boltzmann entropy of the current node-selection distribution."""
        if self._router_agent is None or state.shape[0] < 1:
            return 0.0
        try:
            scores = self._router_agent.compute_scores(state)
            probs = self._router_agent.compute_probabilities(scores)
            return float(-float(np.sum(probs * np.log(probs + 1e-12))))
        except Exception as exc:
            logger.debug("Entropy computation skipped: %s", exc)
            return 0.0

    # ── Persistence ───────────────────────────────────────────────────

    async def save_checkpoint(self, snapshot: MatrixSnapshot) -> str:
        """Persist *snapshot* to disk and Redis; return the primary location.

        Both backends are best-effort: a failure writing to one storage is
        logged but does not prevent the other from succeeding.  The returned
        string is the local file path that was written (or the Redis key
        when only Redis is configured).
        """
        data = snapshot.model_dump_json()
        disk_ok = await self._save_to_disk(snapshot, data)
        await self._save_to_redis(data)

        if disk_ok:
            location = os.path.join(self._storage_path, "snapshot_latest.json")
        else:
            location = self.CHECKPOINT_REDIS_KEY
        return location

    async def _save_to_disk(self, snapshot: MatrixSnapshot, data: str) -> bool:
        try:
            os.makedirs(self._storage_path, exist_ok=True)
            await asyncio.to_thread(self._write_file, "snapshot_latest.json", data)
            await asyncio.to_thread(
                self._write_file,
                f"snapshot_{int(snapshot.timestamp)}.json",
                data,
            )
            return True
        except OSError as exc:
            logger.warning("Failed to persist checkpoint to disk: %s", exc)
            return False

    def _write_file(self, filename: str, data: str) -> None:
        path = os.path.join(self._storage_path, filename)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(data)

    async def _save_to_redis(self, data: str) -> bool:
        if self._redis is None:
            return False
        try:
            await self._redis.set(self.CHECKPOINT_REDIS_KEY, data)
            await self._redis.xadd(self.CHECKPOINT_STREAM_KEY, {"snapshot": data})
            return True
        except Exception as exc:
            logger.warning("Failed to persist checkpoint to Redis: %s", exc)
            return False

    # ── Restoration ───────────────────────────────────────────────────

    async def load_latest_checkpoint(self) -> Optional[MatrixSnapshot]:
        """Load the most recent snapshot from Redis, then disk.

        Returns ``None`` when no checkpoint exists in either backend.
        """
        # 1. Redis (authoritative when reachable).
        if self._redis is not None:
            try:
                raw = await self._redis.get(self.CHECKPOINT_REDIS_KEY)
                if raw:
                    return MatrixSnapshot.model_validate_json(_decode_redis_str(raw))
            except Exception as exc:
                logger.warning("Redis checkpoint load failed (%s); falling back to disk", exc)

        # 2. Local disk fallback.
        path = os.path.join(self._storage_path, "snapshot_latest.json")
        if os.path.exists(path):
            try:
                return await asyncio.to_thread(self._read_snapshot_file, path)
            except Exception as exc:
                logger.warning("Disk checkpoint load failed: %s", exc)

        return None

    def _read_snapshot_file(self, path: str) -> MatrixSnapshot:
        with open(path, "r", encoding="utf-8") as fh:
            return MatrixSnapshot.model_validate_json(fh.read())

    # ── Periodic checkpointing ────────────────────────────────────────

    async def start_periodic_checkpointing(
        self, interval_seconds: int = 60
    ) -> None:
        """Background loop: snapshot + persist every *interval_seconds*.

        Exits cleanly on ``asyncio.CancelledError`` (the canonical shutdown
        signal used by the FastAPI lifespan).
        """
        logger.info("Periodic checkpointing started (interval=%ss)", interval_seconds)
        while True:
            try:
                snapshot = await self.create_snapshot()
                data = snapshot.model_dump_json()
                disk_ok = await self._save_to_disk(snapshot, data)
                redis_ok = await self._save_to_redis(data)
                if disk_ok:
                    stigmergic_checkpoints_created_total.labels(storage_type="disk").inc()
                if redis_ok:
                    stigmergic_checkpoints_created_total.labels(storage_type="redis").inc()
            except asyncio.CancelledError:
                logger.info("Periodic checkpointing cancelled")
                raise
            except Exception as exc:
                logger.warning("Periodic checkpoint failed: %s", exc)

            try:
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                logger.info("Periodic checkpointing cancelled during sleep")
                raise
