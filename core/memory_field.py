"""Pheromone memory field for stigmergic routing.

This module provides a polymorphic memory field with two backends:

* :class:`InMemoryPheromoneMemoryField` — single-process, lock-guarded
  NumPy array (default for local execution and testing).
* :class:`RedisPheromoneMemoryField` — distributed backend backed by Redis,
  using atomic Lua scripts to guarantee consistency when multiple router
  replicas deposit traces concurrently.

Both backends implement the same :class:`BasePheromoneMemoryField` interface
and are selected at startup via ``config.yaml``'s ``storage_backend`` key.

State matrix layout — ``(N_nodes, 4)`` where each row encodes:

Column 0 — V (Success): exponentially-weighted success signal in [0, 1].
    Higher values indicate a node with a strong record of successful
    inferences.

Column 1 — L (Latency): EWMA of observed latency in seconds.
    Lower values are better; high latency penalises the score.

Column 2 — S (Saturation): normalized residual load on the node.
    Higher saturation discourages further traffic.

Column 3 — C (Capability Fit): EWMA of how well the node's declared
    capability tags matched the requests routed to it (∈ [0, 1]).

Extended attraction score::

    Score_i = (alpha * V_i + delta * C_i + eps)
              / (beta * L_i + gamma * S_i + eps)
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import redis.asyncio as redis
except ImportError:
    redis = None


class BasePheromoneMemoryField(ABC):
    """Abstract interface for pheromone memory fields.

    Defines the minimal contract that the :class:`StigmergicRouterAgent`
    and :func:`start_decay_engine` rely on:

    * :meth:`get_state_vector` — read the full state matrix
    * :meth:`deposit_trace` — write EWMA traces for a single node
    * :meth:`apply_evaporation` — bulk multiplicative decay
    * :meth:`node_ids` — the ordered list of tracked nodes
    """

    @property
    @abstractmethod
    def n_nodes(self) -> int:
        """Number of nodes tracked by this field."""
        ...

    @abstractmethod
    def _index_of(self, node_id: str) -> int:
        """Return the row index for *node_id*."""
        ...

    @abstractmethod
    async def get_state_vector(self) -> np.ndarray:
        """Return a copy of the full state matrix ``(N_nodes, 4)``."""
        ...

    @abstractmethod
    async def deposit_trace(
        self,
        node_id: str,
        latency_sec: float,
        tokens: int,
        success: bool,
        active_load: int,
        capability_match: float = 0.5,
    ) -> None:
        """Update pheromone traces for *node_id* after a completed inference."""
        ...

    @abstractmethod
    async def apply_evaporation(self, decay_rate: float) -> None:
        """Apply multiplicative evaporation to every trace."""
        ...

    @abstractmethod
    async def hydrate_from_snapshot(self, snapshot: Any) -> int:
        """Restore V/L/S/C traces for known nodes from a checkpoint snapshot."""
        ...

    @abstractmethod
    async def export_state_dict(self) -> Dict[str, Dict[str, float]]:
        """Return a copy of every node's V/L/S/C traces."""
        ...


class InMemoryPheromoneMemoryField(BasePheromoneMemoryField):
    """Lock-guarded in-memory pheromone field backed by a NumPy array.

    Suitable for single-process execution, testing, and local development.
    All operations are guarded by an :class:`asyncio.Lock` to prevent
    race conditions under concurrent coroutine access.
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
        """Initialise the field.

        Parameters
        ----------
        node_ids
            Ordered list of worker node identifiers.
        initial_state
            Optional ``(N_nodes, 4)`` array to seed the field.  When
            *None* the field starts from baseline ``[V=1.0, L=0.1,
            S=0.0, C=1.0]``.
        saturation_scale
            Scaling factor for the *active_load* → S column.
        decay_success
            When *True* (default), the V column decays during evaporation.
        decay_capability
            When *True* (default), the C column decays during evaporation.
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
        return len(self.node_ids)

    def _index_of(self, node_id: str) -> int:
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
        """Update pheromone traces for *node_id* via EWMA.

        V, L, and C use EWMA with alpha=0.5.  S is set directly from
        active_load scaled by *saturation_scale*.
        """
        async with self._lock:
            idx = self._index_of(node_id)
            success_val = 1.0 if success else 0.0
            alpha = 0.5

            self._state[idx, 0] = (
                alpha * success_val + (1.0 - alpha) * self._state[idx, 0]
            )
            self._state[idx, 1] = (
                alpha * latency_sec + (1.0 - alpha) * self._state[idx, 1]
            )
            self._state[idx, 2] = float(active_load) * self.saturation_scale
            cap = max(0.0, min(1.0, float(capability_match)))
            self._state[idx, 3] = (
                alpha * cap + (1.0 - alpha) * self._state[idx, 3]
            )

    async def apply_evaporation(self, decay_rate: float) -> None:
        """Apply multiplicative evaporation to all traces.

        V decays only if ``decay_success`` is *True*.
        C decays only if ``decay_capability`` is *True*.
        L and S always decay.
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

    async def add_node(
        self, node_id: str, capability_tags: Optional[List[str]] = None
    ) -> None:
        """Dynamically append a new node to the state matrix.

        The new row is initialised to baseline ``[V=1.0, L=0.1,
        S=0.0, C=1.0]`` (or the provided ``initial_state`` baseline).
        """
        async with self._lock:
            if node_id in self._node_index:
                return
            idx = len(self.node_ids)
            self._node_index[node_id] = idx
            self.node_ids.append(node_id)
            row = self._DEFAULT_STATE.copy()
            self._state = np.vstack([self._state, row[np.newaxis, :]])

    async def remove_node(self, node_id: str) -> None:
        """Remove a node from the in-memory state matrix."""
        async with self._lock:
            if node_id not in self._node_index:
                return
            idx = self._node_index.pop(node_id)
            self.node_ids.pop(idx)
            self._node_index = {
                nid: i for i, nid in enumerate(self.node_ids)
            }
            self._state = np.delete(self._state, idx, axis=0)

    async def hydrate_from_snapshot(self, snapshot: Any) -> int:
        """Overwrite V/L/S/C traces for known nodes from a checkpoint snapshot.

        Returns the number of nodes whose rows were updated.  Nodes present
        in *snapshot* but not in this field are skipped; field nodes absent
        from *snapshot* retain their current baseline values.

        ``snapshot`` may be a :class:`~core.checkpointing.MatrixSnapshot` or
        a plain dict with a ``node_metrics`` mapping.
        """
        node_metrics = getattr(snapshot, "node_metrics", None) or snapshot.get("node_metrics", {})
        updated = 0
        async with self._lock:
            for nid, traces in node_metrics.items():
                idx = self._node_index.get(nid)
                if idx is None:
                    continue
                if "V" in traces:
                    self._state[idx, 0] = float(traces["V"])
                if "L" in traces:
                    self._state[idx, 1] = float(traces["L"])
                if "S" in traces:
                    self._state[idx, 2] = float(traces["S"])
                if "C" in traces:
                    self._state[idx, 3] = float(traces["C"])
                updated += 1
        return updated

    async def export_state_dict(self) -> Dict[str, Dict[str, float]]:
        """Return a thread-safe copy of every node's V/L/S/C traces."""
        async with self._lock:
            return {
                nid: {
                    "V": float(self._state[i, 0]),
                    "L": float(self._state[i, 1]),
                    "S": float(self._state[i, 2]),
                    "C": float(self._state[i, 3]),
                }
                for i, nid in enumerate(self.node_ids)
            }


class RedisPheromoneMemoryField(BasePheromoneMemoryField):
    """Distributed pheromone memory field backed by Redis.

    Each node's four traces are stored as a Redis hash at key
    ``stigmergic:node:{node_id}`` with fields ``V``, ``L``, ``S``,
    ``C``, and ``ts`` (last-update timestamp for lazy decay).

    Trace deposition is performed via an atomic Lua script that:
    1. Reads the current state and timestamp.
    2. Computes elapsed-time decay factor ``(1 - decay_rate) ^ steps``.
    3. Applies EWMA updates for V, L, and C.
    4. Sets S from the incoming active_load.
    5. Writes back atomically.

    This guarantees correctness even when multiple router replicas
    deposit traces for the same node concurrently.
    """

    _DEFAULT_STATE = [1.0, 0.1, 0.0, 1.0]

    # Lua script for atomic trace deposit with lazy decay.
    # Keys:   KEYS[1] = node hash key
    # ARGV:   ARGV[1] = decay_rate (float)
    #         ARGV[2] = decay_success flag (0 or 1)
    #         ARGV[3] = decay_interval_sec (float)
    #         ARGV[4] = decay_capability flag (0 or 1)
    #         ARGV[5] = success_val (0.0 or 1.0)
    #         ARGV[6] = now (timestamp)
    #         ARGV[7] = latency_sec (float)
    #         ARGV[8] = saturated_load (float)
    #         ARGV[9] = capability_match (0..1)
    _DEPOSIT_SCRIPT = """
    local key = KEYS[1]
    local fields = redis.call('HMGET', key, 'V', 'L', 'S', 'C', 'ts')
    local V = tonumber(fields[1]) or 1.0
    local L = tonumber(fields[2]) or 0.1
    local S_val = tonumber(fields[3]) or 0.0
    local C = tonumber(fields[4]) or 1.0
    local ts = tonumber(fields[5]) or 0.0

    local now = tonumber(ARGV[6])
    local elapsed = now - ts
    local steps = elapsed / tonumber(ARGV[3])
    if steps < 0 then steps = 0 end
    local decay_rate = tonumber(ARGV[1])
    local factor = math.pow(1.0 - decay_rate, steps)

    local decay_v = tonumber(ARGV[2]) == 1
    local decay_c = tonumber(ARGV[4]) == 1

    if decay_v then V = V * factor end
    L = L * factor
    S_val = S_val * factor
    if decay_c then C = C * factor end

    local alpha = 0.5
    local success_val = tonumber(ARGV[5])
    local latency = tonumber(ARGV[7])
    local saturated_load = tonumber(ARGV[8])
    local cap_match = tonumber(ARGV[9])

    -- Clamp capability match to [0, 1]
    if cap_match < 0 then cap_match = 0 end
    if cap_match > 1 then cap_match = 1 end

    V = alpha * success_val + (1.0 - alpha) * V
    L = alpha * latency + (1.0 - alpha) * L
    S_val = saturated_load
    C = alpha * cap_match + (1.0 - alpha) * C

    redis.call('HMSET', key,
        'V', V, 'L', L, 'S', S_val, 'C', C, 'ts', now)
    return {V, L, S_val, C, now}
    """

    def __init__(
        self,
        node_ids: List[str],
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        saturation_scale: float = 0.1,
        decay_success: bool = True,
        decay_capability: bool = True,
        decay_rate: float = 0.05,
        decay_interval_sec: float = 0.5,
        redis_client: Optional[Any] = None,
    ) -> None:
        if redis is None and redis_client is None:
            raise ImportError(
                "redis-py is not installed. Run: pip install redis"
            )

        self.node_ids: List[str] = list(node_ids)
        self._node_index: Dict[str, int] = {
            nid: i for i, nid in enumerate(self.node_ids)
        }
        self.saturation_scale: float = saturation_scale
        self.decay_success: bool = decay_success
        self.decay_capability: bool = decay_capability
        self._decay_rate: float = decay_rate
        self._decay_interval_sec: float = decay_interval_sec

        if redis_client is not None:
            self._redis = redis_client
        else:
            self._redis = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                decode_responses=False,
            )
        self._prefix = "stigmergic:node:"
        self._deposit_sha: Optional[str] = None
        self._init_lock = asyncio.Lock()
        self._initialized = False

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    def _index_of(self, node_id: str) -> int:
        if node_id not in self._node_index:
            raise KeyError(f"Unknown node_id: {node_id!r}")
        return self._node_index[node_id]

    def _key(self, node_id: str) -> str:
        return f"{self._prefix}{node_id}"

    async def _ensure_initialized(self) -> None:
        """Initialize Redis keys and load the Lua script (idempotent)."""
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return

            pipe = self._redis.pipeline()
            for nid in self.node_ids:
                key = self._key(nid)
                v, l, s, c = self._DEFAULT_STATE
                pipe.hset(key, mapping={
                    "V": v, "L": l, "S": s, "C": c,
                    "ts": time.time(),
                })
            await pipe.execute()

            self._deposit_sha = await self._redis.script_load(
                self._DEPOSIT_SCRIPT
            )
            self._initialized = True

    async def get_state_vector(self) -> np.ndarray:
        """Read all node states from Redis without additional decay.

        Returns a ``(N_nodes, 4)`` NumPy array with columns [V, L, S, C].
        Values are read at face value — all decay is handled by the
        background decay engine (:func:`start_decay_engine`) via
        :meth:`apply_evaporation`, consistent with the in-memory backend.
        """
        await self._ensure_initialized()

        pipe = self._redis.pipeline()
        for nid in self.node_ids:
            pipe.hgetall(self._key(nid))
        raw_results = await pipe.execute()

        state = np.zeros((self.n_nodes, 4), dtype=np.float64)

        for i, (nid, raw) in enumerate(zip(self.node_ids, raw_results)):
            v = float(raw.get(b"V", 1.0) or 1.0)
            l = float(raw.get(b"L", 0.1) or 0.1)
            s = float(raw.get(b"S", 0.0) or 0.0)
            c = float(raw.get(b"C", 1.0) or 1.0)
            state[i] = [v, l, s, c]

        return state

    async def deposit_trace(
        self,
        node_id: str,
        latency_sec: float,
        tokens: int,
        success: bool,
        active_load: int,
        capability_match: float = 0.5,
    ) -> None:
        """Atomically deposit traces for *node_id* via Lua script.

        The script applies lazy decay based on elapsed time since the
        last update, then applies EWMA updates for V, L, and C.
        """
        await self._ensure_initialized()

        self._index_of(node_id)
        success_val = 1.0 if success else 0.0
        saturated_load = float(active_load) * self.saturation_scale
        cap = max(0.0, min(1.0, float(capability_match)))
        now = time.time()

        await self._redis.evalsha(
            self._deposit_sha,
            1,  # 1 key
            self._key(node_id),
            self._decay_rate,
            1 if self.decay_success else 0,
            self._decay_interval_sec,
            1 if self.decay_capability else 0,
            success_val,
            now,
            latency_sec,
            saturated_load,
            cap,
        )

    async def apply_evaporation(self, decay_rate: float) -> None:
        """Apply bulk multiplicative evaporation to all nodes via pipeline.

        V decays only if ``decay_success`` is *True*.
        C decays only if ``decay_capability`` is *True*.

        Updates the ``ts`` field for each node so that subsequent
        :meth:`get_state_vector` lazy-decay calculations start from
        the correct baseline, preventing double-decay.
        """
        if not 0.0 <= decay_rate < 1.0:
            raise ValueError("decay_rate must be in [0, 1)")
        await self._ensure_initialized()

        factor = 1.0 - decay_rate
        now = time.time()

        pipe = self._redis.pipeline()
        for nid in self.node_ids:
            pipe.hgetall(self._key(nid))
        raw_results = await pipe.execute()

        pipe2 = self._redis.pipeline()
        for nid, raw in zip(self.node_ids, raw_results):
            updates: Dict[str, Any] = {"ts": now}
            if self.decay_success:
                updates["V"] = float(raw.get(b"V", 1.0) or 1.0) * factor
            updates["L"] = float(raw.get(b"L", 0.1) or 0.1) * factor
            updates["S"] = float(raw.get(b"S", 0.0) or 0.0) * factor
            if self.decay_capability:
                updates["C"] = float(raw.get(b"C", 1.0) or 1.0) * factor
            pipe2.hset(self._key(nid), mapping=updates)
        await pipe2.execute()

    async def add_node(
        self, node_id: str, capability_tags: Optional[List[str]] = None
    ) -> None:
        """Dynamically add a new node's Redis hash to the field."""
        await self._ensure_initialized()
        if node_id in self._node_index:
            return
        idx = len(self.node_ids)
        self._node_index[node_id] = idx
        self.node_ids.append(node_id)
        v, l, s, c = self._DEFAULT_STATE
        await self._redis.hset(
            self._key(node_id),
            mapping={"V": v, "L": l, "S": s, "C": c, "ts": time.time()},
        )

    async def remove_node(self, node_id: str) -> None:
        """Remove a node's Redis hash and internal tracking."""
        if node_id not in self._node_index:
            return
        await self._ensure_initialized()
        await self._redis.delete(self._key(node_id))
        idx = self._node_index.pop(node_id)
        self.node_ids.pop(idx)
        self._node_index = {
            nid: i for i, nid in enumerate(self.node_ids)
        }

    async def hydrate_from_snapshot(self, snapshot: Any) -> int:
        """Overwrite V/L/S/C traces for known nodes from a checkpoint.

        Returns the number of nodes whose Redis hashes were written.
        """
        await self._ensure_initialized()
        node_metrics = getattr(snapshot, "node_metrics", None) or snapshot.get("node_metrics", {})
        if not node_metrics:
            return 0
        now = time.time()
        pipe = self._redis.pipeline()
        updated = 0
        for nid, traces in node_metrics.items():
            if nid not in self._node_index:
                continue
            mapping = {}
            if "V" in traces:
                mapping["V"] = float(traces["V"])
            if "L" in traces:
                mapping["L"] = float(traces["L"])
            if "S" in traces:
                mapping["S"] = float(traces["S"])
            if "C" in traces:
                mapping["C"] = float(traces["C"])
            if mapping:
                mapping["ts"] = now
                pipe.hset(self._key(nid), mapping=mapping)
                updated += 1
        if updated:
            await pipe.execute()
        return updated

    async def export_state_dict(self) -> Dict[str, Dict[str, float]]:
        """Return a copy of every node's V/L/S/C traces from Redis."""
        await self._ensure_initialized()
        pipe = self._redis.pipeline()
        for nid in self.node_ids:
            pipe.hgetall(self._key(nid))
        raw_results = await pipe.execute()
        state: Dict[str, Dict[str, float]] = {}
        for nid, raw in zip(self.node_ids, raw_results):
            state[nid] = {
                "V": float(raw.get(b"V", 1.0) or 1.0),
                "L": float(raw.get(b"L", 0.1) or 0.1),
                "S": float(raw.get(b"S", 0.0) or 0.0),
                "C": float(raw.get(b"C", 1.0) or 1.0),
            }
        return state

    async def close(self) -> None:
        """Close the Redis connection pool."""
        if self._redis:
            await self._redis.aclose()


# ── Backward-compatible alias ──────────────────────────────────────────

PheromoneMemoryField = InMemoryPheromoneMemoryField


# ── Factory ────────────────────────────────────────────────────────────

def get_memory_field(
    config: dict,
    node_ids: List[str],
    initial_state: Optional[np.ndarray] = None,
) -> BasePheromoneMemoryField:
    """Instantiate the appropriate memory field from *config*.

    Reads ``storage_backend`` from the config dict and returns either an
    :class:`InMemoryPheromoneMemoryField` or
    :class:`RedisPheromoneMemoryField`.

    Parameters
    ----------
    config
        Full configuration dict (typically loaded from ``config.yaml``).
    node_ids
        Ordered list of worker node identifiers.
    initial_state
        Optional ``(N_nodes, 4)`` array to seed the field.
    """
    backend = config.get("storage_backend", "in_memory")

    common_kwargs = dict(
        saturation_scale=config.get("saturation_scale", 0.1),
        decay_success=config.get("decay_success", True),
        decay_capability=config.get("decay_capability", True),
    )

    if backend == "redis":
        redis_cfg = config.get("redis", {})
        return RedisPheromoneMemoryField(
            node_ids=node_ids,
            redis_host=redis_cfg.get("host", "localhost"),
            redis_port=redis_cfg.get("port", 6379),
            redis_db=redis_cfg.get("db", 0),
            redis_password=redis_cfg.get("password"),
            decay_rate=config.get("decay_rate", 0.05),
            decay_interval_sec=config.get("decay_interval_sec", 0.5),
            **common_kwargs,
        )
    else:
        return InMemoryPheromoneMemoryField(
            node_ids=node_ids,
            initial_state=initial_state,
            **common_kwargs,
        )
