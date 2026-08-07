"""Dynamic worker auto-registration via Redis Pub/Sub and heartbeat monitoring.

Workers broadcast their presence on the ``stigmergic:worker:discovery``
Redis channel on startup.  The :class:`WorkerRegistry` subscribes to
that channel, tracks active heartbeats, and automatically prunes
workers whose heartbeat keys expire (TTL-based failure detection).

When a new worker is discovered, the registry calls
``memory_field.add_node()`` to dynamically expand the state matrix.
When a worker is pruned, its latency trace ($L$) is set to infinity so
the softmax naturally excludes it from sampling.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

import redis.asyncio as redis

from core.memory_field import BasePheromoneMemoryField

logger = logging.getLogger(__name__)

HEARTBEAT_CHANNEL = "stigmergic:worker:discovery"
HEARTBEAT_KEY_PREFIX = "stigmergic:heartbeat:"
HEARTBEAT_TTL_SEC = 5
HEARTBEAT_INTERVAL_SEC = 2
PRUNE_INTERVAL_SEC = 1


class WorkerRegistry:
    """Manages dynamic worker discovery, heartbeat, and pruning via Redis.

    Parameters
    ----------
    redis_client
        A connected ``redis.asyncio.Redis`` instance (shared with the
        memory field or a separate connection).
    memory_field
        The pheromone memory field whose node list is dynamically
        expanded as workers register.
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        memory_field: BasePheromoneMemoryField,
        heartbeat_ttl: int = HEARTBEAT_TTL_SEC,
        heartbeat_interval: float = HEARTBEAT_INTERVAL_SEC,
        prune_interval: float = PRUNE_INTERVAL_SEC,
    ) -> None:
        self._redis: redis.Redis = redis_client
        self._memory_field: BasePheromoneMemoryField = memory_field
        self._heartbeat_ttl: int = heartbeat_ttl
        self._heartbeat_interval: float = heartbeat_interval
        self._prune_interval: float = prune_interval
        self._pubsub: Optional[redis.PubSub] = None
        self._running: bool = False
        self._known_nodes: Dict[str, Dict[str, Any]] = {}
        self._discovery_task: Optional[asyncio.Task] = None
        self._prune_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

    def _heartbeat_key(self, node_id: str) -> str:
        return f"{HEARTBEAT_KEY_PREFIX}{node_id}"

    async def register_worker(self, node_config: Dict[str, Any]) -> None:
        """Register a worker with the discovery channel and Redis.

        Publishes the node config on the discovery channel and sets
        a TTL-key heartbeat.  Also dynamically expands the memory
        field matrix via ``add_node()``.
        """
        node_id = node_config["node_id"]
        capability_tags = node_config.get("capability_tags", [])

        # Publish discovery announcement
        await self._redis.publish(
            HEARTBEAT_CHANNEL,
            json.dumps({
                "node_id": node_id,
                "capability_tags": capability_tags,
                "config": node_config,
                "timestamp": time.time(),
                "action": "register",
            }),
        )

        # Set initial heartbeat key
        await self._redis.setex(
            self._heartbeat_key(node_id),
            self._heartbeat_ttl,
            json.dumps({"timestamp": time.time()}),
        )

        self._known_nodes[node_id] = {
            "capability_tags": capability_tags,
            "config": node_config,
            "last_heartbeat": time.time(),
        }

        # Dynamically add to memory field if not already present
        if node_id not in self._memory_field.node_ids:
            await self._memory_field.add_node(node_id, capability_tags)
            logger.info("Worker registered: %s (tags: %s)", node_id, capability_tags)

    async def send_heartbeat(self, node_id: str) -> None:
        """Refresh the heartbeat TTL-key for *node_id*."""
        if node_id not in self._known_nodes:
            return
        await self._redis.setex(
            self._heartbeat_key(node_id),
            self._heartbeat_ttl,
            json.dumps({"timestamp": time.time()}),
        )
        self._known_nodes[node_id]["last_heartbeat"] = time.time()

    async def _listen_for_discovery(self) -> None:
        """Subscribe to the discovery channel and process registrations."""
        self._pubsub = self._redis.pubsub()
        await self._pubsub.subscribe(HEARTBEAT_CHANNEL)
        logger.info("Subscribed to %s", HEARTBEAT_CHANNEL)

        async for message in self._pubsub.listen():
            if not self._running:
                break
            if message["type"] != "pmessage":
                continue
            try:
                data = json.loads(message["data"])
                action = data.get("action", "register")
                if action == "register":
                    node_config = data.get("config", {})
                    node_id = data.get("node_id")
                    if node_id and node_id not in self._known_nodes:
                        self._known_nodes[node_id] = {
                            "capability_tags": data.get("capability_tags", []),
                            "config": node_config,
                            "last_heartbeat": time.time(),
                        }
                        if node_id not in self._memory_field.node_ids:
                            await self._memory_field.add_node(
                                node_id,
                                data.get("capability_tags", []),
                            )
                            logger.info(
                                "Discovered worker via Pub/Sub: %s", node_id
                            )
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Failed to parse discovery message: %s", exc)

    async def prune_unhealthy_workers(self) -> None:
        """Scan heartbeat keys; prune workers whose TTL has expired.

        For each known node, if the heartbeat key does not exist in Redis,
        the node is considered dead.  Its latency trace is set to 999.0
        (effectively removing it from the sampling space).
        """
        for node_id in list(self._known_nodes.keys()):
            key = self._heartbeat_key(node_id)
            exists = await self._redis.exists(key)
            if not exists:
                # Worker has not sent a heartbeat within the TTL window
                logger.warning("Pruning unhealthy worker: %s", node_id)
                await self._memory_field.deposit_trace(
                    node_id=node_id,
                    latency_sec=999.0,
                    tokens=0,
                    success=False,
                    active_load=0,
                    capability_match=0.0,
                )
                del self._known_nodes[node_id]

    async def _heartbeat_loop(self) -> None:
        """Background task: send our own heartbeat (if we are a router)."""
        # Router nodes don't register themselves; only workers do.
        # This loop can be used by workers when running in-process.
        pass

    async def start_background_tasks(self) -> None:
        """Start the discovery listener and pruning background tasks."""
        if self._running:
            return
        self._running = True

        self._discovery_task = asyncio.create_task(self._listen_for_discovery())
        self._prune_task = asyncio.create_task(self._prune_loop())

    async def _prune_loop(self) -> None:
        """Periodically call ``prune_unhealthy_workers``."""
        while self._running:
            await asyncio.sleep(self._prune_interval)
            try:
                await self.prune_unhealthy_workers()
            except Exception as exc:
                logger.warning("Prune loop error: %s", exc)

    async def stop_background_tasks(self) -> None:
        """Cancel all background tasks."""
        self._running = False
        for task in [self._discovery_task, self._prune_task, self._heartbeat_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if self._pubsub:
            await self._pubsub.unsubscribe(HEARTBEAT_CHANNEL)
            await self._pubsub.aclose()

    @property
    def known_nodes(self) -> Dict[str, Dict[str, Any]]:
        """Return a copy of the known nodes registry."""
        return dict(self._known_nodes)

    @property
    def node_ids(self) -> List[str]:
        """Return the list of currently known node IDs."""
        return list(self._known_nodes.keys())
