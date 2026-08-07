"""Asynchronous background decay engine for evaporating pheromone traces."""

import asyncio
import logging

from .memory_field import BasePheromoneMemoryField

logger = logging.getLogger(__name__)


async def start_decay_engine(
    memory_field: BasePheromoneMemoryField,
    decay_rate: float,
    interval_sec: float,
) -> None:
    """Run an infinite evaporation loop on the memory field.

    Every *interval_sec* wall-clock seconds the field's traces are
    scaled by ``(1 - decay_rate)``, causing stale pheromone information
    to fade.  This prevents the field from being dominated by old
    observations and lets the router adapt to changing node conditions.

    The coroutine runs forever and must be cancelled (e.g. via
    ``task.cancel()``) to stop.

    Parameters
    ----------
    memory_field
        Shared :class:`PheromoneMemoryField` instance to evaporate.
    decay_rate
        Fraction of trace value to evaporate per cycle (0 < decay_rate < 1).
    interval_sec
        Wall-clock seconds between evaporation cycles.
    """
    while True:
        await asyncio.sleep(interval_sec)
        await memory_field.apply_evaporation(decay_rate)
        logger.debug("Decay engine applied evaporation with rate %.4f", decay_rate)
