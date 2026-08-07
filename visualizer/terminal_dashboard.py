"""Live terminal dashboard for monitoring the stigmergic mesh router.

Uses ``rich.live.Live`` to render a continuously-updating terminal UI that
shows per-node pheromone state, computed attraction scores, selection
probabilities, and a visual traffic-intensity bar.
"""

import asyncio
from typing import Dict, List, Optional

import numpy as np
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from core.memory_field import PheromoneMemoryField
from core.router_agent import StigmergicRouterAgent
from core.worker_node import BaseWorkerNode


def _bar(value: float, width: int = 20) -> str:
    """Render a simple text bar chart for *value* in [0, 1]."""
    value = max(0.0, min(1.0, value))
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)


def _status_color(value: float) -> str:
    """Return a Rich color name based on a normalized value."""
    if value > 0.7:
        return "green"
    elif value > 0.4:
        return "yellow"
    else:
        return "red"


def build_dashboard_table(
    memory_field: PheromoneMemoryField,
    workers: Dict[str, BaseWorkerNode],
    router: StigmergicRouterAgent,
    state: np.ndarray,
) -> Table:
    """Build a Rich :class:`Table` reflecting the current mesh state.

    Parameters
    ----------
    memory_field
        The shared pheromone memory field.
    workers
        Mapping of node IDs to worker node instances.
    router
        The stigmergic router agent (used for score/probability computation).
    state
        Current ``(N_nodes, 3)`` state matrix copy.

    Returns
    -------
    Table
        A Rich table renderable for display.
    """
    scores = router.compute_scores(state)
    probs = router.compute_probabilities(scores)
    node_ids = memory_field.node_ids

    table = Table(
        title="Stigmergic Mesh — Live Node Status",
        show_lines=True,
        header_style="bold cyan",
        border_style="bright_black",
    )
    table.add_column("Node", style="cyan", no_wrap=True)
    table.add_column("V (Success)", justify="right", style="green")
    table.add_column("L (Latency)", justify="right", style="yellow")
    table.add_column("S (Saturation)", justify="right", style="red")
    table.add_column("Active Load", justify="right", style="magenta")
    table.add_column("Score", justify="right", style="bold")
    table.add_column("Prob", justify="right")
    table.add_column("Traffic", no_wrap=True)

    for i, nid in enumerate(node_ids):
        v = state[i, 0]
        l = state[i, 1]
        s = state[i, 2]
        score = scores[i]
        prob = probs[i]

        worker = workers.get(nid)
        active_load = getattr(worker, "_active_load", 0) if worker else 0

        table.add_row(
            nid,
            f"{v:.4f}",
            f"{l:.4f}",
            f"{s:.4f}",
            str(active_load),
            f"{score:.2f}",
            f"{prob * 100:.1f}%",
            f"[{_status_color(prob)}]{_bar(prob)}[/]",
        )

    return table


def build_summary_panel(
    memory_field: PheromoneMemoryField,
    workers: Dict[str, BaseWorkerNode],
) -> Panel:
    """Build a summary panel with aggregate statistics."""
    total_requests = sum(
        getattr(w, "requests_served", 0) for w in workers.values()
    )
    lines: List[str] = []
    for nid, worker in workers.items():
        served = getattr(worker, "requests_served", 0)
        pct = (served / total_requests * 100) if total_requests > 0 else 0.0
        base_delay = getattr(worker, "base_delay_sec", 0.0)
        lines.append(
            f"  {nid:15s} | {served:4d} reqs | {pct:5.1f}% | base={base_delay:.3f}s"
        )
    summary_text = (
        f"Total requests served: {total_requests}\n\n"
        + "\n".join(lines)
    )
    return Panel(summary_text, title="Mesh Summary", border_style="blue")


async def render_dashboard(
    memory_field: PheromoneMemoryField,
    workers: Dict[str, BaseWorkerNode],
    router: StigmergicRouterAgent,
    refresh_interval: float = 0.5,
    stop_event: Optional[asyncio.Event] = None,
) -> None:
    """Run a live terminal dashboard alongside simulation tasks.

    Continuously polls the shared :class:`PheromoneMemoryField` and
    renders a Rich ``Live`` display with node status tables and traffic
    intensity bars.

    Parameters
    ----------
    memory_field
        The shared memory field to monitor.
    workers
        Dictionary of worker nodes to display.
    router
        The router agent (for computing scores and probabilities).
    refresh_interval
        Seconds between dashboard refreshes.
    stop_event
        Optional :class:`asyncio.Event` that, when set, causes the
        dashboard loop to exit gracefully.
    """
    console = Console()

    with Live(
        renderable=Panel("Loading...", title="Initializing Dashboard"),
        refresh_per_second=2,
        console=console,
    ) as live:
        try:
            while stop_event is None or not stop_event.is_set():
                state = await memory_field.get_state_vector()
                table = build_dashboard_table(
                    memory_field, workers, router, state
                )
                panel = build_summary_panel(memory_field, workers)

                grid = Group(table, panel)

                live.update(grid)
                await asyncio.sleep(refresh_interval)
        except asyncio.CancelledError:
            pass
        finally:
            live.stop()
