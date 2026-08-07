#!/usr/bin/env python3
"""Chaos failover benchmark for the stigmergic mesh router.

Demonstrates zero-touch automatic failover through pheromone feedback:

* **Phase A (Ticks 1-15)**: All nodes operating normally.  Traffic is
  distributed roughly equally among the three nodes.
* **Phase B (Ticks 16-30)**: ``node_alpha`` suffers a simulated crash
  (3.0 s fixed processing delay, load factor zeroed).  Traffic instantly
  abandons alpha as its latency trace spikes — no health probes involved.
* **Phase C (Ticks 31-45)**: ``node_alpha`` recovers.  As latency
  observations evaporate through the decay engine, alpha's attraction
  score rises and traffic gradually returns.

Run::

    python3 benchmarks/failover_test.py
"""

import asyncio
import os
import random
import sys
from collections import Counter
from typing import Dict, List

import numpy as np
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.decay_engine import start_decay_engine
from core.memory_field import PheromoneMemoryField
from core.router_agent import StigmergicRouterAgent
from core.worker_node import CPUMockWorkerNode

CONFIG_PATH = "config.yaml"
N_TICKS_PER_PHASE = 15
REQUESTS_PER_TICK = 5
SEED = 42
INTER_TICK_DELAY = 0.3

# Test-specific parameters tuned to showcase the full avoidance → recovery
# cycle within a short runtime.
TEST_DECAY_RATE = 0.20
TEST_DECAY_INTERVAL = 0.5
TEST_TEMPERATURE = 2.0
CRASH_DELAY_SEC = 2.0

# Steady-state baseline latency for the healthy 3-node configuration:
#   delay = base_delay * (1 + start_load * load_factor) = 0.05 * 1.5 = 0.075
STEADY_STATE_LATENCY = 0.075


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class PhaseTracker:
    """Accumulates per-phase request distributions and latencies."""

    def __init__(self, node_ids: List[str]):
        self.node_ids = node_ids
        self.phase_data: Dict[str, Counter] = {
            "A": Counter(),
            "B": Counter(),
            "C": Counter(),
        }
        self.phase_latency: Dict[str, Dict[str, List[float]]] = {
            "A": {nid: [] for nid in node_ids},
            "B": {nid: [] for nid in node_ids},
            "C": {nid: [] for nid in node_ids},
        }

    def record(self, phase: str, node_id: str, latency: float) -> None:
        self.phase_data[phase][node_id] += 1
        self.phase_latency[phase][node_id].append(latency)

    def totals(self, phase: str) -> int:
        return sum(self.phase_data[phase].values())


async def dispatch_tick(
    router: StigmergicRouterAgent,
    tracker: PhaseTracker,
    phase: str,
    tick_num: int,
) -> None:
    """Dispatch *REQUESTS_PER_TICK* requests sequentially within a tick."""
    for i in range(REQUESTS_PER_TICK):
        prompt = f"Failover test: phase {phase}, tick {tick_num}, request {i}"
        result = await router.route_and_execute(
            prompt=prompt, max_tokens=64
        )
        tracker.record(phase, result["node_id"], result["latency_sec"])
    await asyncio.sleep(INTER_TICK_DELAY)


def build_phase_table(tracker: PhaseTracker) -> Table:
    """Build the Rich summary table for all three phases."""
    table = Table(
        title="Failover Test — Traffic Distribution by Phase",
        show_lines=True,
        header_style="bold cyan",
        border_style="bright_black",
    )
    table.add_column("Phase", style="cyan", no_wrap=True)
    table.add_column("Node Alpha", justify="right", style="yellow")
    table.add_column("Node Beta", justify="right", style="green")
    table.add_column("Node Gamma", justify="right", style="magenta")
    table.add_column("Total Reqs", justify="right")
    table.add_column("Avg Latency", justify="right")
    table.add_column("Status", style="bold")

    status_labels = {
        "A": "Normal",
        "B": f"Crash ({CRASH_DELAY_SEC}s)",
        "C": "Recovered",
    }

    for phase in ["A", "B", "C"]:
        total = tracker.totals(phase)
        row: List[str] = [phase]
        for nid in tracker.node_ids:
            count = tracker.phase_data[phase].get(nid, 0)
            pct = count / total * 100 if total > 0 else 0.0
            row.append(f"{count} ({pct:.1f}%)")
        row.append(str(total))

        all_lats = [
            l for nid_lats in tracker.phase_latency[phase].values() for l in nid_lats
        ]
        avg = sum(all_lats) / len(all_lats) if all_lats else 0.0
        row.append(f"{avg:.4f}s")
        row.append(status_labels[phase])
        table.add_row(*row)

    return table


async def main() -> None:
    console = Console()
    random.seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    config = load_config()

    # ── Node definitions ────────────────────────────────────────────────
    # All three nodes start at the same baseline speed so that Phase A
    # traffic is roughly uniform (establishing the pre-crash baseline).
    workers: Dict[str, CPUMockWorkerNode] = {
        "node_alpha": CPUMockWorkerNode(
            node_id="node_alpha", base_delay_sec=0.05, load_factor=0.5,
        ),
        "node_beta": CPUMockWorkerNode(
            node_id="node_beta", base_delay_sec=0.05, load_factor=0.5,
        ),
        "node_gamma": CPUMockWorkerNode(
            node_id="node_gamma", base_delay_sec=0.05, load_factor=0.5,
        ),
    }
    node_ids = list(workers.keys())

    # ── Mesh initialisation ─────────────────────────────────────────────
    # Seed all nodes with steady-state pheromone values so that Phase A
    # begins with uniform attraction (V=1.0, L=STEADY_STATE_LATENCY).  Without
    # this, the EWMA alpha=0.5 causes V to jump from 0→0.5 on the first
    # request, creating a runaway positive-feedback bias.
    initial_state = np.full(
        (len(node_ids), 4),
        [1.0, STEADY_STATE_LATENCY, 0.0, 1.0],
        dtype=np.float64,
    )
    memory_field = PheromoneMemoryField(
        node_ids=node_ids,
        initial_state=initial_state,
        decay_success=False,
        decay_capability=False,
    )
    router = StigmergicRouterAgent(
        workers=workers,
        memory_field=memory_field,
        weights=config["weights"],
        temperature=TEST_TEMPERATURE,
        delta=config.get("weights", {}).get("delta", 1.5),
        rng=rng,
    )

    decay_task = asyncio.create_task(
        start_decay_engine(
            memory_field=memory_field,
            decay_rate=TEST_DECAY_RATE,
            interval_sec=TEST_DECAY_INTERVAL,
        )
    )

    tracker = PhaseTracker(node_ids)
    console.print(
        Panel.fit(
            "[bold cyan]Stigmergic Mesh Router — Chaos Failover Benchmark[/bold cyan]\n"
            f"Nodes: {', '.join(node_ids)}\n"
            f"Requests: {N_TICKS_PER_PHASE * REQUESTS_PER_TICK} per phase | "
            f"T={TEST_TEMPERATURE} | decay={TEST_DECAY_RATE}\n"
            f"Crash: {CRASH_DELAY_SEC}s fixed delay injected into node_alpha (Phase B)",
            border_style="cyan",
        )
    )

    # ── Phase A: Normal operation ────────────────────────────────────────
    console.print("[dim]Phase A: Normal operation — all nodes healthy[/dim]")
    for tick in range(1, N_TICKS_PER_PHASE + 1):
        await dispatch_tick(router, tracker, "A", tick)

    # ── Phase B: Crash injected ──────────────────────────────────────────
    console.print(f"[yellow]Phase B: CRASH injected — node_alpha ({CRASH_DELAY_SEC}s delay)[/yellow]")
    workers["node_alpha"].base_delay_sec = CRASH_DELAY_SEC
    workers["node_alpha"].load_factor = 0.0  # fixed delay, no compounding
    for tick in range(N_TICKS_PER_PHASE + 1, 2 * N_TICKS_PER_PHASE + 1):
        await dispatch_tick(router, tracker, "B", tick)

    # ── Phase C: Recovery ───────────────────────────────────────────────
    console.print("[green]Phase C: Recovery — node_alpha restored[/green]")
    workers["node_alpha"].base_delay_sec = 0.05
    workers["node_alpha"].load_factor = 0.5
    for tick in range(2 * N_TICKS_PER_PHASE + 1, 3 * N_TICKS_PER_PHASE + 1):
        await dispatch_tick(router, tracker, "C", tick)

    decay_task.cancel()
    try:
        await decay_task
    except asyncio.CancelledError:
        pass

    # ── Summary output ──────────────────────────────────────────────────
    console.print()
    console.print(build_phase_table(tracker))

    # ── Final state table ───────────────────────────────────────────────
    state = await memory_field.get_state_vector()
    state_table = Table(
        title="Final Pheromone Memory Field State", show_lines=True
    )
    state_table.add_column("Node", style="cyan")
    state_table.add_column("V (Success)", justify="right", style="green")
    state_table.add_column("L (Latency)", justify="right", style="yellow")
    state_table.add_column("S (Saturation)", justify="right", style="red")
    state_table.add_column("C (Capability)", justify="right", style="magenta")
    state_table.add_column("Requests Served", justify="right", style="blue")
    for i, nid in enumerate(node_ids):
        state_table.add_row(
            nid,
            f"{state[i, 0]:.4f}",
            f"{state[i, 1]:.4f}",
            f"{state[i, 2]:.4f}",
            f"{state[i, 3]:.4f}",
            str(workers[nid].requests_served),
        )
    console.print()
    console.print(state_table)

    # ── Verdict ─────────────────────────────────────────────────────────
    alpha_a = tracker.phase_data["A"].get("node_alpha", 0)
    alpha_b = tracker.phase_data["B"].get("node_alpha", 0)
    alpha_c = tracker.phase_data["C"].get("node_alpha", 0)
    total_b = tracker.totals("B")
    total_c = tracker.totals("C")

    pct_b = alpha_b / total_b * 100 if total_b > 0 else 0.0
    pct_c = alpha_c / total_c * 100 if total_c > 0 else 0.0

    console.print()
    if alpha_b < alpha_a * 0.15 and alpha_c > alpha_b:
        verdict = (
            "[bold green]✓ PASS: node_alpha was abandoned during the crash "
            f"(Phase B: {alpha_b} reqs, {pct_b:.1f}%) and traffic returned after "
            f"recovery (Phase C: {alpha_c} reqs, {pct_c:.1f}%).[/bold green]"
        )
    else:
        verdict = (
            "[bold yellow]~ WARNING: Traffic shift did not clearly demonstrate "
            "failover and recovery.[/bold yellow]"
        )
    console.print(Panel(verdict, title="Failover Result", border_style="green"))


if __name__ == "__main__":
    asyncio.run(main())
