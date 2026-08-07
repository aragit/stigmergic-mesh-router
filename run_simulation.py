#!/usr/bin/env python3
"""Stigmergic Mesh Router — Phase 1 & 2 verification simulation.

Instantiates a mock cluster of three CPU worker nodes (two fast, one slow),
starts the background pheromone decay engine, dispatches a batch of
concurrent inference requests through the :class:`StigmergicRouterAgent`,
and prints a formatted summary demonstrating traffic avoidance of the
slow node.
"""

import asyncio
import random
from collections import Counter
from typing import Dict

import numpy as np
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core.decay_engine import start_decay_engine
from core.memory_field import PheromoneMemoryField
from core.router_agent import StigmergicRouterAgent
from core.worker_node import CPUMockWorkerNode

CONFIG_PATH = "config.yaml"
N_REQUESTS = 30
CONCURRENCY = 6
SEED = 42


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


async def main() -> None:
    config = load_config()

    # --- Mock cluster: 2 fast nodes, 1 slow node -------------------------
    worker_specs = {
        "cpu_fast_0": {"base_delay_sec": 0.05, "load_factor": 0.5},
        "cpu_fast_1": {"base_delay_sec": 0.05, "load_factor": 0.5},
        "cpu_slow_0": {"base_delay_sec": 0.30, "load_factor": 0.5},
    }
    workers: Dict[str, CPUMockWorkerNode] = {
        nid: CPUMockWorkerNode(node_id=nid, **cfg)
        for nid, cfg in worker_specs.items()
    }
    node_ids = list(workers.keys())

    # --- Memory field & router -------------------------------------------
    rng = np.random.default_rng(SEED)
    random.seed(SEED)
    memory_field = PheromoneMemoryField(node_ids=node_ids)
    router = StigmergicRouterAgent(
        workers=workers,
        memory_field=memory_field,
        weights=config["weights"],
        temperature=config["temperature"],
        rng=rng,
    )

    # --- Decay engine (background) ---------------------------------------
    decay_task = asyncio.create_task(
        start_decay_engine(
            memory_field=memory_field,
            decay_rate=config["decay_rate"],
            interval_sec=config["decay_interval_sec"],
        )
    )

    console = Console()

    console.print(
        Panel.fit(
            "[bold cyan]Stigmergic Mesh Router — Simulation[/bold cyan]\n"
            f"Nodes: {', '.join(node_ids)}\n"
            f"Requests: {N_REQUESTS} | Concurrency: {CONCURRENCY}\n"
            f"Temperature: {config['temperature']} | Decay rate: {config['decay_rate']}",
            border_style="cyan",
        )
    )

    # --- Dispatch 30 concurrent requests with bounded concurrency ---------
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def dispatch(i: int) -> str:
        async with semaphore:
            prompt = (
                f"Explain the concept of stigmergy in distributed systems. "
                f"Request #{i}."
            )
            result = await router.route(prompt=prompt, max_tokens=128)
            return result["routed_to"]

    tasks = [asyncio.create_task(dispatch(i)) for i in range(N_REQUESTS)]
    routed_to = await asyncio.gather(*tasks)
    distribution = Counter(routed_to)

    # --- Stop the decay engine -------------------------------------------
    decay_task.cancel()
    try:
        await decay_task
    except asyncio.CancelledError:
        pass

    # --- Final memory field state ----------------------------------------
    state = await memory_field.get_state_vector()

    # --- Pretty-print summary --------------------------------------------
    table = Table(title="Request Distribution per Node", show_lines=True)
    table.add_column("Node", style="cyan", no_wrap=True)
    table.add_column("Requests", justify="right", style="magenta")
    table.add_column("Share", justify="right", style="yellow")
    table.add_column("Base Delay", justify="right", style="green")
    table.add_column("Avg Latency", justify="right", style="green")
    table.add_column("Requests Served", justify="right", style="blue")

    fast_total = 0
    slow_total = 0
    for nid in sorted(node_ids):
        count = distribution.get(nid, 0)
        pct = count / N_REQUESTS * 100
        worker = workers[nid]
        base_delay = worker.base_delay_sec
        served = worker.requests_served
        avg_latency = state[workers and node_ids.index(nid), 1]
        table.add_row(
            nid,
            str(count),
            f"{pct:.1f}%",
            f"{base_delay:.3f}s",
            f"{avg_latency:.4f}s",
            str(served),
        )

    fast_total = sum(distribution.get(n, 0) for n in ["cpu_fast_0", "cpu_fast_1"])
    slow_total = distribution.get("cpu_slow_0", 0)

    console.print()
    console.print(table)

    console.print()
    state_table = Table(title="Final Pheromone Memory Field State")
    state_table.add_column("Node", style="cyan")
    state_table.add_column("V (Success)", justify="right", style="green")
    state_table.add_column("L (Latency)", justify="right", style="yellow")
    state_table.add_column("S (Saturation)", justify="right", style="red")
    for i, nid in enumerate(node_ids):
        state_table.add_row(
            nid,
            f"{state[i, 0]:.4f}",
            f"{state[i, 1]:.4f}",
            f"{state[i, 2]:.4f}",
        )
    console.print(state_table)

    console.print()
    analysis = (
        f"Fast nodes combined: {fast_total}/{N_REQUESTS} "
        f"({fast_total / N_REQUESTS * 100:.1f}%)\n"
        f"Slow node (cpu_slow_0): {slow_total}/{N_REQUESTS} "
        f"({slow_total / N_REQUESTS * 100:.1f}%)"
    )
    if slow_total < fast_total / 3:
        verdict = "[bold green]✓ Traffic successfully avoided the slow node![/bold green]"
    else:
        verdict = "[bold yellow]~ Slow node still receiving notable traffic.[/bold yellow]"
    console.print(Panel(analysis + "\n" + verdict, title="Traffic Analysis"))


if __name__ == "__main__":
    asyncio.run(main())
