#!/usr/bin/env python3
"""Capability-aware stigmergic routing benchmark.

Demonstrates that the extended 4D scpre (V, L, S, C) with capability-fit
tracing naturally routes requests to the most appropriate node based on
prompt semantics — without explicit model selection logic.

Benchmark design
────────────────
Three worker nodes are configured:

* **slm-fast**    — lightweight SLM, 0.03 s / request, tags = [slm, fast, low-latency]
* **llm-reasoner** — heavy LLM,   0.15 s / request, tags = [llm, reasoning]
* **llm-balanced**  — mid-range,   0.08 s / request, tags = [llm, balanced]

Two request types are dispatched:

1. **Short prompts** (e.g. "Hello", "What's the weather?") — capability
   context biases toward ``slm`` / ``low-latency``.  Expected routing:
   majority to ``slm-fast``.

2. **Deep reasoning prompts** (containing "think", "step by step",
   "let's think") — capability context biases toward ``llm`` /
   ``reasoning``.  Expected routing: majority to ``llm-reasoner``.

Run::

    python3 benchmarks/capability_routing_test.py
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.decay_engine import start_decay_engine
from core.memory_field import PheromoneMemoryField
from core.router_agent import StigmergicRouterAgent
from core.worker_node import CPUMockWorkerNode
from api.server import analyze_prompt

CONFIG_PATH = "config.yaml"
N_SHORT_PROMPTS = 60
N_REASONING_PROMPTS = 60
SEED = 42


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_test_workers() -> Dict[str, CPUMockWorkerNode]:
    """Create three mock workers with distinct capability profiles."""
    return {
        "slm-fast": CPUMockWorkerNode(
            node_id="slm-fast",
            base_delay_sec=0.03,
            load_factor=0.3,
            capability_tags=["slm", "fast", "low-latency"],
        ),
        "llm-reasoner": CPUMockWorkerNode(
            node_id="llm-reasoner",
            base_delay_sec=0.15,
            load_factor=0.5,
            capability_tags=["llm", "reasoning"],
        ),
        "llm-balanced": CPUMockWorkerNode(
            node_id="llm-balanced",
            base_delay_sec=0.08,
            load_factor=0.4,
            capability_tags=["llm", "balanced"],
        ),
    }


def build_state_table(table: Table, state: np.ndarray, node_ids: List[str]) -> None:
    for i, nid in enumerate(node_ids):
        table.add_row(
            nid,
            f"{state[i, 0]:.4f}",
            f"{state[i, 1]:.4f}",
            f"{state[i, 2]:.4f}",
            f"{state[i, 3]:.4f}",
        )


async def main() -> None:
    console = Console()
    random.seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    config = load_config()
    weights = config["weights"]

    workers = build_test_workers()
    node_ids = list(workers.keys())

    # Seed all nodes with steady-state so Phase A starts uniform
    initial_state = np.tile(
        np.array([1.0, 0.075, 0.0, 1.0], dtype=np.float64),
        (len(node_ids), 1),
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
        weights=weights,
        temperature=2.0,
        delta=weights.get("delta", 1.5),
        rng=rng,
    )

    decay_task = asyncio.create_task(
        start_decay_engine(
            memory_field=memory_field,
            decay_rate=0.15,
            interval_sec=0.5,
        )
    )

    console.print(
        Panel.fit(
            "[bold cyan]Capability-Aware Stigmergic Routing Benchmark[/bold cyan]\n"
            f"Nodes: {', '.join(node_ids)}\n"
            f"Short prompts: {N_SHORT_PROMPTS} | Reasoning prompts: {N_REASONING_PROMPTS}\n"
            f"Temperature: 2.0 | Decay: 0.15 | Delta: {weights.get('delta', 1.5)}\n"
            "Phase A: short prompts → expect slm-fast preference\n"
            "Phase B: reasoning prompts → expect llm-reasoner preference",
            border_style="cyan",
        )
    )

    short_prompts = [
        "Hello!",
        "What's the weather today?",
        "Tell me a joke.",
        "What time is it?",
        "Hi, how are you?",
        "Thanks!",
    ]

    reasoning_prompts = [
        "Let's think step by step about the implications of quantum computing.",
        "Please think through this problem carefully: what are the ethical implications of AI?",
        "Explain the chain of thought reasoning behind this mathematical proof.",
        "Analyze the reasoning steps behind this complex algorithm and explain each.",
        "Think deeply about why this code might be failing and propose a fix.",
        "Because the data shows a clear pattern, explain what this means about the underlying theory.",
    ]

    # ── Phase A: short prompts ─────────────────────────────────────────
    console.print("[dim]Phase A: Short prompts — expecting SLM routing[/dim]")
    short_counter = Counter()
    short_latencies: Dict[str, List[float]] = {nid: [] for nid in node_ids}

    for i in range(N_SHORT_PROMPTS):
        prompt = short_prompts[i % len(short_prompts)]
        cap_ctx = analyze_prompt(prompt)
        result = await router.route_and_execute(
            prompt=prompt, max_tokens=32, capability_context=cap_ctx
        )
        short_counter[result["node_id"]] += 1
        short_latencies[result["node_id"]].append(result["latency_sec"])

    state = await memory_field.get_state_vector()

    # ── Phase B: reasoning prompts ───────────────────────────────────────
    console.print("[yellow]Phase B: Deep reasoning prompts — expecting LLM routing[/yellow]")
    reasoning_counter = Counter()
    reasoning_latencies: Dict[str, List[float]] = {nid: [] for nid in node_ids}

    for i in range(N_REASONING_PROMPTS):
        prompt = reasoning_prompts[i % len(reasoning_prompts)]
        cap_ctx = analyze_prompt(prompt)
        result = await router.route_and_execute(
            prompt=prompt, max_tokens=128, capability_context=cap_ctx
        )
        reasoning_counter[result["node_id"]] += 1
        reasoning_latencies[result["node_id"]].append(result["latency_sec"])

    decay_task.cancel()
    try:
        await decay_task
    except asyncio.CancelledError:
        pass

    # ── Results tables ──────────────────────────────────────────────────
    console.print()
    table_a = Table(title="Phase A — Short Prompt Routing", show_lines=True)
    table_a.add_column("Node", style="cyan")
    table_a.add_column("Requests", justify="right", style="yellow")
    table_a.add_column("Share", justify="right", style="green")
    table_a.add_column("Avg Latency", justify="right", style="magenta")
    for nid in node_ids:
        count = short_counter.get(nid, 0)
        pct = count / N_SHORT_PROMPTS * 100
        lats = short_latencies[nid]
        avg_lat = sum(lats) / len(lats) if lats else 0.0
        table_a.add_row(nid, str(count), f"{pct:.1f}%", f"{avg_lat:.4f}s")

    table_b = Table(title="Phase B — Reasoning Prompt Routing", show_lines=True)
    table_b.add_column("Node", style="cyan")
    table_b.add_column("Requests", justify="right", style="yellow")
    table_b.add_column("Share", justify="right", style="green")
    table_b.add_column("Avg Latency", justify="right", style="magenta")
    for nid in node_ids:
        count = reasoning_counter.get(nid, 0)
        pct = count / N_REASONING_PROMPTS * 100
        lats = reasoning_latencies[nid]
        avg_lat = sum(lats) / len(lats) if lats else 0.0
        table_b.add_row(nid, str(count), f"{pct:.1f}%", f"{avg_lat:.4f}s")

    console.print(table_a)
    console.print()
    console.print(table_b)

    # ── Final state ─────────────────────────────────────────────────────
    state = await memory_field.get_state_vector()
    state_table = Table(title="Final Pheromone Memory Field State", show_lines=True)
    state_table.add_column("Node", style="cyan")
    state_table.add_column("V (Success)", justify="right", style="green")
    state_table.add_column("L (Latency)", justify="right", style="yellow")
    state_table.add_column("S (Saturation)", justify="right", style="red")
    state_table.add_column("C (Capability)", justify="right", style="magenta")
    for i, nid in enumerate(node_ids):
        state_table.add_row(
            nid,
            f"{state[i, 0]:.4f}",
            f"{state[i, 1]:.4f}",
            f"{state[i, 2]:.4f}",
            f"{state[i, 3]:.4f}",
        )
    console.print()
    console.print(state_table)

    # ── Verdict ─────────────────────────────────────────────────────────
    slm_share_a = short_counter.get("slm-fast", 0) / N_SHORT_PROMPTS * 100
    llm_share_b = reasoning_counter.get("llm-reasoner", 0) / N_REASONING_PROMPTS * 100

    console.print()
    if slm_share_a > 45 and llm_share_b > 45:
        verdict = (
            f"[bold green]✓ PASS: Short prompts routed to SLM ({slm_share_a:.1f}% "
            f"to slm-fast) and reasoning prompts routed to LLM ({llm_share_b:.1f}% "
            "to llm-reasoner).[/bold green]"
        )
    else:
        verdict = (
            f"[bold yellow]~ PARTIAL: SLM share={slm_share_a:.1f}%, "
            f"LLM reasoning share={llm_share_b:.1f}%[/bold yellow]"
        )
    console.print(Panel(verdict, title="Capability Routing Result", border_style="cyan"))


if __name__ == "__main__":
    asyncio.run(main())
