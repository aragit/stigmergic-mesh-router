#!/usr/bin/env python3
"""Locust load-testing harness for the stigmergic mesh router.

Simulates realistic client traffic against the OpenAI-compatible API
endpoints.  Two user profiles generate traffic:

* **FastQueryUser** (weight 70): short, simple prompts that exercise the
  SLM routing path (``slm`` / ``fast`` / ``low-latency`` capability tags).
* **ReasoningQueryUser** (weight 30): long prompts containing chain-of-
  thought instructions that exercise the LLM routing path (``llm`` /
  ``reasoning`` capability tags).

A custom event handler reports p95/p99 latency and throughput at the
end of the run.

Usage::

    locust -f benchmarks/locustfile.py --headless -u 20 -r 5 --run-time 30s --host http://localhost:18000

Interactive (web UI)::

    locust -f benchmarks/locustfile.py --host http://localhost:18000
    # then open http://localhost:8089
"""

import random
import time

from locust import (
    HttpUser,
    events,
    task,
    between,
)

# ── Prompt pools ──────────────────────────────────────────────────────────────

FAST_PROMPTS = [
    "Hello!",
    "What time is it?",
    "Hi",
    "Good morning",
    "How are you?",
    "What's up?",
    "Hello world",
    "Test",
    "OK",
    "Thanks",
    "Please summarize this.",
    "Quick check.",
    "Are you there?",
    "Brief status report.",
    "One-word answer: ok?",
]

REASONING_PROMPTS = [
    "Let's think step by step: Explain how backpropagation works in neural networks.",
    "Let's think step by step: Derive the time and space complexity of merge sort.",
    "Please reason carefully: What are the trade-offs between supervised and unsupervised learning?",
    "Let's think step by step: Implement a binary search tree insert in Python.",
    "You are a helpful assistant. Let's think step by step: What happens during the encoding process in transformers?",
    "Let's think step by step: Solve the Monty Hall problem and explain your reasoning.",
    "Please think carefully and explain step by step: How does gradient descent optimization work?",
    "Let's think step by step: Compare LSTM and GRU architectures in detail.",
    "Reason step by step: What are eigenvalues and eigenvectors, and why are they important?",
    "Let's think step by step: Explain the bias-variance tradeoff with an example.",
]

# Pre-recorded metrics for end-of-run reporting
_run_start_time: float = 0.0
_request_latencies: list = []


# ── Event handlers for custom reporting ──────────────────────────────────────

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when the load test starts."""
    global _run_start_time, _request_latencies
    _run_start_time = time.time()
    _request_latencies = []


@events.request.add_listener
def on_request(
    request_type, name, response_time, response_length,
    exception, **kwargs,
):
    """Accumulate per-request latencies for custom p95/p99 computation."""
    if exception is None:
        _request_latencies.append(response_time / 1000.0)  # ms → seconds


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Print p95, p99, and throughput at end of run."""
    global _request_latencies
    if not _request_latencies:
        print("\n[Metrics] No successful requests recorded.")
        return

    sorted_lat = sorted(_request_latencies)
    n = len(sorted_lat)
    p50 = sorted_lat[int(n * 0.50)]
    p95 = sorted_lat[int(n * 0.95)]
    p99 = sorted_lat[int(n * 0.99)]
    max_lat = sorted_lat[-1]
    min_lat = sorted_lat[0]
    mean_lat = sum(sorted_lat) / n

    duration = time.time() - _run_start_time
    rps = n / duration if duration > 0 else 0

    print("\n" + "=" * 60)
    print("Stigmergic Mesh Router — Load Test Results")
    print("=" * 60)
    print(f"  Total Requests:     {n}")
    print(f"  Test Duration:      {duration:.1f}s")
    print(f"  Throughput (RPS):   {rps:.2f}")
    print()
    print("  Latency Distribution (seconds):")
    print(f"    Min:    {min_lat:.4f}s")
    print(f"    Mean:   {mean_lat:.4f}s")
    print(f"    p50:    {p50:.4f}s")
    print(f"    p95:    {p95:.4f}s")
    print(f"    p99:    {p99:.4f}s")
    print(f"    Max:    {max_lat:.4f}s")
    print("=" * 60 + "\n")


# ── User profiles ─────────────────────────────────────────────────────────────

class FastQueryUser(HttpUser):
    """Simulates short, simple queries routed to SLM/low-latency nodes.

    Uses a tight wait between requests to generate high request volume.
    """

    wait_time = between(0.1, 0.3)
    weight = 70

    @task
    def short_query(self):
        prompt = random.choice(FAST_PROMPTS)
        self.client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 32,
            },
        )


class ReasoningQueryUser(HttpUser):
    """Simulates deep-reasoning queries routed to LLM/reasoning nodes.

    Higher max_tokens and longer prompts exercise the C trace routing.
    """

    wait_time = between(0.5, 1.0)
    weight = 30

    @task
    def reasoning_query(self):
        prompt = random.choice(REASONING_PROMPTS)
        self.client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 128,
            },
        )
