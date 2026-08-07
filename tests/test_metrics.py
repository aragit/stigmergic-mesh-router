"""Unit tests for Prometheus metrics integration.

Verifies that:
* ``GET /metrics`` returns HTTP 200 with standard Prometheus text format.
* All expected gauge names (``stigmergic_node_success_trace``, etc.)
  appear in the metrics output after the mesh is initialised.
* Routing a request increments ``stigmergic_requests_total``.
* Duration histograms are observed.
"""

import asyncio
import os
import yaml
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

# Set STIGMERGIC_STORAGE_BACKEND to in_memory to avoid requiring Redis
os.environ.pop("STIGMERGIC_STORAGE_BACKEND", None)
os.environ.pop("STIGMERGIC_REDIS_HOST", None)

from api.server import app


@pytest.fixture
def client():
    """Create a TestClient for the FastAPI app.

    The lifespan context manager initialises the mesh on enter.
    """
    with TestClient(app) as c:
        yield c


def test_metrics_endpoint_returns_200(client):
    """GET /metrics should return HTTP 200."""
    resp = client.get("/metrics")
    assert resp.status_code == 200


def test_metrics_output_contains_expected_gauges(client):
    """The metrics output should contain all expected stigmergic gauges."""
    resp = client.get("/metrics")
    text = resp.text

    expected_names = [
        "stigmergic_node_success_trace",
        "stigmergic_node_latency_trace",
        "stigmergic_node_saturation_trace",
        "stigmergic_node_capability_trace",
        "stigmergic_node_attraction_score",
        "stigmergic_requests_total",
        "stigmergic_request_duration_seconds",
    ]

    for name in expected_names:
        assert name in text, f"Metric '{name}' not found in /metrics output"


def test_metrics_contains_node_labels(client):
    """Metrics should contain entries for each configured node."""
    resp = client.get("/metrics")
    text = resp.text

    # Read config to get expected node IDs
    config_path = ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        worker_specs = cfg.get("server", {}).get("workers", [])
    else:
        worker_specs = [
            {"node_id": "mock_node_0", "type": "mock"},
            {"node_id": "mock_node_1", "type": "mock"},
            {"node_id": "mock_node_2", "type": "mock"},
        ]

    for spec in worker_specs:
        node_id = spec["node_id"]
        # At least the success trace gauge should have this node_id label
        assert f'stigmergic_node_success_trace{{node_id="{node_id}"}}' in text, (
            f"Metric for node '{node_id}' not found"
        )


def test_routing_increments_request_counter(client):
    """After routing a request, stigmergic_requests_total should show a
    non-zero count for at least one node_id."""
    # First, capture the metrics before routing
    resp_before = client.get("/metrics")
    text_before = resp_before.text

    # Send a chat completion request
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Hello!"}],
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 200

    # Wait a brief moment for the metrics background loop to sync
    import time
    time.sleep(1.5)

    # Check metrics after routing
    resp_after = client.get("/metrics")
    text_after = resp_after.text

    # The requests_total counter should have increased
    # Look for any stigmergic_requests_total line with a count > 0
    found_increment = False
    for line in text_after.splitlines():
        if line.startswith("stigmergic_requests_total{") and not line.endswith(" 0.0"):
            found_increment = True
            break

    assert found_increment, (
        "stigmergic_requests_total did not show any non-zero counts "
        "after routing a request"
    )


def test_duration_histogram_observed(client):
    """After routing a request, the duration histogram should have
    observations."""
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Quick test"}],
            "max_tokens": 16,
        },
    )
    assert resp.status_code == 200

    # Wait for metrics loop
    import time
    time.sleep(1.5)

    resp_metrics = client.get("/metrics")
    text = resp_metrics.text

    # Look for histogram bucket lines or _sum / _count for the duration metric
    has_histogram = (
        "stigmergic_request_duration_seconds_bucket{" in text
        or "stigmergic_request_duration_seconds_sum{" in text
        or "stigmergic_request_duration_seconds_count{" in text
    )
    assert has_histogram, "Duration histogram metrics not found in output"
