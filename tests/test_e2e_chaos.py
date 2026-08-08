"""Unit tests for the Phase 13 E2E chaos/HPA validation script.

Loads ``scripts/e2e_chaos_hpa_validation.py`` without the ``kubernetes``
dependency and exercises every phase with httpx ``MockTransport`` and a
lightweight fake K8s client so the suite is fully hermetic.
"""

import importlib.util
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "e2e_chaos_hpa_validation.py"


@pytest.fixture(scope="module")
def m():
    """Load the E2E script as an importable module.

    Registered in ``sys.modules`` so dataclass field-type resolution works.
    """
    spec = importlib.util.spec_from_file_location("e2e_chaos", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["e2e_chaos"] = module
    spec.loader.exec_module(module)
    return module


# --- fakes --------------------------------------------------------------------


class _Status:
    def __init__(self, current, desired):
        self.current_replicas = current
        self.desired_replicas = desired


class _HPA:
    def __init__(self, current, desired):
        self.status = _Status(current, desired)


class FakeAppsApi:
    def __init__(self, hpa):
        self._hpa = hpa
        self.calls = 0

    def read_namespaced_horizontal_pod_autoscaler(self, name, namespace):
        self.calls += 1
        return self._hpa


class _PodMeta:
    def __init__(self, name):
        self.name = name


class _Pod:
    def __init__(self, name):
        self.metadata = _PodMeta(name)


class _PodList:
    def __init__(self, pods):
        self.items = pods


class FakeCoreApi:
    def __init__(self, pods):
        self._pods = pods
        self.deleted = []

    def list_namespaced_pod(self, namespace, label_selector=None):
        return _PodList(self._pods)

    def delete_namespaced_pod(self, name, namespace):
        self.deleted.append(name)


# --- Phase A: traffic -------------------------------------------------------


async def test_simulate_tenant_traffic_counts_statuses(m):
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        code = 200 if call_count["n"] % 2 == 0 else 429
        return httpx.Response(code, text='{"object":"text_completion"}')

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        stats = await m.simulate_tenant_traffic(
            client, "tenant-prod", "key", rps=100.0, duration_seconds=0.5
        )

    assert stats.issued == call_count["n"]
    assert stats.succeeded > 0
    assert stats.rate_limited > 0
    assert stats.succeeded + stats.rate_limited + stats.errors == stats.issued


# --- Phase B: autoscaling ---------------------------------------------------


async def test_verify_hpa_scaling_within_bounds(m):
    hpa = _HPA(current=4, desired=4)
    api = FakeAppsApi(hpa)
    result = m.verify_hpa_scaling(k8s_client=api)
    assert result.current_replicas == 4
    assert result.desired_replicas == 4
    assert result.is_scaled_up() is True
    assert api.calls == 1


async def test_verify_hpa_scaling_raises_out_of_bounds(m):
    hpa = _HPA(current=0, desired=0)
    api = FakeAppsApi(hpa)
    with pytest.raises(AssertionError):
        m.verify_hpa_scaling(k8s_client=api)


async def test_inject_pod_chaos_terminates_matching_pods(m):
    pods = [_Pod("router-1"), _Pod("router-2")]
    api = FakeCoreApi(pods)
    killed = m.inject_pod_chaos(label_selector="app=router", k8s_client=api)
    assert killed == ["router-1", "router-2"]
    assert api.deleted == ["router-1", "router-2"]


async def test_inject_pod_chaos_raises_when_no_match(m):
    api = FakeCoreApi([])
    with pytest.raises(AssertionError):
        m.inject_pod_chaos(label_selector="nope", k8s_client=api)


# --- Phase C: governance ----------------------------------------------------


def _metrics_transport(restore_value):
    text = (
        "# HELP stigmergic_checkpoint_restore_status warm start marker\n"
        "# TYPE stigmergic_checkpoint_restore_status gauge\n"
        f"stigmergic_checkpoint_restore_status {restore_value}\n"
        "stigmergic_requests_total{node_id=n1,status=200} 42\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=text)

    return httpx.MockTransport(handler)


async def test_fetch_metric_parses_exact_gauge(m):
    transport = _metrics_transport(1)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        value = await m.fetch_metric(client, "stigmergic_checkpoint_restore_status")
    assert value == 1.0


async def test_verify_warmstart_ok(m):
    transport = _metrics_transport(1)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        assert await m.verify_warmstart(client) is True


async def test_verify_warmstart_raises_on_missing_restore(m):
    transport = _metrics_transport(0)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        with pytest.raises(AssertionError):
            await m.verify_warmstart(client)


async def test_verify_rate_limit_detects_429(m):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text='{"detail":"rate limited"}')

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        throttled = await m.verify_rate_limit(
            client, "tenant-prod", "key", prompts=5
        )
    assert throttled == 5
