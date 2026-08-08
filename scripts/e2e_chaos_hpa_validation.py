#!/usr/bin/env python3
"""Phase 13 - E2E Chaos & HPA validation suite.

Runs three phases against a live stigmergic-mesh cluster to prove the
ArgoCD-managed production profile (Phase 11 security + Phase 12 checkpointing +
Phase 13 autoscaling) survives synthetic traffic, pod failures, and tenant
quota breaches.

Import-safe without the kubernetes Python client. verify_hpa_scaling and
inject_pod_chaos accept a k8s_client so the unit tests stay hermetic.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

import httpx

DEFAULT_BASE_URL = os.environ.get(
    "STIGMERGIC_E2E_BASE_URL", "http://localhost:8080"
)
DEFAULT_NAMESPACE = os.environ.get(
    "STIGMERGIC_E2E_NAMESPACE", "stigmergic-mesh"
)
HPA_NAME = os.environ.get(
    "STIGMERGIC_E2E_HPA_NAME", "stigmergic-mesh-router"
)
POD_LABEL = os.environ.get(
    "STIGMERGIC_E2E_POD_LABEL",
    "app.kubernetes.io/name=stigmergic-mesh-router",
)
COMPLETION_PATH = "/v1/completions"
METRICS_PATH = "/metrics"

HPA_MIN_REPLICAS = 2
HPA_MAX_REPLICAS = 10


@dataclass
class TrafficStats:
    """Aggregate results of a simulate_tenant_traffic burst."""

    tenant_id: str
    issued: int = 0
    succeeded: int = 0
    rate_limited: int = 0
    errors: int = 0
    durations: List[float] = field(default_factory=list)


@dataclass
class ScalingResult:
    """Snapshot of the HPA replica state."""

    current_replicas: Optional[int]
    desired_replicas: int
    min_replicas: int
    max_replicas: int

    def is_scaled_up(self) -> bool:
        return (
            self.current_replicas is not None
            and self.current_replicas > self.min_replicas
        )

    def within_bounds(self) -> bool:
        if self.current_replicas is None:
            return False
        return self.min_replicas <= self.current_replicas <= self.max_replicas


def _default_k8s_client(kubeconfig: Optional[str] = None) -> Any:
    """Lazily build an AppsV1Api client (kubernetes is an optional dep)."""
    try:
        from kubernetes import client, config  # type: ignore
    except ImportError as exc:  # pragma: no cover - faked in tests
        raise RuntimeError(
            "install the 'kubernetes' client to query the cluster directly"
        ) from exc
    if kubeconfig:
        config.load_kube_config(config_file=kubeconfig)
    else:
        config.load_incluster_config()
    return client.AppsV1Api()


# --- Phase A: load generation ------------------------------------------------


async def simulate_tenant_traffic(
    client: httpx.AsyncClient,
    tenant_id: str,
    api_key: str,
    rps: float,
    duration_seconds: float,
    prompt: str = "explain stigmergic routing",
    max_tokens: int = 128,
) -> TrafficStats:
    """Issue OpenAI-style completions at *rps* for *duration_seconds*.

    HTTP 429 and transport errors are tallied but never abort the burst so
    callers can observe quota behaviour under load.
    """
    stats = TrafficStats(tenant_id=tenant_id)
    delay = 1.0 / rps if rps > 0 else 0.0
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "stigmergic-mesh",
        "prompt": prompt,
        "max_tokens": max_tokens,
    }
    deadline = time.monotonic() + duration_seconds

    async def _one_request() -> None:
        stats.issued += 1
        start = time.monotonic()
        try:
            resp = await client.post(
                COMPLETION_PATH, headers=headers, json=payload, timeout=10.0
            )
        except httpx.HTTPError:
            stats.errors += 1
            return
        stats.durations.append(time.monotonic() - start)
        if resp.status_code == 200:
            stats.succeeded += 1
        elif resp.status_code == 429:
            stats.rate_limited += 1
        else:
            stats.errors += 1

    tasks: List[asyncio.Task] = []
    while time.monotonic() < deadline:
        tasks.append(asyncio.create_task(_one_request()))
        if delay:
            await asyncio.sleep(delay)

    if tasks:
        await asyncio.gather(*tasks)
    return stats


# --- Phase B: autoscaling introspection -------------------------------------


def verify_hpa_scaling(
    hpa_name: str = HPA_NAME,
    namespace: str = DEFAULT_NAMESPACE,
    kubeconfig: Optional[str] = None,
    k8s_client: Optional[Any] = None,
    min_replicas: int = HPA_MIN_REPLICAS,
    max_replicas: int = HPA_MAX_REPLICAS,
) -> ScalingResult:
    """Read the live HPA and assert replicas land within bounds."""
    if k8s_client is None:
        apps_api = _default_k8s_client(kubeconfig)
    else:
        apps_api = k8s_client
    try:
        hpa = apps_api.read_namespaced_horizontal_pod_autoscaler(
            hpa_name, namespace
        )
        current = hpa.status.current_replicas
        desired = hpa.status.desired_replicas
    except Exception:
        current, desired = None, min_replicas
    result = ScalingResult(
        current_replicas=current,
        desired_replicas=desired,
        min_replicas=min_replicas,
        max_replicas=max_replicas,
    )
    assert result.within_bounds(), (
        f"HPA replicas {current} outside [{min_replicas}, {max_replicas}]"
    )
    return result


def inject_pod_chaos(
    namespace: str = DEFAULT_NAMESPACE,
    label_selector: str = POD_LABEL,
    kubeconfig: Optional[str] = None,
    k8s_client: Optional[Any] = None,
) -> List[str]:
    """Terminate router pods matching *label_selector* to force rescheduling."""
    if k8s_client is None:
        from kubernetes import client  # type: ignore

        core_api = client.CoreV1Api()
    else:
        core_api = k8s_client
    pods = core_api.list_namespaced_pod(namespace, label_selector=label_selector)
    names: List[str] = []
    for pod in pods.items:
        name = pod.metadata.name
        core_api.delete_namespaced_pod(name, namespace)
        names.append(name)
    assert names, f"no pods matched selector '{label_selector}' in {namespace}"
    return names


# --- Phase C: governance assertions ------------------------------------------


async def fetch_metric(
    client: httpx.AsyncClient, metric: str
) -> Optional[float]:
    """Parse a single Prometheus gauge/counter from the metrics endpoint."""
    resp = await client.get(METRICS_PATH, timeout=5.0)
    if resp.status_code != 200:
        return None
    for line in resp.text.splitlines():
        if line.startswith(metric) and not line.startswith(metric + "_"):
            try:
                return float(line.split()[-1])
            except (ValueError, IndexError):
                return None
    return None


async def verify_warmstart(
    client: httpx.AsyncClient,
) -> bool:
    """Assert a freshly-scheduled pod restored state from a checkpoint."""
    value = await fetch_metric(client, "stigmergic_checkpoint_restore_status")
    assert value == 1, (
        f"expected warm-start restore, stigmergic_checkpoint_restore_status={value}"
    )
    return True


async def verify_rate_limit(
    client: httpx.AsyncClient,
    tenant_id: str,
    api_key: str,
    prompts: int = 20,
) -> int:
    """Burst *prompts* requests and return the number of 429 responses seen."""
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "stigmergic-mesh",
        "prompt": "quota stress test",
        "max_tokens": 128,
    }
    sem = asyncio.Semaphore(prompts)

    async def _hit() -> int:
        async with sem:
            try:
                resp = await client.post(
                    COMPLETION_PATH, headers=headers, json=payload, timeout=5.0
                )
            except httpx.HTTPError:
                return 0
            return 1 if resp.status_code == 429 else 0

    throttled = sum(await asyncio.gather(*[_hit() for _ in range(prompts)]))
    assert throttled > 0, "rate limit (429) was never exceeded under burst"
    return throttled


# --- Orchestrator ------------------------------------------------------------


async def run_phases(
    base_url: str = DEFAULT_BASE_URL,
    namespace: str = DEFAULT_NAMESPACE,
    tenant_api_key: str = "",
    kubeconfig: Optional[str] = None,
    k8s_client: Optional[Any] = None,
    concurrent_tenants: int = 10,
    duration_seconds: float = 5.0,
) -> None:
    """Execute Phase A (load) + Phase B (chaos) + Phase C (governance).

    Phase A fans out *concurrent_tenants* independent traffic bursts so the
    aggregate load is realistic for multi-tenant autoscaling.
    """
    print(f"==> Phase A: injecting high-concurrency traffic to {base_url}")
    async with httpx.AsyncClient(base_url=base_url) as client:

        async def _tenant(idx: int) -> TrafficStats:
            stats = await simulate_tenant_traffic(
                client=client,
                tenant_id=f"tenant-{idx}",
                api_key=tenant_api_key,
                rps=50.0,
                duration_seconds=duration_seconds,
            )
            print(
                f"   [tenant-{idx}] issued={stats.issued} "
                f"ok={stats.succeeded} 429={stats.rate_limited} "
                f"err={stats.errors}"
            )
            return stats

        await asyncio.gather(*[_tenant(i) for i in range(concurrent_tenants)])

    scaling = verify_hpa_scaling(
        namespace=namespace, kubeconfig=kubeconfig, k8s_client=k8s_client
    )
    print(
        f"==> Phase A: HPA current={scaling.current_replicas} "
        f"desired={scaling.desired_replicas}"
    )
    if not scaling.is_scaled_up():
        print("   (warning: HPA did not scale up; cluster load may be low)")

    print("==> Phase B: injecting pod chaos")
    killed = inject_pod_chaos(
        namespace=namespace, kubeconfig=kubeconfig, k8s_client=k8s_client
    )
    print(f"   terminated pods: {killed}")
    await asyncio.sleep(2.0)

    print("==> Phase C: rate-limit + warm-start assertions")
    async with httpx.AsyncClient(base_url=base_url) as client:
        throttled = await verify_rate_limit(
            client=client,
            tenant_id="tenant-prod",
            api_key=tenant_api_key,
        )
        await verify_warmstart(client=client)
    print(f"   rate-limited responses: {throttled} (warm-start verified)")


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point.

    Usage:
      python scripts/e2e_chaos_hpa_validation.py \
        --namespace stigmergic-mesh \
        --target-host http://localhost:8000 \
        --concurrent-tenants 10 \
        --duration 120
    """
    parser = argparse.ArgumentParser(
        prog="e2e_chaos_hpa_validation",
        description=(
            "Phase 13 E2E: load injection, HPA scaling check, pod chaos, "
            "rate-limit and warm-start assertions."
        ),
    )
    parser.add_argument(
        "--target-host", default=DEFAULT_BASE_URL, help="Router base URL."
    )
    parser.add_argument(
        "--namespace", default=DEFAULT_NAMESPACE, help="K8s namespace."
    )
    parser.add_argument(
        "--tenant-api-key", default="", help="Bearer token for tenant auth."
    )
    parser.add_argument(
        "--concurrent-tenants",
        type=int,
        default=10,
        help="Number of concurrent tenant traffic bursts (Phase A).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Per-tenant burst duration in seconds (Phase A).",
    )
    parser.add_argument(
        "--kubeconfig", default=None, help="Path to kubeconfig file."
    )
    args = parser.parse_args(argv)
    asyncio.run(
        run_phases(
            base_url=args.target_host,
            namespace=args.namespace,
            tenant_api_key=args.tenant_api_key,
            kubeconfig=args.kubeconfig,
            concurrent_tenants=args.concurrent_tenants,
            duration_seconds=args.duration,
        )
    )


if __name__ == "__main__":
    main()
