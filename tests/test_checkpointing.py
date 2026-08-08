"""Phase 12 tests: pheromone state checkpointing & cold-start warmup.

Covers:
* ``MatrixSnapshot`` serialization / schema compliance.
* ``CheckpointManager.create_snapshot`` extracts V/L/S/C per node.
* Disk and Redis (fakeredis) save + load round-trips.
* ``MemoryField`` ``export_state_dict`` / ``hydrate_from_snapshot`` round-trip.
* Cold-start fallback (no checkpoint present).
* FastAPI ``_warmup_checkpoint`` warm-start hydration when enabled.
* Administrative CLI: ``export``, ``import``, ``inspect``.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import fakeredis.aioredis
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.pop("STIGMERGIC_STORAGE_BACKEND", None)
os.environ.pop("STIGMERGIC_REDIS_HOST", None)

import api.server as server_mod  # noqa: E402
from api.metrics import stigmergic_checkpoint_restore_status  # noqa: E402
from cli import checkpoint_ctl  # noqa: E402
from core.checkpointing import CheckpointManager, MatrixSnapshot  # noqa: E402
from core.memory_field import (  # noqa: E402
    InMemoryPheromoneMemoryField,
    RedisPheromoneMemoryField,
)
from core.router_agent import StigmergicRouterAgent  # noqa: E402

MOCK_NODES = ["mock_node_0", "mock_node_1", "mock_node_2"]


def _gauge_value() -> int:
    """Read the label-less restore-status gauge value."""
    try:
        return int(stigmergic_checkpoint_restore_status._value.get())
    except Exception:
        return 0


# ── Snapshot schema ─────────────────────────────────────────────────────

def test_matrix_snapshot_roundtrip():
    """A snapshot survives JSON serialisation with all fields intact."""
    snap = MatrixSnapshot(
        timestamp=1234.0,
        version="1.0",
        node_metrics={
            "n0": {"V": 1.0, "L": 0.2, "S": 0.1, "C": 0.9},
            "n1": {"V": 0.7, "L": 0.4, "S": 0.2, "C": 0.5},
        },
        entropy_rate=0.34,
        total_routed_requests=42,
    )
    data = snap.model_dump_json()
    restored = MatrixSnapshot.model_validate_json(data)

    assert restored.version == "1.0"
    assert restored.timestamp == 1234.0
    assert restored.entropy_rate == pytest.approx(0.34)
    assert restored.total_routed_requests == 42
    assert restored.node_metrics["n0"]["V"] == 1.0
    assert restored.node_metrics["n1"]["C"] == 0.5


def test_matrix_snapshot_defaults():
    """Optional fields default sensibly when omitted."""
    snap = MatrixSnapshot(timestamp=1.0)
    assert snap.version == "1.0"
    assert snap.node_metrics == {}
    assert snap.entropy_rate == 0.0
    assert snap.total_routed_requests == 0


# ── Snapshot creation & persistence round-trips ─────────────────────────

@pytest.fixture
def memory_field():
    return InMemoryPheromoneMemoryField(node_ids=["n0", "n1", "n2"])


@pytest.fixture
def router(memory_field):
    return StigmergicRouterAgent(
        workers={},
        memory_field=memory_field,
        weights={"alpha": 1.0, "beta": 2.0, "gamma": 1.5, "delta": 1.5},
        temperature=0.5,
        delta=1.5,
    )


@pytest.fixture
async def fake_redis():
    return fakeredis.aioredis.FakeRedis()


@pytest.mark.asyncio
async def test_create_snapshot_matches_memory_field(memory_field, router, tmp_path):
    """Snapshot node_metrics must reflect the live memory field state."""
    await memory_field.deposit_trace(
        "n0", latency_sec=0.5, tokens=10, success=True, active_load=2,
        capability_match=0.8,
    )
    cm = CheckpointManager(
        memory_field=memory_field, storage_path=str(tmp_path), router_agent=router,
    )
    snap = await cm.create_snapshot()

    exported = await memory_field.export_state_dict()
    assert set(snap.node_metrics) == {"n0", "n1", "n2"}
    for nid in exported:
        assert snap.node_metrics[nid]["V"] == pytest.approx(exported[nid]["V"])
        assert snap.node_metrics[nid]["L"] == pytest.approx(exported[nid]["L"])
    assert snap.total_routed_requests >= 0
    assert snap.entropy_rate >= 0.0


@pytest.mark.asyncio
async def test_disk_save_load_roundtrip(memory_field, router, tmp_path):
    """save_checkpoint then load_latest_checkpoint must return equal data."""
    await memory_field.deposit_trace(
        "n1", latency_sec=0.3, tokens=5, success=True, active_load=1,
        capability_match=0.6,
    )
    cm = CheckpointManager(
        memory_field=memory_field, storage_path=str(tmp_path), router_agent=router,
    )
    snap = await cm.create_snapshot()
    await cm.save_checkpoint(snap)

    cm2 = CheckpointManager(storage_path=str(tmp_path))
    loaded = await cm2.load_latest_checkpoint()

    assert loaded is not None
    assert loaded.node_metrics == snap.node_metrics
    assert loaded.entropy_rate == pytest.approx(snap.entropy_rate)
    assert os.path.exists(os.path.join(str(tmp_path), "snapshot_latest.json"))


@pytest.mark.asyncio
async def test_redis_save_load_roundtrip(memory_field, router, fake_redis, tmp_path):
    """Redis is preferred over disk on load."""
    await memory_field.deposit_trace(
        "n0", latency_sec=0.7, tokens=12, success=False, active_load=3,
        capability_match=0.4,
    )
    cm = CheckpointManager(
        memory_field=memory_field, redis_client=fake_redis,
        storage_path=str(tmp_path), router_agent=router,
    )
    snap = await cm.create_snapshot()
    await cm.save_checkpoint(snap)

    cm2 = CheckpointManager(redis_client=fake_redis, storage_path=str(tmp_path))
    loaded = await cm2.load_latest_checkpoint()

    assert loaded is not None
    assert loaded.node_metrics["n0"]["V"] == pytest.approx(snap.node_metrics["n0"]["V"])
    assert loaded.node_metrics["n0"]["L"] == pytest.approx(snap.node_metrics["n0"]["L"])


@pytest.mark.asyncio
async def test_load_returns_none_when_no_checkpoint(fake_redis, tmp_path):
    """Empty store must yield None (cold-start trigger)."""
    cm = CheckpointManager(redis_client=fake_redis, storage_path=str(tmp_path))
    assert await cm.load_latest_checkpoint() is None


@pytest.mark.asyncio
async def test_periodic_checkpointing_saves_repeatedly(memory_field, router, tmp_path):
    """The periodic loop must persist snapshots on each tick."""
    cm = CheckpointManager(
        memory_field=memory_field, storage_path=str(tmp_path), router_agent=router,
    )
    task = asyncio.create_task(cm.start_periodic_checkpointing(interval_seconds=0.3))
    try:
        await asyncio.sleep(0.45)  # allow the immediate + at least one tick
    finally:
        await _cancel_task(task)

    # snapshot_latest.json should now exist from the immediate + ticked save.
    assert os.path.exists(os.path.join(str(tmp_path), "snapshot_latest.json"))


# ── Memory field hydration ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_memory_field_hydrate_overwrites_baseline(tmp_path):
    """hydrate_from_snapshot must copy V/L/S/C exactly for known nodes."""
    mf = InMemoryPheromoneMemoryField(node_ids=["n0", "n1", "n2"])
    await mf.deposit_trace(
        "n0", latency_sec=0.5, tokens=10, success=True, active_load=2,
        capability_match=0.8,
    )
    cm = CheckpointManager(memory_field=mf, storage_path=str(tmp_path))
    snap = await cm.create_snapshot()

    mf2 = InMemoryPheromoneMemoryField(node_ids=["n0", "n1", "n2"])
    updated = await mf2.hydrate_from_snapshot(snap)
    after = await mf2.export_state_dict()

    assert updated == 3
    for nid in snap.node_metrics:
        assert after[nid]["V"] == pytest.approx(snap.node_metrics[nid]["V"])
        assert after[nid]["L"] == pytest.approx(snap.node_metrics[nid]["L"])
        assert after[nid]["S"] == pytest.approx(snap.node_metrics[nid]["S"])
        assert after[nid]["C"] == pytest.approx(snap.node_metrics[nid]["C"])


@pytest.mark.asyncio
async def test_hydrate_skips_unknown_nodes(tmp_path):
    """Nodes in the snapshot absent from the field must be ignored."""
    mf = InMemoryPheromoneMemoryField(node_ids=["a"])
    snapshot = MatrixSnapshot(
        timestamp=1.0,
        node_metrics={
            "a": {"V": 0.5, "L": 0.2, "S": 0.1, "C": 0.6},
            "ghost": {"V": 0.9, "L": 0.1, "S": 0.0, "C": 1.0},
        },
    )
    updated = await mf.hydrate_from_snapshot(snapshot)
    assert updated == 1
    state = await mf.export_state_dict()
    assert state["a"]["V"] == pytest.approx(0.5)
    assert "ghost" not in state


@pytest.mark.asyncio
async def test_redis_memory_field_export_and_hydrate(fake_redis):
    """The Redis backend must round-trip traces via export/hydrate."""
    field = RedisPheromoneMemoryField(
        node_ids=["x", "y"], redis_client=fake_redis, decay_rate=0.0,
    )
    await field.deposit_trace("x", 0.4, 8, True, 1, 0.7)
    exported = await field.export_state_dict()
    # EWMA from baseline V=1.0, L=0.1, S=0.0, C=1.0 (alpha=0.5):
    # L -> 0.5*0.4 + 0.5*0.1 = 0.25 ; V stays 1.0 (success)
    assert exported["x"]["L"] == pytest.approx(0.25)

    field2 = RedisPheromoneMemoryField(
        node_ids=["x", "y"], redis_client=fake_redis, decay_rate=0.0,
    )
    await field2.hydrate_from_snapshot(MatrixSnapshot(timestamp=1.0, node_metrics={
        "x": {"V": 0.9, "L": 0.2, "S": 0.1, "C": 0.8},
        "y": {"V": 0.5, "L": 0.5, "S": 0.3, "C": 0.4},
    }))
    state = await field2.get_state_vector()
    idx_x = field2.node_ids.index("x")
    np.testing.assert_allclose(state[idx_x], [0.9, 0.2, 0.1, 0.8], rtol=1e-4)


# ── FastAPI lifespan warmup integration ────────────────────────────────

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from api.server import app

    with TestClient(app) as c:
        yield c


def test_cold_start_when_checkpoints_disabled(client):
    """With checkpointing disabled (default) startup reports cold start."""
    assert _state_attr("checkpoint_manager") is None
    assert _state_attr("checkpoint_task") is None
    assert _gauge_value() == 0


@pytest.mark.asyncio
async def test_warmup_hydrates_when_enabled(tmp_path):
    """_warmup_checkpoint hydrates the field when checkpoints.enabled is set."""
    saved = {
        "memory_field": server_mod._state.memory_field,
        "router": server_mod._state.router,
        "checkpoint_manager": server_mod._state.checkpoint_manager,
        "checkpoint_task": server_mod._state.checkpoint_task,
        "config_checkpoints": server_mod._state.config.get("checkpoints"),
    }
    try:
        mf = InMemoryPheromoneMemoryField(node_ids=MOCK_NODES)
        server_mod._state.memory_field = mf
        server_mod._state.router = StigmergicRouterAgent(
            workers={}, memory_field=mf,
            weights={"alpha": 1.0, "beta": 2.0, "gamma": 1.5, "delta": 1.5},
            temperature=0.5, delta=1.5,
        )
        server_mod._state.config["checkpoints"] = {
            "enabled": True, "storage_path": str(tmp_path), "interval": 999,
        }

        snap = MatrixSnapshot(
            timestamp=99.0, version="1.0",
            node_metrics={
                "mock_node_0": {"V": 0.3, "L": 0.04, "S": 0.1, "C": 0.7},
                "mock_node_1": {"V": 0.5, "L": 0.06, "S": 0.2, "C": 0.5},
            },
            entropy_rate=0.4, total_routed_requests=5,
        )
        cm = CheckpointManager(
            memory_field=mf, storage_path=str(tmp_path),
            router_agent=server_mod._state.router,
        )
        await cm.save_checkpoint(snap)

        await server_mod._warmup_checkpoint()

        assert server_mod._state.checkpoint_manager is not None
        assert _gauge_value() == 1
        state = await mf.export_state_dict()
        assert state["mock_node_0"]["V"] == pytest.approx(0.3)
        assert state["mock_node_1"]["L"] == pytest.approx(0.06)
    finally:
        await _cancel_task(server_mod._state.checkpoint_task)
        server_mod._state.memory_field = saved["memory_field"]
        server_mod._state.router = saved["router"]
        server_mod._state.checkpoint_manager = saved["checkpoint_manager"]
        server_mod._state.checkpoint_task = saved["checkpoint_task"]
        if saved["config_checkpoints"] is None:
            server_mod._state.config.pop("checkpoints", None)
        else:
            server_mod._state.config["checkpoints"] = saved["config_checkpoints"]


def _state_attr(name: str):
    return getattr(server_mod._state, name)


async def _cancel_task(task):
    if task is None:
        return
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


# ── Admin CLI ───────────────────────────────────────────────────────────

def _sample_snapshot() -> MatrixSnapshot:
    return MatrixSnapshot(
        timestamp=1_700_000_000.0,
        version="1.0",
        node_metrics={
            "node-fast": {"V": 1.0, "L": 0.03, "S": 0.0, "C": 0.9},
            "node-slow": {"V": 0.6, "L": 0.42, "S": 0.25, "C": 0.4},
        },
        entropy_rate=0.57,
        total_routed_requests=123,
    )


def test_cli_inspect(tmp_path):
    """inspect reads a snapshot file and prints a human-readable summary."""
    snap_file = tmp_path / "snap.json"
    snap_file.write_text(_sample_snapshot().model_dump_json())
    # Should print without error.
    checkpoint_ctl.main(["inspect", "--input", str(snap_file)])


def test_cli_export_from_disk(tmp_path):
    """export reads the latest disk checkpoint and writes it to --output."""
    cm = CheckpointManager(storage_path=str(tmp_path))
    snap = _sample_snapshot()
    asyncio.run(cm.save_checkpoint(snap))

    out = tmp_path / "exported.json"
    checkpoint_ctl.main(
        ["export", "--output", str(out), "--storage-path", str(tmp_path)]
    )
    data = json.loads(out.read_text())
    assert data["node_metrics"]["node-fast"]["V"] == 1.0
    assert data["entropy_rate"] == pytest.approx(0.57)


def test_cli_import_writes_latest(tmp_path):
    """import reads a JSON snapshot and persists it to the store."""
    snap_file = tmp_path / "incoming.json"
    snap_file.write_text(_sample_snapshot().model_dump_json())

    checkpoint_ctl.main(
        ["import", "--input", str(snap_file), "--storage-path", str(tmp_path)]
    )
    assert (tmp_path / "snapshot_latest.json").exists()

    cm = CheckpointManager(storage_path=str(tmp_path))
    loaded = asyncio.run(cm.load_latest_checkpoint())
    assert loaded is not None
    assert loaded.node_metrics["node-slow"]["L"] == pytest.approx(0.42)


def test_cli_build_parser_selects_command():
    parser = checkpoint_ctl.build_parser()
    args = parser.parse_args(["inspect", "--input", "/tmp/x.json"])
    assert args.command == "inspect"
    assert args.func == checkpoint_ctl.cmd_inspect


def test_cli_export_errors_without_checkpoint(tmp_path):
    """export without any checkpoint should exit non-zero."""
    with pytest.raises(SystemExit):
        checkpoint_ctl.main(
            ["export", "--output", str(tmp_path / "o.json"),
             "--storage-path", str(tmp_path)]
        )
