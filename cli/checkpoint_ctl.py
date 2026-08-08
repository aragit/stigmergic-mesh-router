"""Command-line tool for administrative checkpoint state management.

Usage
-----
    python -m cli.checkpoint_ctl export  --output snapshot.json --redis-url redis://localhost:6379/0
    python -m cli.checkpoint_ctl import  --input snapshot.json --redis-url redis://localhost:6379/0
    python -m cli.checkpoint_ctl inspect  --input snapshot.json
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.checkpointing import CheckpointManager, MatrixSnapshot  # noqa: E402

try:
    import redis.asyncio as redis
except ImportError:  # pragma: no cover
    redis = None


def _build_manager(args: argparse.Namespace) -> CheckpointManager:
    redis_client = None
    if args.redis_url:
        if redis is None:
            raise SystemExit("redis-py is required for --redis-url")
        redis_client = redis.from_url(args.redis_url)
    return CheckpointManager(
        memory_field=None,
        redis_client=redis_client,
        storage_path=args.storage_path,
    )


def _load_file(path: str) -> MatrixSnapshot:
    with open(path, "r", encoding="utf-8") as fh:
        return MatrixSnapshot.model_validate_json(fh.read())


async def cmd_export(args: argparse.Namespace) -> None:
    """Dump the latest checkpoint to a JSON file."""
    manager = _build_manager(args)
    snapshot = await manager.load_latest_checkpoint()
    if snapshot is None:
        raise SystemExit("No checkpoint found in Redis or disk store")

    data = json.dumps(json.loads(snapshot.model_dump_json()), indent=2)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(data)
    print(f"Exported checkpoint (t={snapshot.timestamp:.0f}, "
          f"{len(snapshot.node_metrics)} nodes) -> {args.output}")


async def cmd_import(args: argparse.Namespace) -> None:
    """Load a JSON snapshot and persist it as the latest checkpoint."""
    snapshot = _load_file(args.input)
    manager = _build_manager(args)
    location = await manager.save_checkpoint(snapshot)
    print(f"Imported {len(snapshot.node_metrics)} nodes from {args.input} "
          f"-> {location}")


def cmd_inspect(args: argparse.Namespace) -> None:
    """Print a human-readable summary of a snapshot file."""
    snapshot = _load_file(args.input)
    import time

    now = time.time()
    age = now - snapshot.timestamp
    print(f"Checkpoint snapshot v{snapshot.version}")
    print(f"  timestamp         : {snapshot.timestamp:.0f} ({age:.1f}s ago)")
    print(f"  entropy_rate      : {snapshot.entropy_rate:.4f}")
    print(f"  total_routed_reqs : {snapshot.total_routed_requests}")
    print(f"  nodes             : {len(snapshot.node_metrics)}")
    print("  node metrics (V/L/S/C):")
    for nid, traces in snapshot.node_metrics.items():
        v = traces.get("V", 0.0)
        lat = traces.get("L", 0.0)
        s = traces.get("S", 0.0)
        c = traces.get("C", 0.0)
        print(f"    {nid:<20} V={v:.3f}  L={lat:.3f}  S={s:.3f}  C={c:.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="checkpoint_ctl",
        description="Manage stigmergic pheromone matrix checkpoints.",
    )
    parser.add_argument("--storage-path", default="./data/checkpoints",
                        help="Local checkpoint directory (default: ./data/checkpoints)")

    sub = parser.add_subparsers(dest="command", required=True)

    p_export = sub.add_parser("export", help="Dump the latest checkpoint to a JSON file.")
    p_export.add_argument("--output", required=True, help="Destination JSON file path.")
    p_export.add_argument("--redis-url", default=None,
                          help="Redis URL to read the checkpoint from.")
    p_export.add_argument("--storage-path", default="./data/checkpoints",
                          help="Local checkpoint directory.")
    p_export.set_defaults(func=cmd_export)

    p_import = sub.add_parser("import", help="Load a JSON snapshot into the checkpoint store.")
    p_import.add_argument("--input", required=True, help="Source JSON snapshot file.")
    p_import.add_argument("--redis-url", default=None,
                          help="Redis URL to write the checkpoint to.")
    p_import.add_argument("--storage-path", default="./data/checkpoints",
                          help="Local checkpoint directory.")
    p_import.set_defaults(func=cmd_import)

    p_inspect = sub.add_parser("inspect", help="Print a human-readable summary of a snapshot.")
    p_inspect.add_argument("--input", required=True, help="Snapshot JSON file to inspect.")
    p_inspect.set_defaults(func=cmd_inspect)

    return parser


def main(argv: Optional[list] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    func = args.func
    if asyncio.iscoroutinefunction(func):
        asyncio.run(func(args))
    else:
        func(args)


if __name__ == "__main__":
    main()
