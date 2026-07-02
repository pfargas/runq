"""The runq CLI — every command is a thin wrapper over the Python API.

    runq run point.py --axis L=0.5,0.8 --axis N=2,5 --seeds 0 1 2 --gpus 0,1
    runq enqueue point.py --axis lr=1e-3,3e-3        # queue without running
    runq status / runq failed / runq requeue
    runq merge merged.db pc1.db pc2.db

``--seeds 0 1 2`` is sugar for ``--axis seed=0,1,2`` (the target must have a ``seed``
parameter). Artifacts land in ``<out-root>/runs/<label>/`` next to the DB by default.
"""

from __future__ import annotations

import argparse
import os
import sys

from runq import merge as merge_mod
from runq import store
from runq.backends import local
from runq.grid import build_grid, parse_axes
from runq.params import ParamSpace, key_json, run_label
from runq.runner import drain
from runq.target import load_target


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="runq", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--db", default=store.DEFAULT_DB)

    def add_grid(p):
        p.add_argument("target", help="file.py[:func] or module[:func] (default func: run_point)")
        p.add_argument("--axis", action="append", default=[], metavar="NAME=v1,v2,...",
                       help="sweep axis; repeatable; values coerced to the default's type")
        p.add_argument("--seeds", type=int, nargs="+", default=None,
                       help="sugar for --axis seed=...")

    pe = sub.add_parser("enqueue", help="build the grid and add todo rows (no running)")
    add_grid(pe)
    add_common(pe)

    pr = sub.add_parser("run", help="enqueue the grid, then drain it with local workers")
    add_grid(pr)
    add_common(pr)
    pr.add_argument("--out-root", default=None, help="artifact root (default: DB's dir)")
    pr.add_argument("--gpus", default=None, help="comma list, e.g. 0,1 (default: detect)")
    pr.add_argument("--serial", action="store_true",
                    help="drain in this process (no subprocess workers); for debugging")
    pr.add_argument("--cpu-per-worker", type=int, default=0,
                    help="CPU threads per worker (0 = split host cores evenly)")
    pr.add_argument("--log-dir", default=None,
                    help="per-worker logs when >1 worker (default: <out-root>/logs)")

    ps = sub.add_parser("status", help="queue counts")
    add_common(ps)

    pf = sub.add_parser("failed", help="list failed points with their errors")
    add_common(pf)
    pf.add_argument("--full", action="store_true", help="full tracebacks")

    pq = sub.add_parser("requeue", help="reset interrupted (and optionally failed) rows")
    add_common(pq)
    pq.add_argument("--failed", action="store_true", help="also requeue failed rows")

    pm = sub.add_parser("merge", help="merge source DBs into dest (done-precedence)")
    pm.add_argument("dest")
    pm.add_argument("sources", nargs="+")

    args = ap.parse_args(argv)
    return _dispatch(args)


def _dispatch(args) -> int:
    if args.cmd == "enqueue":
        conn = store.connect(args.db)
        n, total = _enqueue_grid(conn, args)
        print(f"enqueued {n} new point(s) of {total}; status={store.status_counts(conn)}")
        conn.close()
        return 0

    if args.cmd == "run":
        out_root = args.out_root or (os.path.dirname(args.db) or ".")
        conn = store.connect(args.db)
        requeued = store.requeue(conn)
        if requeued:
            print(f"requeued {requeued} interrupted run(s)")
        n, total = _enqueue_grid(conn, args)
        print(f"enqueued {n} new point(s) of {total}; status={store.status_counts(conn)}")

        if args.serial:
            fn = load_target(args.target)
            drain(conn, fn, ParamSpace.from_function(fn), out_root)
        else:
            conn.close()  # workers open their own connections
            local.run_local(args.db, args.target, out_root,
                            gpus=args.gpus.split(",") if args.gpus else None,
                            cpu_per_worker=args.cpu_per_worker, log_dir=args.log_dir)
            conn = store.connect(args.db)
        counts = store.status_counts(conn)
        print(f"final status: {counts}")
        conn.close()
        return 1 if counts.get("failed") else 0

    if args.cmd == "status":
        conn = store.connect(args.db)
        counts = store.status_counts(conn)
        print(f"{args.db}: {counts or 'empty'}  (total {sum(counts.values())})")
        conn.close()
        return 0

    if args.cmd == "failed":
        conn = store.connect(args.db)
        rows = store.fetch(conn, "failed")
        for r in rows:
            err = r["error"] or ""
            shown = err if args.full else (err.strip().splitlines() or [""])[-1]
            print(f"{r['label']}\n    {shown}")
        print(f"{len(rows)} failed point(s)")
        conn.close()
        return 0

    if args.cmd == "requeue":
        conn = store.connect(args.db)
        statuses = ("running", "failed") if args.failed else ("running",)
        n = store.requeue(conn, statuses)
        print(f"requeued {n} row(s) from {statuses}; status={store.status_counts(conn)}")
        conn.close()
        return 0

    if args.cmd == "merge":
        counts = merge_mod.merge_paths(args.dest, args.sources)
        print(f"{args.dest}: {counts}")
        return 0

    raise SystemExit(f"unknown command {args.cmd!r}")


def _enqueue_grid(conn, args) -> tuple[int, int]:
    """Enqueue the grid from --axis/--seeds. Returns (newly enqueued, grid size)."""
    fn = load_target(args.target)
    space = ParamSpace.from_function(fn)
    axes = parse_axes(args.axis, space)
    if args.seeds is not None:
        if "seed" in axes:
            raise SystemExit("give either --seeds or --axis seed=..., not both")
        axes["seed"] = [space.coerce("seed", str(s)) for s in args.seeds]
    swept = list(axes)
    n = 0
    grid = build_grid(space, axes)
    for params in grid:
        n += store.enqueue(conn, key_json(params), run_label(params, swept))
    return n, len(grid)


if __name__ == "__main__":
    sys.exit(main())
