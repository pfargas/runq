"""The runq CLI — every command is a thin wrapper over the Python API.

    runq run point.py --axis L=0.5,0.8 --axis N=2,5 --seeds 0 1 2 --gpus 0,1
    runq enqueue point.py --axis lr=1e-3,3e-3        # queue without running
    runq status / runq failed / runq requeue
    runq table --group L,N --sort e_per_n            # seed-averaged results
    runq dirs --where L=0.8 --where N=40             # artifact paths, for the shell
    runq merge merged.db pc1.db pc2.db

``--seeds 0 1 2`` is sugar for ``--axis seed=0,1,2`` (the target must have a ``seed``
parameter). Artifacts land in ``<out-root>/runs/<hash>/`` next to the DB by default;
you find them with ``runq dirs``, not by reading the path.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from runq import merge as merge_mod
from runq import notify, query, store
from runq import table as table_mod
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
    pr.add_argument("--notify", action="store_true",
                    help="email the final queue status when the drain finishes "
                         "(config: ~/.config/runq/notify.toml; verify with runq notify --test)")

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

    pn = sub.add_parser("notify", help="email the queue status now "
                                       "(config: ~/.config/runq/notify.toml)")
    add_common(pn)
    pn.add_argument("--test", action="store_true",
                    help="send a test email to verify the SMTP config")

    pdir = sub.add_parser("dirs", help="artifact directories of the runs matching a filter")
    add_common(pdir)
    pdir.add_argument("--status", default="done", choices=(*store.STATUSES, "all"),
                      help="rows to consider (default: done)")
    pdir.add_argument("--where", action="append", default=[], metavar="NAME=VALUE",
                      help="filter, repeatable: L=0.5, N>=5, seed!=3 (=,!=,<,<=,>,>=)")
    pdir.add_argument("--sort", default=None, metavar="COL", help="sort by this column")
    pdir.add_argument("--desc", action="store_true", help="sort descending")
    pdir.add_argument("--limit", type=int, default=0, help="max paths printed (0 = all)")
    pdir.add_argument("--exists", action="store_true",
                      help="only paths that are actually on disk")
    pdir.add_argument("--label", action="store_true",
                      help="prefix each path with the run's label (TAB-separated)")
    pdir.add_argument("-0", "--null", action="store_true",
                      help="NUL-separate the output, for xargs -0")

    pt = sub.add_parser("table", help="print the results table (params + results)")
    add_common(pt)
    pt.add_argument("--status", default="done", choices=(*store.STATUSES, "all"),
                    help="rows to show (default: done)")
    pt.add_argument("--where", action="append", default=[], metavar="NAME=VALUE",
                    help="filter, repeatable: L=0.5, N>=5, label!=bad (=,!=,<,<=,>,>=)")
    pt.add_argument("--cols", default=None, metavar="A,B,C",
                    help="show only these columns (default: axes + results)")
    pt.add_argument("--group", default=None, metavar="A,B",
                    help="average the other axes away (seeds): mean + _sem + n per group")
    pt.add_argument("--sort", default=None, metavar="COL", help="sort by this column")
    pt.add_argument("--desc", action="store_true", help="sort descending")
    pt.add_argument("--limit", type=int, default=50,
                    help="max rows printed, 0 = no limit (default: 50)")
    pt.add_argument("--csv", default=None, metavar="PATH",
                    help="write the full table here instead of printing it")

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
        started = time.monotonic()
        out_root = args.out_root or (os.path.dirname(args.db) or ".")
        conn = store.connect(args.db)
        requeued = store.requeue(conn)
        if requeued:
            print(f"requeued {requeued} interrupted run(s)")
        # No --axis/--seeds ⇒ pure drain of an already-filled queue (e.g. by a project's
        # enqueue script). Enqueueing the bare default point must be asked for explicitly
        # (runq enqueue TARGET), or it would pollute externally planned sweeps.
        if args.axis or args.seeds is not None:
            n, total = _enqueue_grid(conn, args)
            print(f"enqueued {n} new point(s) of {total}; status={store.status_counts(conn)}")
        else:
            counts = store.status_counts(conn)
            print(f"no axes given — draining the existing queue; status={counts}")
            if not counts.get("todo"):
                print("nothing todo. Fill the queue first (project enqueue script, or "
                      "runq enqueue/run with --axis/--seeds).")
                conn.close()
                return 0

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
        if args.notify:
            # a broken mail setup must never turn a finished sweep into a failure
            try:
                subject = notify.notify_queue(args.db, elapsed_s=time.monotonic() - started)
                print(f"notification email sent: {subject}")
            except Exception as exc:
                print(f"WARNING: could not send notification email: {exc}")
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

    if args.cmd == "notify":
        try:
            if args.test:
                notify.notify_test()
                print("test email sent")
            else:
                print(f"email sent: {notify.notify_queue(args.db)}")
            return 0
        except Exception as exc:
            print(f"could not send email: {exc}")
            return 1

    if args.cmd == "dirs":
        return _dirs(args)

    if args.cmd == "table":
        return _table(args)

    raise SystemExit(f"unknown command {args.cmd!r}")


def _dirs(args) -> int:
    """Filter → artifact paths, one per line, so the shell can take it from there.

    This is what lets the run directories be named by hash alone: nobody has to read a
    path to find a point any more, they ask for it by the physics.
    """
    status = None if args.status == "all" else args.status
    conn = store.connect(args.db)
    try:
        rows = query.load_rows(conn, status=status, db_path=args.db)
    finally:
        conn.close()

    try:
        rows = query.filter_rows(rows, args.where)
    except (KeyError, ValueError) as exc:
        raise SystemExit(exc.args[0] if exc.args else str(exc)) from None

    if args.sort:
        if not any(args.sort in r for r in rows):
            raise SystemExit(f"cannot sort by {args.sort!r}: not in the table")
        # rows missing the key sort last, whichever direction we are going
        rows.sort(key=lambda r: (r.get(args.sort) is None, r.get(args.sort, 0)),
                  reverse=args.desc)

    rows = [r for r in rows if r.get("run_dir")]  # a todo row has no artifacts yet
    if args.exists:
        rows = [r for r in rows if os.path.isdir(r["run_dir"])]
    if args.limit:
        rows = rows[: args.limit]

    if not rows:
        # nothing on stdout: a no-match must not feed a stray path to whatever consumes this
        print("no run directories matched the filter", file=sys.stderr)
        return 1

    end = "\0" if args.null else "\n"
    for r in rows:
        line = f"{r['label']}\t{r['run_dir']}" if args.label else r["run_dir"]
        print(line, end=end)
    return 0


def _table(args) -> int:
    status = None if args.status == "all" else args.status
    conn = store.connect(args.db)
    try:
        df = table_mod.load_table(conn, status=status)
    except ImportError:  # pandas lives behind the [table] extra
        raise SystemExit("runq table needs pandas: pip install 'runq[table]'") from None
    finally:
        conn.close()
    if df.empty:
        print(f"{args.db}: no {args.status} rows")
        return 0

    try:
        df = table_mod.filter_rows(df, args.where)
        if args.group:
            df = table_mod.group_mean(df, args.group.split(","))
        elif not args.cols:
            # status only means something when several are on screen; error only when
            # the failures are what you asked to see
            keep = ("status",) if status is None else ("error",) if status == "failed" else ()
            df = table_mod.natural_columns(df, keep=keep)
        if args.cols:
            wanted = [c.strip() for c in args.cols.split(",")]
            missing = [c for c in wanted if c not in df.columns]
            if missing:
                raise KeyError(f"no column(s) {missing} in the table; "
                               f"have: {', '.join(map(str, df.columns))}")
            df = df[wanted]
        if args.sort:
            if args.sort not in df.columns:
                raise KeyError(f"cannot sort by {args.sort!r}: not in the table")
            df = df.sort_values(args.sort, ascending=not args.desc)
    except (KeyError, ValueError) as exc:
        # KeyError.__str__ reprs its argument; args[0] is the message we wrote
        raise SystemExit(exc.args[0] if exc.args else str(exc)) from None

    if df.empty:
        print("no rows matched the filter")
        return 0

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        df.to_csv(args.csv, index=False)  # the CSV keeps the full traceback
        print(f"wrote {len(df)} row(s) x {len(df.columns)} column(s) to {args.csv}")
        return 0

    print(table_mod.format_table(table_mod.shorten_errors(df), max_rows=args.limit or None))
    print(f"\n{len(df)} row(s)")
    return 0


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
