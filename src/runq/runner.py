"""Executing claimed points: call the target, record the outcome, keep draining.

``drain`` is the whole worker loop: claim -> run -> record, until the queue is empty.
A failure is recorded (status ``failed`` + traceback) and the loop keeps going; raising
:class:`Skip` inside the target records the point as ``skipped`` instead.
"""

from __future__ import annotations

import json
import os
import traceback

from runq import store
from runq.params import ParamSpace, dir_hash

RUNS_SUBDIR = "runs"


class Skip(Exception):
    """Raise inside a target function to record the point as skipped, not failed.

    e.g. ``raise Skip("R >= L/2: box too small")`` for an infeasible point.
    """


def execute_claimed(conn, row, fn, space: ParamSpace, out_root: str) -> str:
    """Run one claimed (status ``running``) row. Returns the final status.

    Creates ``<out_root>/runs/<dir_hash>/`` and injects it as ``run_dir`` if the target
    accepts one; writes ``run.json`` (params + result) there on success. Failures are
    recorded and re-raised so the caller decides whether to keep draining.

    The directory is named by the hash of the full parameter dict, not by the label: a
    label carrying every swept axis grows without bound as a sweep gains dimensions, and
    hits the filesystem's name limit. Nothing parses the path — the DB stores ``run_dir``,
    and ``runq dirs --where ...`` is how you find a point — so the name only has to be
    unique, which is exactly what the hash guarantees. ``label`` stays human-readable for
    the worker log and ``runq failed``; ``run.json`` inside carries the full params.
    """
    params = json.loads(row["params_json"])
    rel = os.path.join(RUNS_SUBDIR, dir_hash(params))
    run_path = os.path.join(out_root, rel)
    os.makedirs(run_path, exist_ok=True)

    kwargs = dict(params)
    if space.accepts_run_dir:
        kwargs["run_dir"] = run_path
    try:
        result = _validate_result(fn(**kwargs))
    except Skip as skip:
        store.mark_skipped(conn, row["id"], str(skip) or "skipped")
        return "skipped"
    except Exception:
        store.mark_failed(conn, row["id"], traceback.format_exc())
        raise

    with open(os.path.join(run_path, "run.json"), "w") as fh:
        json.dump({"label": row["label"], "params": params, "result": result}, fh, indent=2)
    store.save_result(conn, row["id"], json.dumps(result), run_dir=rel)
    return "done"


def drain(conn, fn, space: ParamSpace, out_root: str, log=print) -> dict:
    """Claim-and-run until the queue is empty. Returns ``{status: count}`` for this worker."""
    pid = os.getpid()
    counts = {"done": 0, "failed": 0, "skipped": 0}
    while True:
        row = store.claim_next(conn)
        if row is None:
            break
        log(f"[worker {pid}] run     {row['label']}")
        try:
            status = execute_claimed(conn, row, fn, space, out_root)
            counts[status] += 1
            log(f"[worker {pid}] {status:7s} {row['label']}")
        except Exception as exc:  # recorded as failed inside execute_claimed
            counts["failed"] += 1
            log(f"[worker {pid}] FAILED  {row['label']}: {exc!r}")
    log(f"[worker {pid}] queue empty; {counts}")
    return counts


def _validate_result(result) -> dict:
    """Normalise a target's return value into a JSON-safe dict."""
    if result is None:
        return {}
    if not isinstance(result, dict):
        raise TypeError(
            f"target must return a dict of scalars (or None), got {type(result).__name__}"
        )
    out = {}
    for k, v in result.items():
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            # numpy/jax scalars expose .item(); anything else is stored as str
            out[k] = v.item() if hasattr(v, "item") else str(v)
    return out
