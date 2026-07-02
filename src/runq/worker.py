"""One worker process: drain the shared queue until it is empty.

Spawned by the local backend (one per GPU, ``CUDA_VISIBLE_DEVICES`` pinned before this
process starts, so heavy imports inside the target see the right device). Can also be
run by hand for a single-device drain:

    CUDA_VISIBLE_DEVICES=0 python -m runq.worker --db outputs/runq.db --target point.py
"""

from __future__ import annotations

import argparse
import os
import sys

from runq import store
from runq.params import ParamSpace
from runq.runner import drain
from runq.target import load_target


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=store.DEFAULT_DB)
    ap.add_argument("--target", required=True, help="file.py[:func] or module[:func]")
    ap.add_argument("--out-root", default=None,
                    help="artifact root (default: the DB's directory)")
    args = ap.parse_args(argv)
    out_root = args.out_root or (os.path.dirname(args.db) or ".")

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    print(f"[worker {os.getpid()}] CUDA_VISIBLE_DEVICES={cvd}  db={args.db}", flush=True)

    conn = store.connect(args.db)
    fn = load_target(args.target)
    space = ParamSpace.from_function(fn)
    counts = drain(conn, fn, space, out_root)
    conn.close()
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
