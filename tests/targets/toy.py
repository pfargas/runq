"""Toy target for tests: fast, deterministic, with failure and skip triggers."""

import os

from runq import Skip


def run_point(a=1.0, b=2, tag="x", flag=False, seed=0, run_dir=None):
    if a < 0:
        raise ValueError("negative a")
    if b == 13:
        raise Skip("unlucky b")
    if run_dir is not None:
        with open(os.path.join(run_dir, "artifact.txt"), "w") as fh:
            fh.write(tag)
    return {"energy": a * b + seed, "flag_used": flag}
