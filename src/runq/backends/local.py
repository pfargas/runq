"""Local backend: one worker subprocess per GPU, all draining one shared queue.

Each worker is pinned to a single device via ``CUDA_VISIBLE_DEVICES`` (set before the
process starts, so JAX/torch inside the target see exactly one GPU), and capped to a
slice of the host CPUs — N unthrottled numerical processes on one node thrash each
other on the shared cores (measured ~28x slowdown with 2 unpinned JAX workers).

With more than one worker, stdout goes to per-worker log files (progress bars from
several processes clobber a shared terminal); ``tail -f`` one to watch it.
"""

from __future__ import annotations

import os
import subprocess
import sys


def detect_gpus() -> list[str]:
    """GPU indices from ``nvidia-smi -L``; [] if none / no nvidia-smi."""
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        return [str(i) for i, line in enumerate(out.splitlines()) if line.strip()]
    except Exception:
        return []


def worker_env(gpu: str, cpu_threads: int, base: dict | None = None) -> dict:
    """Environment for one pinned worker: device + CPU-thread caps + headless MPL."""
    env = dict(base if base is not None else os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        MPLBACKEND="Agg",
        OMP_NUM_THREADS=str(cpu_threads),
        MKL_NUM_THREADS=str(cpu_threads),
        OPENBLAS_NUM_THREADS=str(cpu_threads),
        NUMEXPR_NUM_THREADS=str(cpu_threads),
    )
    return env


def cpu_threads_per_worker(n_workers: int, requested: int = 0) -> int:
    """Cores per worker: the requested value, or an even split of the host (>=1)."""
    if requested > 0:
        return requested
    return max(1, (os.cpu_count() or 8) // max(1, n_workers))


def run_local(
    db_path: str,
    target: str,
    out_root: str,
    gpus: list[str] | None = None,
    cpu_per_worker: int = 0,
    log_dir: str | None = None,
) -> int:
    """Spawn one worker per GPU (or one CPU worker if none), wait, return exit code."""
    gpus = list(gpus) if gpus else detect_gpus()
    if not gpus:
        print("no GPU detected — running a single CPU worker.")
        gpus = [""]

    per = cpu_threads_per_worker(len(gpus), cpu_per_worker)
    redirect = len(gpus) > 1
    log_dir = log_dir or os.path.join(out_root, "logs")

    cmd = [sys.executable, "-u", "-m", "runq.worker",
           "--db", db_path, "--target", target, "--out-root", out_root]

    procs, logs = [], []
    for g in gpus:
        env = worker_env(g, per)
        if redirect:
            os.makedirs(log_dir, exist_ok=True)
            path = os.path.join(log_dir, f"worker_gpu{g or 'cpu'}.log")
            fh = open(path, "w")
            logs.append(fh)
            print(f"launch worker on GPU {g or '(cpu)'} -> {path}   (tail -f to watch)")
            procs.append(subprocess.Popen(cmd, env=env, stdout=fh,
                                          stderr=subprocess.STDOUT))
        else:
            print(f"launch worker on GPU {g or '(cpu)'} ({per} CPU threads)")
            procs.append(subprocess.Popen(cmd, env=env))

    rc = 0
    for p in procs:
        rc |= p.wait()
    for fh in logs:
        fh.close()
    return rc
