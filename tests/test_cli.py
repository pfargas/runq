"""End-to-end CLI tests: serial in-process, and real multi-worker subprocesses."""

import json
import os

import pytest

from runq import store
from runq.backends import local
from runq.cli import main


def _db(tmp_path):
    return str(tmp_path / "outputs" / "q.db")


def test_enqueue_then_status(tmp_path, toy_path, capsys):
    db = _db(tmp_path)
    assert main(["enqueue", toy_path, "--axis", "a=1,2", "--seeds", "0", "1",
                 "--db", db]) == 0
    assert "enqueued 4 new point(s) of 4" in capsys.readouterr().out
    # idempotent: same grid again enqueues nothing new
    main(["enqueue", toy_path, "--axis", "a=1,2", "--seeds", "0", "1", "--db", db])
    assert "enqueued 0 new point(s) of 4" in capsys.readouterr().out


def test_run_serial_end_to_end(tmp_path, toy_path):
    db = _db(tmp_path)
    rc = main(["run", toy_path, "--axis", "a=1,2", "--axis", "b=2,3",
               "--seeds", "0", "--db", db, "--serial"])
    assert rc == 0
    conn = store.connect(db)
    assert store.status_counts(conn) == {"done": 4}
    for row in store.fetch(conn, "done"):
        run_dir = os.path.join(os.path.dirname(db), row["run_dir"])
        assert os.path.isfile(os.path.join(run_dir, "run.json"))
    conn.close()


def test_run_serial_reports_failures_in_exit_code(tmp_path, toy_path):
    db = _db(tmp_path)
    rc = main(["run", toy_path, "--axis", "a=-1", "--db", db, "--serial"])
    assert rc == 1
    conn = store.connect(db)
    assert store.status_counts(conn) == {"failed": 1}
    conn.close()


def test_failed_and_requeue_commands(tmp_path, toy_path, capsys):
    db = _db(tmp_path)
    main(["run", toy_path, "--axis", "a=-1,1", "--db", db, "--serial"])
    capsys.readouterr()

    main(["failed", "--db", db])
    out = capsys.readouterr().out
    assert "negative a" in out and "1 failed point(s)" in out

    main(["requeue", "--failed", "--db", db])
    conn = store.connect(db)
    counts = store.status_counts(conn)
    assert counts == {"todo": 1, "done": 1}
    conn.close()


def test_seeds_and_axis_seed_conflict(tmp_path, toy_path):
    with pytest.raises(SystemExit, match="not both"):
        main(["enqueue", toy_path, "--axis", "seed=0,1", "--seeds", "2",
              "--db", _db(tmp_path)])


def test_run_multiworker_subprocesses(tmp_path, toy_path):
    """Two real worker subprocesses drain one queue: every point done exactly once."""
    db = _db(tmp_path)
    rc = main(["run", toy_path, "--axis", "a=1,2,3,4", "--axis", "b=2,3",
               "--seeds", "0", "--db", db, "--gpus", "0,1"])
    assert rc == 0
    conn = store.connect(db)
    assert store.status_counts(conn) == {"done": 8}
    energies = sorted(
        json.loads(r["result_json"])["energy"] for r in store.fetch(conn, "done")
    )
    assert energies == sorted(a * b for a in (1, 2, 3, 4) for b in (2, 3))
    conn.close()
    # >1 worker ⇒ per-worker logs, not a shared terminal
    log_dir = os.path.join(os.path.dirname(db), "logs")
    assert sorted(os.listdir(log_dir)) == ["worker_gpu0.log", "worker_gpu1.log"]


def test_merge_command(tmp_path, toy_path, capsys):
    db1, db2 = str(tmp_path / "a.db"), str(tmp_path / "b.db")
    main(["run", toy_path, "--axis", "a=1", "--db", db1, "--serial"])
    main(["run", toy_path, "--axis", "a=2", "--db", db2, "--serial"])
    capsys.readouterr()
    merged = str(tmp_path / "m.db")
    assert main(["merge", merged, db1, db2]) == 0
    conn = store.connect(merged)
    assert store.status_counts(conn) == {"done": 2}
    conn.close()


def test_worker_env_pins_device_and_threads():
    env = local.worker_env("1", 4, base={"PATH": "/bin"})
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["OMP_NUM_THREADS"] == "4"
    assert env["MPLBACKEND"] == "Agg"
    assert env["PATH"] == "/bin"


def test_cpu_threads_split():
    assert local.cpu_threads_per_worker(2, requested=3) == 3
    n = os.cpu_count() or 8
    assert local.cpu_threads_per_worker(2) == max(1, n // 2)
    assert local.cpu_threads_per_worker(10_000) == 1
