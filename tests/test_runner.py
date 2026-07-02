import json
import os

import pytest

from runq import store
from runq.grid import build_grid
from runq.params import ParamSpace, key_json, run_label
from runq.runner import drain, execute_claimed


def _enqueue_grid(conn, space, axes):
    grid = build_grid(space, axes)
    for p in grid:
        store.enqueue(conn, key_json(p), run_label(p, list(axes)))
    return grid


def test_drain_runs_everything(tmp_path, toy_fn, toy_space):
    conn = store.connect(str(tmp_path / "q.db"))
    _enqueue_grid(conn, toy_space, {"a": [1.0, 2.0], "b": [2, 3], "seed": [0, 1]})
    counts = drain(conn, toy_fn, toy_space, str(tmp_path), log=lambda *_: None)
    assert counts == {"done": 8, "failed": 0, "skipped": 0}
    assert store.status_counts(conn) == {"done": 8}
    row = store.fetch(conn, "done")[0]
    assert json.loads(row["result_json"])["energy"] == 1.0 * 2 + 0


def test_run_dir_injected_and_run_json_written(tmp_path, toy_fn, toy_space):
    conn = store.connect(str(tmp_path / "q.db"))
    _enqueue_grid(conn, toy_space, {"tag": ["hello"]})
    drain(conn, toy_fn, toy_space, str(tmp_path), log=lambda *_: None)
    run_dir = os.path.join(str(tmp_path), store.fetch(conn, "done")[0]["run_dir"])
    with open(os.path.join(run_dir, "artifact.txt")) as fh:
        assert fh.read() == "hello"  # the target received run_dir
    with open(os.path.join(run_dir, "run.json")) as fh:
        meta = json.load(fh)
    assert meta["params"]["tag"] == "hello"
    assert meta["result"]["energy"] == 2.0


def test_failure_recorded_and_drain_continues(tmp_path, toy_fn, toy_space):
    conn = store.connect(str(tmp_path / "q.db"))
    _enqueue_grid(conn, toy_space, {"a": [-1.0, 1.0]})  # a=-1 raises ValueError
    counts = drain(conn, toy_fn, toy_space, str(tmp_path), log=lambda *_: None)
    assert counts == {"done": 1, "failed": 1, "skipped": 0}
    failed = store.fetch(conn, "failed")[0]
    assert "negative a" in failed["error"]
    assert "Traceback" in failed["error"]


def test_skip_recorded(tmp_path, toy_fn, toy_space):
    conn = store.connect(str(tmp_path / "q.db"))
    _enqueue_grid(conn, toy_space, {"b": [13, 2]})  # b=13 raises Skip
    counts = drain(conn, toy_fn, toy_space, str(tmp_path), log=lambda *_: None)
    assert counts == {"done": 1, "failed": 0, "skipped": 1}
    assert store.fetch(conn, "skipped")[0]["error"] == "unlucky b"


def test_resume_skips_done(tmp_path, toy_fn, toy_space):
    """Re-enqueueing and re-draining runs nothing that is already done."""
    conn = store.connect(str(tmp_path / "q.db"))
    axes = {"a": [1.0, 2.0]}
    _enqueue_grid(conn, toy_space, axes)
    drain(conn, toy_fn, toy_space, str(tmp_path), log=lambda *_: None)
    first = {r["id"]: r["finished_at"] for r in store.fetch(conn, "done")}

    _enqueue_grid(conn, toy_space, axes)  # same sweep again
    counts = drain(conn, toy_fn, toy_space, str(tmp_path), log=lambda *_: None)
    assert counts == {"done": 0, "failed": 0, "skipped": 0}  # nothing re-ran
    assert {r["id"]: r["finished_at"] for r in store.fetch(conn, "done")} == first


def test_non_dict_result_is_failure(tmp_path):
    def bad(a=1.0, b=2, tag="x", flag=False, seed=0):
        return 42

    space = ParamSpace.from_function(bad)
    conn = store.connect(str(tmp_path / "q.db"))
    store.enqueue(conn, key_json(space.resolve()), "pt")
    row = store.claim_next(conn)
    with pytest.raises(TypeError, match="must return a dict"):
        execute_claimed(conn, row, bad, space, str(tmp_path))
    assert store.status_of(conn, row["params_json"]) == "failed"


def test_unserializable_result_values_are_coerced(tmp_path):
    class FakeNumpyScalar:
        def item(self):
            return 1.25

    def fn(a=1.0, b=2, tag="x", flag=False, seed=0):
        return {"e": FakeNumpyScalar(), "obj": object()}

    space = ParamSpace.from_function(fn)
    conn = store.connect(str(tmp_path / "q.db"))
    store.enqueue(conn, key_json(space.resolve()), "pt")
    row = store.claim_next(conn)
    execute_claimed(conn, row, fn, space, str(tmp_path))
    result = json.loads(store.fetch(conn, "done")[0]["result_json"])
    assert result["e"] == 1.25
    assert isinstance(result["obj"], str)
