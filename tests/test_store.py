import json
import threading

from runq import store


def _db(tmp_path):
    return str(tmp_path / "q.db")


def _enqueue_n(conn, n):
    for i in range(n):
        store.enqueue(conn, json.dumps({"i": i}), f"i{i}")


def test_enqueue_idempotent(tmp_path):
    conn = store.connect(_db(tmp_path))
    assert store.enqueue(conn, '{"a":1}', "a1") is True
    assert store.enqueue(conn, '{"a":1}', "a1") is False
    assert store.status_counts(conn) == {"todo": 1}


def test_enqueue_does_not_reset_done(tmp_path):
    conn = store.connect(_db(tmp_path))
    store.enqueue(conn, '{"a":1}', "a1")
    row = store.claim_next(conn)
    store.save_result(conn, row["id"], '{"e":2}', "runs/a1")
    store.enqueue(conn, '{"a":1}', "a1")  # re-running a sweep must not requeue done work
    assert store.status_of(conn, '{"a":1}') == "done"


def test_claim_marks_running_in_order(tmp_path):
    conn = store.connect(_db(tmp_path))
    _enqueue_n(conn, 3)
    first = store.claim_next(conn)
    assert first["label"] == "i0"
    assert store.status_of(conn, first["params_json"]) == "running"
    assert store.claim_next(conn)["label"] == "i1"


def test_claim_empty_returns_none(tmp_path):
    conn = store.connect(_db(tmp_path))
    assert store.claim_next(conn) is None


def test_lifecycle_done_failed_skipped(tmp_path):
    conn = store.connect(_db(tmp_path))
    _enqueue_n(conn, 3)
    r1, r2, r3 = (store.claim_next(conn) for _ in range(3))
    store.save_result(conn, r1["id"], '{"e": 1.5}', "runs/i0")
    store.mark_failed(conn, r2["id"], "Traceback ...")
    store.mark_skipped(conn, r3["id"], "box too small")
    assert store.status_counts(conn) == {"done": 1, "failed": 1, "skipped": 1}
    done = store.fetch(conn, "done")[0]
    assert json.loads(done["result_json"]) == {"e": 1.5}
    assert done["run_dir"] == "runs/i0"
    assert done["finished_at"] is not None
    assert store.fetch(conn, "failed")[0]["error"].startswith("Traceback")


def test_requeue_interrupted_and_failed(tmp_path):
    conn = store.connect(_db(tmp_path))
    _enqueue_n(conn, 3)
    r1 = store.claim_next(conn)  # left running (a crash)
    r2 = store.claim_next(conn)
    store.mark_failed(conn, r2["id"], "boom")
    assert store.requeue(conn) == 1  # only the running one
    assert store.status_counts(conn) == {"todo": 2, "failed": 1}
    assert store.requeue(conn, ("failed",)) == 1
    assert store.status_counts(conn) == {"todo": 3}
    row = store.fetch(conn)[int(r1["id"]) - 1]
    assert row["error"] is None  # requeue clears the stale error


def test_concurrent_claims_are_distinct(tmp_path):
    """Many threads with their own connections never claim the same row."""
    path = _db(tmp_path)
    conn = store.connect(path)
    _enqueue_n(conn, 40)
    claimed: list[int] = []
    lock = threading.Lock()

    def work():
        c = store.connect(path)
        while True:
            row = store.claim_next(c)
            if row is None:
                break
            with lock:
                claimed.append(row["id"])
        c.close()

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(claimed) == list(range(1, 41))  # every row exactly once
    assert store.status_counts(conn) == {"running": 40}
