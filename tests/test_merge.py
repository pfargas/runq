import json

from runq import store
from runq.merge import merge_paths


def _mkdb(path, rows):
    """rows: list of (params, label, status, result_json)."""
    conn = store.connect(str(path))
    for params, label, status, result in rows:
        store.enqueue(conn, json.dumps(params), label)
        if status != "todo":
            row = conn.execute(
                "SELECT id FROM runs WHERE params_json=?", (json.dumps(params),)
            ).fetchone()
            conn.execute(
                "UPDATE runs SET status=?, result_json=? WHERE id=?",
                (status, result, row["id"]),
            )
    conn.close()
    return str(path)


def test_merge_unions_disjoint_sets(tmp_path):
    a = _mkdb(tmp_path / "a.db", [({"i": 0}, "i0", "done", '{"e":1}')])
    b = _mkdb(tmp_path / "b.db", [({"i": 1}, "i1", "done", '{"e":2}')])
    counts = merge_paths(str(tmp_path / "m.db"), [a, b])
    assert counts == {"done": 2}


def test_done_beats_todo_and_running_regardless_of_order(tmp_path):
    key = {"i": 0}
    done = _mkdb(tmp_path / "done.db", [(key, "i0", "done", '{"e":1}')])
    todo = _mkdb(tmp_path / "todo.db", [(key, "i0", "todo", None)])
    running = _mkdb(tmp_path / "run.db", [(key, "i0", "running", None)])

    for order in ([done, todo, running], [todo, running, done]):
        dest = tmp_path / f"m_{order[0].split('/')[-1]}.db"
        counts = merge_paths(str(dest), order)
        assert counts == {"done": 1}
        conn = store.connect(str(dest))
        assert json.loads(store.fetch(conn, "done")[0]["result_json"]) == {"e": 1}
        conn.close()


def test_skipped_beats_failed(tmp_path):
    key = {"i": 0}
    failed = _mkdb(tmp_path / "f.db", [(key, "i0", "failed", None)])
    skipped = _mkdb(tmp_path / "s.db", [(key, "i0", "skipped", None)])
    counts = merge_paths(str(tmp_path / "m.db"), [failed, skipped])
    assert counts == {"skipped": 1}


def test_merge_into_existing_dest(tmp_path):
    dest = _mkdb(tmp_path / "dest.db", [({"i": 0}, "i0", "todo", None)])
    src = _mkdb(tmp_path / "src.db", [({"i": 0}, "i0", "done", '{"e":9}')])
    counts = merge_paths(dest, [src])
    assert counts == {"done": 1}
