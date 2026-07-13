"""The pandas-free row filter behind ``runq dirs``."""

import json

import pytest

from runq import store
from runq.query import filter_rows, load_rows


def _seed(conn):
    rows = [
        ({"L": 0.8, "N": 40, "seed": 0, "tag": "good"}, "a", {"e": 1.0}),
        ({"L": 0.8, "N": 40, "seed": 1, "tag": "bad"}, "b", {"e": 2.0}),
        ({"L": 1.2, "N": 40, "seed": 0, "tag": "good"}, "c", {"e": 3.0}),
    ]
    for params, label, result in rows:
        store.enqueue(conn, json.dumps(params), label)
    for row in store.fetch(conn, "todo"):
        params = json.loads(row["params_json"])
        result = next(r for p, _, r in rows if p == params)
        store.claim_next(conn)
        store.save_result(conn, row["id"], json.dumps(result),
                          run_dir=f"runs/{row['label']}")
    # one point that never ran: it has no run_dir and no result
    store.enqueue(conn, json.dumps({"L": 9.9, "N": 1, "seed": 0, "tag": "x"}), "todo1")


def test_load_rows_flattens_params_and_results(tmp_path):
    conn = store.connect(str(tmp_path / "q.db"))
    _seed(conn)
    rows = load_rows(conn, status="done")
    assert len(rows) == 3
    assert {"L", "N", "seed", "tag", "e", "label", "run_dir"} <= set(rows[0])


def test_load_rows_resolves_run_dir_against_the_db(tmp_path):
    db = str(tmp_path / "outputs" / "q.db")
    conn = store.connect(db)
    _seed(conn)
    rows = load_rows(conn, status="done", db_path=db)
    for r in rows:
        # relative in the DB (a sweep must stay movable), absolute for the caller
        assert r["run_dir"].startswith(str(tmp_path / "outputs" / "runs"))


def test_filter_numeric_and_string(tmp_path):
    conn = store.connect(str(tmp_path / "q.db"))
    _seed(conn)
    rows = load_rows(conn, status="done")
    assert len(filter_rows(rows, ["L=0.8"])) == 2
    assert len(filter_rows(rows, ["L=0.8", "seed=1"])) == 1
    assert len(filter_rows(rows, ["tag=good"])) == 2      # strings compare as strings
    assert len(filter_rows(rows, ["tag!=good"])) == 1
    assert len(filter_rows(rows, ["e>=2"])) == 2          # result columns filter too
    assert len(filter_rows(rows, ["N>100"])) == 0


def test_filter_unknown_column_is_an_error(tmp_path):
    conn = store.connect(str(tmp_path / "q.db"))
    _seed(conn)
    rows = load_rows(conn, status="done")
    # silently returning every row for a typo'd axis is how you publish the wrong number
    with pytest.raises(KeyError):
        filter_rows(rows, ["Lambda=0.8"])
    with pytest.raises(ValueError):
        filter_rows(rows, ["L 0.8"])


def test_rows_without_the_column_never_match(tmp_path):
    conn = store.connect(str(tmp_path / "q.db"))
    _seed(conn)
    rows = load_rows(conn, status=None)  # None = every status, as in table.load_table
    assert len(rows) == 4
    # the todo row has no result: it must not match a filter on one
    assert all(r["status"] == "done" for r in filter_rows(rows, ["e>0"]))
