import json

from runq import store
from runq.table import load_table


def _seed_db(tmp_path):
    conn = store.connect(str(tmp_path / "q.db"))
    for i, (L, seed) in enumerate([(0.5, 0), (0.5, 1), (0.8, 0)]):
        params = {"L": L, "N": 5, "seed": seed}
        store.enqueue(conn, json.dumps(params, sort_keys=True), f"pt{i}")
        row = store.claim_next(conn)
        store.save_result(conn, row["id"], json.dumps({"e_per_n": L * 2 + seed}), f"runs/pt{i}")
    store.enqueue(conn, '{"L": 9.0, "N": 5, "seed": 0}', "leftover")  # stays todo
    return conn


def test_load_table_expands_params_and_results(tmp_path):
    df = load_table(_seed_db(tmp_path))
    assert len(df) == 3  # done only by default
    assert {"L", "N", "seed", "e_per_n", "label", "status", "run_dir"} <= set(df.columns)
    assert sorted(df["L"].unique()) == [0.5, 0.8]
    # the DataFrame is ready for seed-averaging
    means = df.groupby("L")["e_per_n"].mean()
    assert means[0.5] == (1.0 + 2.0) / 2


def test_load_table_all_statuses(tmp_path):
    df = load_table(_seed_db(tmp_path), status=None)
    assert len(df) == 4
    assert set(df["status"]) == {"done", "todo"}


def test_load_table_empty(tmp_path):
    conn = store.connect(str(tmp_path / "empty.db"))
    assert load_table(conn).empty


def test_result_name_collision_gets_prefix(tmp_path):
    conn = store.connect(str(tmp_path / "q.db"))
    store.enqueue(conn, '{"L": 1.0}', "pt")
    row = store.claim_next(conn)
    store.save_result(conn, row["id"], json.dumps({"L": 99.0, "e": 1.0}), "runs/pt")
    df = load_table(conn)
    assert df["L"].iloc[0] == 1.0  # the parameter
    assert df["result_L"].iloc[0] == 99.0  # the colliding result key, prefixed
