import json

import pytest

from runq import store
from runq.table import (
    filter_rows,
    format_table,
    group_mean,
    load_table,
    natural_columns,
)


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


def test_filter_rows_coerces_to_column_dtype(tmp_path):
    df = load_table(_seed_db(tmp_path))
    assert len(filter_rows(df, ["L=0.5"])) == 2  # string "0.5" matched against floats
    assert len(filter_rows(df, ["L=0.5", "seed=1"])) == 1  # filters are ANDed
    assert len(filter_rows(df, ["seed>=1"])) == 1
    assert len(filter_rows(df, ["L!=0.5"])) == 1
    assert len(filter_rows(df, ["label=pt2"])) == 1  # strings compare as strings


def test_filter_rows_rejects_unknown_column_and_junk(tmp_path):
    df = load_table(_seed_db(tmp_path))
    with pytest.raises(KeyError, match="nope"):
        filter_rows(df, ["nope=1"])
    with pytest.raises(ValueError, match="malformed"):
        filter_rows(df, ["L"])


def test_group_mean_averages_seeds_away(tmp_path):
    df = load_table(_seed_db(tmp_path))
    g = group_mean(df, ["L"])
    assert list(g["L"]) == [0.5, 0.8]
    assert g.loc[g["L"] == 0.5, "e_per_n"].iloc[0] == 1.5  # mean of 1.0 and 2.0
    assert list(g["n"]) == [2, 1]
    assert "e_per_n_sem" in g.columns
    assert "id" not in g.columns  # bookkeeping is never averaged
    # only results are aggregated: N is a parameter, so there is no meaningless N_sem
    assert "seed" not in g.columns and "N" not in g.columns and "N_sem" not in g.columns


def test_group_mean_survives_a_filter(tmp_path):
    """The params/results split must survive filter_rows, or grouping breaks downstream."""
    df = filter_rows(load_table(_seed_db(tmp_path)), ["L=0.5"])
    g = group_mean(df, ["L"])
    assert list(g["n"]) == [2]


def test_group_mean_refuses_a_frame_it_cannot_read(tmp_path):
    import pandas as pd

    plain = pd.DataFrame({"L": [0.5, 0.5], "e_per_n": [1.0, 2.0]})
    with pytest.raises(ValueError, match="cannot tell results from parameters"):
        group_mean(plain, ["L"])
    # ...but an explicit column list is always honoured
    assert group_mean(plain, ["L"], cols=["e_per_n"])["e_per_n"].iloc[0] == 1.5


def test_group_mean_rejects_unknown_key(tmp_path):
    with pytest.raises(KeyError, match="nope"):
        group_mean(load_table(_seed_db(tmp_path)), ["nope"])


def test_natural_columns_hides_bookkeeping(tmp_path):
    full = load_table(_seed_db(tmp_path))
    df = natural_columns(full)
    assert "id" not in df.columns and "started_at" not in df.columns
    assert {"L", "N", "seed", "e_per_n", "label"} <= set(df.columns)
    assert "status" in natural_columns(full, keep=("status",))


def test_format_table_truncates(tmp_path):
    df = load_table(_seed_db(tmp_path))
    assert "2 more row(s)" in format_table(df, max_rows=1)
    assert "more row(s)" not in format_table(df)
    assert format_table(df.head(0)) == "(no rows)"
