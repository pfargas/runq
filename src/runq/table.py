"""Results as a pandas DataFrame: params + results expanded from JSON to columns.

The only module that touches pandas (``pip install runq[table]``). Fully grid-agnostic:
whatever axes you swept and whatever scalars your target returned become columns you can
filter and group by. Seed-averaging is a one-liner from here::

    df.groupby(["L", "N"])["e_per_n"].agg(["mean", "sem"])

:func:`filter_rows` and :func:`group_mean` are that same vocabulary made available to
``runq table`` for people who would rather not open a notebook.
"""

from __future__ import annotations

import json

from runq.query import OPS as _OPS  # one operator table, shared with the pandas-free filter

# Queue bookkeeping, not physics: never averaged over, and hidden from the default view.
BOOKKEEPING = ("id", "status", "run_dir", "error", "started_at", "finished_at")


def load_table(conn, status: str | None = "done"):
    """All rows (default: ``done`` only) with ``params_json``/``result_json`` expanded.

    A result key that collides with a parameter name (or a bookkeeping column) gets a
    ``result_`` prefix so nothing is silently overwritten.
    """
    import pandas as pd

    if status is None:
        rows = conn.execute("SELECT * FROM runs ORDER BY id").fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM runs WHERE status=? ORDER BY id", (status,)
        ).fetchall()
    if not rows:
        return pd.DataFrame()

    base = pd.DataFrame([dict(r) for r in rows])
    params = pd.json_normalize(base["params_json"].map(json.loads))
    results = pd.json_normalize(
        # pandas renders a NULL result_json (todo/failed rows) as None or NaN
        base["result_json"].map(lambda s: json.loads(s) if isinstance(s, str) else {})
    )
    base = base.drop(columns=["params_json", "result_json"])

    taken = set(base.columns) | set(params.columns)
    results.columns = [c if c not in taken else f"result_{c}" for c in results.columns]
    out = pd.concat([base, params, results], axis=1)
    # Which columns are knobs and which are measurements is not recoverable from the
    # frame alone, and group_mean must never average a parameter axis (a mean over
    # `lr` is nonsense). Record the split; it survives masking, slicing and sorting.
    out.attrs["params"] = list(params.columns)
    out.attrs["results"] = list(results.columns)
    return out


def _coerce(series, raw: str):
    """Cast a CLI string to the column's dtype so ``N=5`` matches the integer 5."""
    import pandas as pd

    if pd.api.types.is_bool_dtype(series):
        return raw.lower() in ("1", "true", "yes")
    if pd.api.types.is_numeric_dtype(series):
        return float(raw)
    return raw


def filter_rows(df, exprs):
    """Keep rows matching every ``NAME<op>VALUE`` expression (``=``/``!=``/``<``/``>``...).

    Values are coerced to the column's dtype, so ``L=0.5`` works on floats and ``label``
    comparisons stay strings. An unknown column name is a hard error — silently returning
    every row for a typo'd axis is how you publish the wrong number.
    """
    for expr in exprs:
        for token, op in _OPS:
            name, sep, raw = expr.partition(token)
            if not sep:
                continue
            name, raw = name.strip(), raw.strip()
            if name not in df.columns:
                raise KeyError(
                    f"no column {name!r} in the table; have: {', '.join(map(str, df.columns))}"
                )
            df = df[op(df[name], _coerce(df[name], raw))]
            break
        else:
            raise ValueError(f"malformed --where {expr!r}; expected NAME=VALUE")
    return df


def group_mean(df, by, cols=None):
    """Average the remaining axes away (normally the seeds): mean + ``_sem`` per result.

    ``by`` are the axes you keep. Only **result** columns are aggregated — the other
    parameter axes are dropped, never averaged, since the mean of an ``lr`` is not a
    thing. ``n`` is how many runs went into each mean: if it is bigger than your seed
    count, you pooled over an axis you forgot to name in ``by``, and it is the number
    to check before believing the error bars.
    """
    import pandas as pd

    by = list(by)
    missing = [c for c in by if c not in df.columns]
    if missing:
        raise KeyError(f"cannot group by {missing}: not in the table")

    if cols is None:
        known = df.attrs.get("results")
        if known is None:
            raise ValueError(
                "group_mean cannot tell results from parameters in this frame; "
                "pass cols=[...] explicitly or start from load_table()"
            )
        cols = [
            c for c in known
            if c in df.columns
            and c not in by
            and pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])
        ]
    cols = list(cols)
    if not cols:
        raise ValueError("no numeric result column left to average")

    g = df.groupby(by, dropna=False)
    out = g[cols].agg(["mean", "sem"])
    # flatten the MultiIndex: e_per_n / e_per_n_sem, not ('e_per_n', 'mean')
    out.columns = [c if stat == "mean" else f"{c}_sem" for c, stat in out.columns]
    out["n"] = g.size()
    return out.reset_index()


def shorten_errors(df, width: int = 60):
    """Collapse an ``error`` column to the exception line — a table is no place for a stack.

    The full traceback stays in the DB; ``runq failed --full`` is how you read it.
    """
    if "error" not in df.columns:
        return df

    def last_line(err):
        if not isinstance(err, str):
            return err
        line = (err.strip().splitlines() or [""])[-1]
        return line if len(line) <= width else line[: width - 1] + "…"

    df = df.copy()
    df["error"] = df["error"].map(last_line)
    return df


def format_table(df, max_rows: int | None = None) -> str:
    """Render for a terminal: no index column, NaNs as ``-``, optionally truncated."""
    if df.empty:
        return "(no rows)"
    shown = df if max_rows is None else df.head(max_rows)
    text = shown.to_string(index=False, na_rep="-")
    if max_rows is not None and len(df) > max_rows:
        text += f"\n... {len(df) - max_rows} more row(s); raise --limit or use --csv"
    return text


def natural_columns(df, keep=()):
    """Default view: the sweep axes and results, with queue bookkeeping dropped.

    ``keep`` un-hides bookkeeping columns that carry information in context — ``status``
    when several statuses are on screen, ``error`` when you asked for the failures.
    """
    hide = [c for c in BOOKKEEPING if c not in keep]
    cols = [c for c in df.columns if c not in hide]
    return df[cols] if cols else df
