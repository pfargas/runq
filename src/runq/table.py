"""Results as a pandas DataFrame: params + results expanded from JSON to columns.

The only module that touches pandas (``pip install runq[table]``). Fully grid-agnostic:
whatever axes you swept and whatever scalars your target returned become columns you can
filter and group by. Seed-averaging is a one-liner from here::

    df.groupby(["L", "N"])["e_per_n"].agg(["mean", "sem"])
"""

from __future__ import annotations

import json


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
    return pd.concat([base, params, results], axis=1)
