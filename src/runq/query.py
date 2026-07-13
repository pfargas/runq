"""Row selection without pandas — the ``--where`` vocabulary on plain dicts.

:mod:`runq.table` speaks the same language but only exists behind the ``[table]`` extra,
and ``runq dirs`` must keep working on a bare install (its whole job is to hand paths to
other tools, on machines where nobody wants a pandas). The operator table lives here so
the two filters cannot drift apart.
"""

from __future__ import annotations

import json
import operator
import os

# Two-character operators must be tried first, or "L>=1" would split on ">".
OPS = (
    ("!=", operator.ne),
    (">=", operator.ge),
    ("<=", operator.le),
    ("=", operator.eq),
    (">", operator.gt),
    ("<", operator.lt),
)


def load_rows(conn, status: str | None = "done", db_path: str | None = None) -> list[dict]:
    """Rows with ``params_json``/``result_json`` flattened into the dict.

    ``run_dir`` is stored relative to the DB's directory (so a sweep stays movable); when
    ``db_path`` is given it is resolved to a real path the caller can hand to ``open``.
    """
    if status is None:
        cur = conn.execute("SELECT * FROM runs ORDER BY id")
    else:
        cur = conn.execute("SELECT * FROM runs WHERE status=? ORDER BY id", (status,))

    root = os.path.dirname(os.path.abspath(db_path)) if db_path else None
    rows = []
    for r in cur:
        d = dict(r)
        params = json.loads(d.pop("params_json", None) or "{}")
        result = json.loads(d.pop("result_json", None) or "{}")
        # a result key must never silently overwrite a param or a bookkeeping column
        taken = set(d) | set(params)
        d.update(params)
        d.update({(k if k not in taken else f"result_{k}"): v for k, v in result.items()})
        if root and d.get("run_dir"):
            d["run_dir"] = os.path.join(root, d["run_dir"])
        rows.append(d)
    return rows


def _coerce(value, raw: str):
    """Cast the CLI string to the type of the value it is being compared against."""
    if isinstance(value, bool):
        return raw.lower() in ("1", "true", "yes")
    if isinstance(value, (int, float)):
        try:
            return float(raw)
        except ValueError:
            return raw
    return raw


def filter_rows(rows: list[dict], exprs) -> list[dict]:
    """Keep rows matching every ``NAME<op>VALUE`` expression.

    An unknown column is a hard error: silently returning every row for a typo'd axis is
    how you publish the wrong number.
    """
    known = set().union(*(set(r) for r in rows)) if rows else set()
    for expr in exprs:
        for token, op in OPS:
            name, sep, raw = expr.partition(token)
            if not sep:
                continue
            name, raw = name.strip(), raw.strip()
            if name not in known:
                raise KeyError(
                    f"no column {name!r} in the table; have: {', '.join(sorted(map(str, known)))}"
                )
            kept = []
            for r in rows:
                v = r.get(name)
                if v is None:
                    continue  # a todo row has no result yet; it cannot match on one
                try:
                    if op(v, _coerce(v, raw)):
                        kept.append(r)
                except TypeError:  # e.g. "<" between a string and a float
                    continue
            rows = kept
            break
        else:
            raise ValueError(f"malformed --where {expr!r}; expected NAME=VALUE")
    return rows
