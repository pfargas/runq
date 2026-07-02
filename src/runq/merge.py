"""Merge several sweep DBs into one (multi-machine / multi-task runs).

Rows are keyed by ``params_json``, so merging is done-precedence upsert: a finished
result never loses to a leftover ``todo``/``running`` from another copy. Copy each
machine's ``runs/`` artifact dirs into a shared ``runs/`` too — labels are unique per
point, so the union is collision-free.
"""

from __future__ import annotations

import sqlite3

from runq import store

# skipped is a legitimate final state; failed may still be retried
_RANK = {"done": 4, "skipped": 3, "failed": 2, "running": 1, "todo": 0}

_COLS = ("params_json", "label", "status", "result_json", "run_dir", "error",
         "started_at", "finished_at")


def merge_into(dest: sqlite3.Connection, source_path: str) -> int:
    """Fold one source DB into ``dest``. Returns the number of rows taken from it."""
    src = sqlite3.connect(source_path)
    src.row_factory = sqlite3.Row
    taken = 0
    try:
        for r in src.execute(f"SELECT {', '.join(_COLS)} FROM runs").fetchall():
            existing = dest.execute(
                "SELECT status FROM runs WHERE params_json=?", (r["params_json"],)
            ).fetchone()
            if existing and _RANK.get(existing["status"], 0) >= _RANK.get(r["status"], 0):
                continue  # keep the better-status row we already have
            dest.execute("DELETE FROM runs WHERE params_json=?", (r["params_json"],))
            dest.execute(
                f"INSERT INTO runs ({', '.join(_COLS)}) "
                f"VALUES ({','.join('?' * len(_COLS))})",
                tuple(r[c] for c in _COLS),
            )
            taken += 1
    finally:
        src.close()
    return taken


def merge_paths(dest_path: str, source_paths: list[str]) -> dict:
    """Merge ``source_paths`` into ``dest_path`` (created if missing). Final counts."""
    dest = store.connect(dest_path)
    try:
        for p in source_paths:
            n = merge_into(dest, p)
            print(f"merged {n} row(s) from {p}")
        return store.status_counts(dest)
    finally:
        dest.close()
