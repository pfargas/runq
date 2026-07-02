"""Resumable SQLite work queue. One row per point, keyed by canonical params JSON.

Statuses: ``todo`` -> ``running`` -> ``done`` | ``failed`` | ``skipped``. Enqueue is
idempotent (``INSERT OR IGNORE``), so re-running a sweep extends it; ``claim_next`` uses
``BEGIN IMMEDIATE`` so concurrent workers on one WAL database serialise on the write
lock and never grab the same point. Results are one JSON blob (``result_json``) — no
project-specific columns, ever; :func:`runq.table.load_table` expands them.

Same-host only: SQLite's locking is unsafe across nodes on NFS/Lustre. Multi-node runs
use per-task DBs merged afterwards (:mod:`runq.merge`, SLURM backend).
"""

from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime

DEFAULT_DB = "outputs/runq.db"

STATUSES = ("todo", "running", "done", "failed", "skipped")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY,
    params_json TEXT NOT NULL UNIQUE,
    label       TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'todo',
    result_json TEXT,
    run_dir     TEXT,
    error       TEXT,
    started_at  TEXT,
    finished_at TEXT
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect(path: str = DEFAULT_DB) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # autocommit (isolation_level=None): every statement commits itself, and the
    # explicit BEGIN IMMEDIATE in claim_next owns its transaction unambiguously.
    conn = sqlite3.connect(path, timeout=60, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


def enqueue(conn, params_json: str, label: str) -> bool:
    """Insert a ``todo`` row unless this point is already recorded. True if inserted."""
    cur = conn.execute(
        "INSERT OR IGNORE INTO runs (params_json, label, status) VALUES (?, ?, 'todo')",
        (params_json, label),
    )
    return cur.rowcount == 1


def status_of(conn, params_json: str) -> str | None:
    row = conn.execute(
        "SELECT status FROM runs WHERE params_json=?", (params_json,)
    ).fetchone()
    return row["status"] if row else None


def claim_next(conn) -> sqlite3.Row | None:
    """Atomically claim one ``todo`` row (multi-worker safe). Row or None (queue empty).

    The returned row carries ``id, params_json, label`` — everything a worker needs.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT id, params_json, label FROM runs WHERE status='todo' "
            "ORDER BY id LIMIT 1"
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None
        conn.execute(
            "UPDATE runs SET status='running', started_at=?, error=NULL WHERE id=?",
            (_now(), row["id"]),
        )
        conn.execute("COMMIT")
        return row
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def save_result(conn, run_id: int, result_json: str, run_dir: str | None = None) -> None:
    conn.execute(
        "UPDATE runs SET status='done', result_json=?, run_dir=?, finished_at=? "
        "WHERE id=?",
        (result_json, run_dir, _now(), run_id),
    )


def mark_failed(conn, run_id: int, error: str) -> None:
    conn.execute(
        "UPDATE runs SET status='failed', error=?, finished_at=? WHERE id=?",
        (error[:8000], _now(), run_id),
    )


def mark_skipped(conn, run_id: int, reason: str) -> None:
    conn.execute(
        "UPDATE runs SET status='skipped', error=?, finished_at=? WHERE id=?",
        (reason[:2000], _now(), run_id),
    )


def requeue(conn, statuses: tuple[str, ...] = ("running",)) -> int:
    """Reset rows in ``statuses`` back to ``todo`` (crashed or retried). Returns count."""
    qs = ",".join("?" * len(statuses))
    cur = conn.execute(
        f"UPDATE runs SET status='todo', error=NULL WHERE status IN ({qs})", statuses
    )
    return cur.rowcount


def status_counts(conn) -> dict:
    rows = conn.execute("SELECT status, COUNT(*) AS n FROM runs GROUP BY status")
    return {r["status"]: r["n"] for r in rows.fetchall()}


def fetch(conn, status: str | None = None) -> list[sqlite3.Row]:
    if status is None:
        return conn.execute("SELECT * FROM runs ORDER BY id").fetchall()
    return conn.execute(
        "SELECT * FROM runs WHERE status=? ORDER BY id", (status,)
    ).fetchall()
