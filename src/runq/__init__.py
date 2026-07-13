"""runq — resumable parameter sweeps on a SQLite queue.

The whole contract is one plain function with keyword defaults returning a dict of
scalars. Everything else (grid, queue, workers, resume, merge, results table) is runq's
job. See README.md.
"""

from runq.grid import build_grid, parse_axes
from runq.params import ParamSpace, dir_hash, hash8, key_json, run_label
from runq.query import filter_rows, load_rows
from runq.runner import Skip, drain
from runq.store import connect
from runq.table import load_table
from runq.target import load_target

__version__ = "0.1.0"

__all__ = [
    "ParamSpace",
    "Skip",
    "build_grid",
    "connect",
    "dir_hash",
    "drain",
    "filter_rows",
    "hash8",
    "key_json",
    "load_rows",
    "load_table",
    "load_target",
    "parse_axes",
    "run_label",
]
