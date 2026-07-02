"""runq — resumable parameter sweeps on a SQLite queue.

The whole contract is one plain function with keyword defaults returning a dict of
scalars. Everything else (grid, queue, workers, resume, merge, results table) is runq's
job. See README.md.
"""

from runq.grid import build_grid, parse_axes
from runq.params import ParamSpace, key_json, run_label
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
    "drain",
    "key_json",
    "load_table",
    "load_target",
    "parse_axes",
    "run_label",
]
