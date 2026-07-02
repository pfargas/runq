"""Loading the target function: ``path/to/file.py[:func]`` or ``pkg.module[:func]``.

The default function name is ``run_point``. Loading a file by path puts its directory
on ``sys.path`` first, so sibling imports inside a project directory keep working
(the same trick the qvarnet sweep scripts relied on).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys


def load_target(spec: str):
    mod_spec, _, func = spec.partition(":")
    func = func or "run_point"

    if mod_spec.endswith(".py") or os.sep in mod_spec:
        path = os.path.abspath(mod_spec)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"target file not found: {path}")
        name = "_runq_target_" + os.path.splitext(os.path.basename(path))[0]
        sys.path.insert(0, os.path.dirname(path))  # sibling imports resolve
        spec_ = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec_)
        sys.modules[name] = module
        spec_.loader.exec_module(module)
    else:
        module = importlib.import_module(mod_spec)

    try:
        fn = getattr(module, func)
    except AttributeError:
        raise AttributeError(
            f"{mod_spec!r} has no function {func!r}; pass TARGET as file.py:func "
            "or module:func (default func: run_point)"
        ) from None
    if not callable(fn):
        raise TypeError(f"{spec!r} resolved to a non-callable {type(fn).__name__}")
    return fn
