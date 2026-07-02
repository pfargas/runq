"""Grid construction: cartesian product of axes over the function's defaults.

An *axis* maps a parameter name to the list of values it takes; every parameter not
named keeps its default. ``build_grid`` with no axes is the single default point.
Random search and tuners are planned on top of the same representation (see README).
"""

from __future__ import annotations

import itertools

from runq.params import ParamSpace


def parse_axes(specs: list[str], space: ParamSpace) -> dict[str, list]:
    """``["L=0.5,0.8", "N=2,5"]`` -> ``{"L": [0.5, 0.8], "N": [2, 5]}`` (typed).

    Values are coerced to the type of each parameter's default. Unknown names raise
    :class:`runq.params.UnknownParamError` listing the known parameters.
    """
    axes: dict[str, list] = {}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f"bad axis {spec!r}; use NAME=v1,v2,...")
        name, raw = spec.split("=", 1)
        name = name.strip()
        values = [space.coerce(name, v) for v in raw.split(",") if v != ""]
        if not values:
            raise ValueError(f"axis {name!r} has no values")
        axes[name] = values
    return axes


def build_grid(space: ParamSpace, axes: dict[str, list] | None = None) -> list[dict]:
    """All fully resolved parameter dicts of the cartesian product of ``axes``.

    Deterministic order (axes in given order, values in given order). No axes ⇒ one
    point: the defaults.
    """
    axes = axes or {}
    names = list(axes)
    return [
        space.resolve(dict(zip(names, combo)))
        for combo in itertools.product(*(axes[n] for n in names))
    ]
