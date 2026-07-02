"""Parameter-space introspection: one plain function defines the whole space.

Every keyword argument with a default is a parameter of the space — there is no
physics/hyperparameter split anywhere in runq: one flat namespace, one ``--axis``
vocabulary. The defaults provide the types used to coerce CLI strings, and the values
used for any axis not being swept.

``run_dir`` is reserved: if the function accepts it, the runner injects the per-run
artifact directory; it is never part of the parameter space or the key.

The **key** of a run is the canonical JSON of the fully resolved parameter dict
(defaults + overrides) — see :func:`key_json`. The human-readable **label** puts the
swept axes up front and closes with a short hash of the full dict, so two runs that
differ only in an unswept default still get distinct labels.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import re
from dataclasses import dataclass, field

RESERVED = ("run_dir",)


class UnknownParamError(KeyError):
    """An override or axis names a parameter the target function does not have."""


@dataclass(frozen=True)
class ParamSpace:
    """The flat parameter space of one target function."""

    defaults: dict = field(default_factory=dict)
    accepts_run_dir: bool = False

    @classmethod
    def from_function(cls, fn) -> "ParamSpace":
        """Read the space off ``fn``'s signature. Every parameter needs a default."""
        defaults: dict = {}
        accepts_run_dir = False
        for name, p in inspect.signature(fn).parameters.items():
            if name in RESERVED:
                accepts_run_dir = True
                continue
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                raise TypeError(
                    f"{getattr(fn, '__name__', fn)!r} uses *args/**kwargs; runq needs "
                    "explicit keyword parameters to know the space"
                )
            if p.default is inspect.Parameter.empty:
                raise TypeError(
                    f"parameter {name!r} of {getattr(fn, '__name__', fn)!r} has no "
                    "default; runq reads types and unswept values from the defaults"
                )
            defaults[name] = p.default
        return cls(defaults=defaults, accepts_run_dir=accepts_run_dir)

    def resolve(self, overrides: dict | None = None) -> dict:
        """Full parameter dict = defaults + overrides. Rejects unknown names."""
        overrides = overrides or {}
        unknown = set(overrides) - set(self.defaults)
        if unknown:
            raise UnknownParamError(
                f"unknown parameter(s) {sorted(unknown)}; "
                f"known: {', '.join(self.defaults)}"
            )
        return {**self.defaults, **overrides}

    def coerce(self, name: str, raw: str):
        """Coerce a CLI string to the type of ``name``'s default."""
        if name not in self.defaults:
            raise UnknownParamError(
                f"unknown parameter {name!r}; known: {', '.join(self.defaults)}"
            )
        return coerce_like(raw, self.defaults[name])


def coerce_like(raw: str, like):
    """Coerce string ``raw`` to the type of ``like`` (a parameter's default value)."""
    if isinstance(like, bool):
        v = raw.strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
        if v in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"cannot read {raw!r} as a bool")
    if isinstance(like, int):
        return int(raw)
    if isinstance(like, float):
        return float(raw)
    if isinstance(like, str):
        return raw
    if like is None:
        # no type to coerce toward: take Python literals, fall back to the string
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return raw
    raise TypeError(f"unsupported default type {type(like).__name__}")


# ── canonical key & human-readable label ─────────────────────────────────────────────


def key_json(params: dict) -> str:
    """Canonical JSON of a fully resolved parameter dict — the DB key."""
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def hash8(params: dict) -> str:
    return hashlib.sha1(key_json(params).encode()).hexdigest()[:8]


def run_label(params: dict, swept: list[str] | tuple[str, ...] = ()) -> str:
    """Filesystem-safe label: swept axes readable up front, hash of the full dict last.

    The hash covers *all* params (swept or not), so the label is unique per point even
    when two sweeps share swept values but differ in a default.
    """
    parts = [f"{name}{_fmt(params[name])}" for name in swept]
    parts.append(hash8(params))
    return "_".join(parts)


def _fmt(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        s = f"{v:g}"
    else:
        s = str(v)
    return re.sub(r"[^A-Za-z0-9.+-]", "-", s)
