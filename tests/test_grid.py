import pytest

from runq.grid import build_grid, parse_axes
from runq.params import ParamSpace, UnknownParamError


def target(L=0.8, N=5, kind="jastrow", seed=0):
    return {}


@pytest.fixture
def space():
    return ParamSpace.from_function(target)


def test_parse_axes_coerces_types(space):
    axes = parse_axes(["L=0.5,0.8", "N=2,5", "kind=mlp"], space)
    assert axes == {"L": [0.5, 0.8], "N": [2, 5], "kind": ["mlp"]}
    assert all(type(v) is float for v in axes["L"])
    assert all(type(v) is int for v in axes["N"])


def test_parse_axes_rejects_unknown_and_malformed(space):
    with pytest.raises(UnknownParamError):
        parse_axes(["bogus=1"], space)
    with pytest.raises(ValueError, match="bad axis"):
        parse_axes(["L:0.5"], space)
    with pytest.raises(ValueError, match="no values"):
        parse_axes(["L="], space)


def test_build_grid_cartesian_product(space):
    grid = build_grid(space, {"L": [0.5, 0.8], "N": [2, 5], "seed": [0, 1]})
    assert len(grid) == 8
    assert all(p["kind"] == "jastrow" for p in grid)  # unswept default filled in
    assert len({(p["L"], p["N"], p["seed"]) for p in grid}) == 8


def test_build_grid_no_axes_is_default_point(space):
    grid = build_grid(space)
    assert grid == [{"L": 0.8, "N": 5, "kind": "jastrow", "seed": 0}]


def test_build_grid_deterministic_order(space):
    grid = build_grid(space, {"N": [2, 5]})
    assert [p["N"] for p in grid] == [2, 5]
