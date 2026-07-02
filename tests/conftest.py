import pathlib

import pytest

from runq.params import ParamSpace
from runq.target import load_target

TOY_PY = pathlib.Path(__file__).parent / "targets" / "toy.py"


@pytest.fixture
def toy_path() -> str:
    return str(TOY_PY)


@pytest.fixture
def toy_fn(toy_path):
    return load_target(toy_path)


@pytest.fixture
def toy_space(toy_fn) -> ParamSpace:
    return ParamSpace.from_function(toy_fn)
