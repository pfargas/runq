import pytest

from runq.target import load_target


def test_load_from_file_default_func(toy_path):
    fn = load_target(toy_path)
    assert fn.__name__ == "run_point"
    assert fn(a=2.0, b=3)["energy"] == 6.0


def test_load_from_file_explicit_func(toy_path):
    fn = load_target(f"{toy_path}:run_point")
    assert callable(fn)


def test_load_from_module():
    fn = load_target("json:dumps")
    assert fn({"a": 1}) == '{"a": 1}'


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_target("does/not/exist.py")


def test_missing_function(toy_path):
    with pytest.raises(AttributeError, match="no function"):
        load_target(f"{toy_path}:nope")


def test_non_callable():
    with pytest.raises(TypeError, match="non-callable"):
        load_target("json:__name__")
