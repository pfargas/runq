import pytest

from runq.params import (
    ParamSpace,
    UnknownParamError,
    coerce_like,
    hash8,
    key_json,
    run_label,
)


def target(L=0.8, N=5, kind="jastrow", fast=False, seed=0, run_dir=None):
    return {}


def test_from_function_reads_defaults_and_run_dir():
    space = ParamSpace.from_function(target)
    assert space.defaults == {"L": 0.8, "N": 5, "kind": "jastrow", "fast": False, "seed": 0}
    assert space.accepts_run_dir


def test_run_dir_not_required():
    def f(a=1):
        return {}

    space = ParamSpace.from_function(f)
    assert not space.accepts_run_dir
    assert space.defaults == {"a": 1}


def test_missing_default_rejected():
    def f(a, b=2):
        return {}

    with pytest.raises(TypeError, match="no default"):
        ParamSpace.from_function(f)


def test_var_kwargs_rejected():
    def f(a=1, **kw):
        return {}

    with pytest.raises(TypeError, match="kwargs"):
        ParamSpace.from_function(f)


def test_resolve_fills_defaults_and_rejects_unknown():
    space = ParamSpace.from_function(target)
    full = space.resolve({"L": 1.5})
    assert full["L"] == 1.5 and full["N"] == 5
    with pytest.raises(UnknownParamError, match="nope"):
        space.resolve({"nope": 1})


@pytest.mark.parametrize(
    ("raw", "like", "expected"),
    [
        ("0.5", 1.0, 0.5),
        ("7", 3, 7),
        ("true", False, True),
        ("off", True, False),
        ("mlp", "jastrow", "mlp"),
        ("3", 1.0, 3.0),  # float default keeps float type even for int-looking input
        ("[1, 2]", None, [1, 2]),  # None default: literal_eval
        ("hello", None, "hello"),  # ... falling back to str
    ],
)
def test_coerce_like(raw, like, expected):
    got = coerce_like(raw, like)
    assert got == expected
    assert type(got) is type(expected)


def test_coerce_bad_bool():
    with pytest.raises(ValueError):
        coerce_like("maybe", True)


def test_key_json_is_order_independent_and_canonical():
    assert key_json({"b": 1, "a": 2.0}) == key_json({"a": 2.0, "b": 1})
    # int and float are distinct keys — coercion by default type prevents accidents,
    # but the key itself must distinguish N=5 from N=5.0
    assert key_json({"a": 5}) != key_json({"a": 5.0})


def test_run_label_readable_and_unique():
    p1 = {"L": 0.8, "N": 5, "lr": 3e-3}
    p2 = {"L": 0.8, "N": 5, "lr": 1e-3}  # differs only in an unswept default
    l1 = run_label(p1, swept=["L", "N"])
    l2 = run_label(p2, swept=["L", "N"])
    assert l1.startswith("L0.8_N5_")
    assert l1 != l2  # hash covers the full dict
    assert l1 == run_label(p1, swept=["L", "N"])  # stable


def test_run_label_sanitizes():
    label = run_label({"kind": "a/b c"}, swept=["kind"])
    assert "/" not in label and " " not in label


def test_hash8_stable():
    assert hash8({"a": 1}) == hash8({"a": 1})
    assert len(hash8({"a": 1})) == 8
