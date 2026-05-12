# SPDX-FileCopyrightText: 2024-present praneeth <praneeth@mit.edu>
#
# SPDX-License-Identifier: MIT

import os
import time

import dill
import pytest

import datanest


def test_cache_me_if_you_can_round_trip(tmp_path):
    """First call computes and writes; second call reads from disk."""
    fname = tmp_path / "x.pkl"
    calls = {"n": 0}

    @datanest.cache_me_if_you_can(fname)
    def f():
        calls["n"] += 1
        return 42

    assert f() == 42
    assert os.path.exists(fname)
    assert f() == 42
    assert calls["n"] == 1


def test_cache_me_if_you_can_suffix_distinct_files(tmp_path):
    """Different suffix outputs produce different cache files."""
    fname = tmp_path / "x.pkl"
    calls = {"n": 0}

    @datanest.cache_me_if_you_can(fname, suffix=lambda *a, **kw: "_" + a[0])
    def f(tag):
        calls["n"] += 1
        return f"computed_{tag}"

    assert f("a") == "computed_a"
    assert f("b") == "computed_b"
    assert calls["n"] == 2
    assert (tmp_path / "x_a.pkl").exists()
    assert (tmp_path / "x_b.pkl").exists()
    # cache hit on repeat
    assert f("a") == "computed_a"
    assert calls["n"] == 2


def test_cache_me_if_you_can_verbose_does_not_crash(tmp_path, capsys):
    fname = tmp_path / "x.pkl"

    @datanest.cache_me_if_you_can(fname, verbose=True)
    def f():
        return 1

    f()
    f()
    out = capsys.readouterr().out
    assert "is creating" in out
    assert "is loading data from" in out


def test_cache_me_if_you_can_str_path_accepted(tmp_path):
    """cache_fname accepts both str and Path."""
    fname = str(tmp_path / "x.pkl")

    @datanest.cache_me_if_you_can(fname)
    def f():
        return [1, 2, 3]

    assert f() == [1, 2, 3]
    assert os.path.exists(fname)


def test_cache_me_if_you_can_incremental_accumulates(tmp_path):
    """Dict accumulator grows across calls."""
    fname = tmp_path / "y.pkl"

    @datanest.cache_me_if_you_can_incremental(
        fname, return_name="ret", return_default={}
    )
    def g(key, value, ret=None):
        if key not in ret:
            ret[key] = value
        return ret

    g("a", 1)
    r = g("b", 2)
    assert r == {"a": 1, "b": 2}

    # Persisted on disk
    with open(fname, "rb") as f:
        assert dill.load(f) == {"a": 1, "b": 2}


def test_cache_me_if_you_can_incremental_no_op_skips_rewrite(tmp_path):
    """Unchanged dict keys skip the file rewrite (mtime unchanged)."""
    fname = tmp_path / "y.pkl"

    @datanest.cache_me_if_you_can_incremental(
        fname, return_name="ret", return_default={}
    )
    def g(key, value, ret=None):
        if key not in ret:
            ret[key] = value
        return ret

    g("a", 1)
    mtime_before = os.path.getmtime(fname)
    time.sleep(0.05)

    # Key already present -> function returns unchanged dict -> no rewrite
    r = g("a", 99)
    assert r == {"a": 1}
    assert os.path.getmtime(fname) == mtime_before


def test_cache_me_if_you_can_incremental_force_save(tmp_path):
    """force_save=True rewrites the file even when keys are unchanged."""
    fname = tmp_path / "y.pkl"

    @datanest.cache_me_if_you_can_incremental(
        fname, return_name="ret", return_default={}, force_save=True
    )
    def g(key, value, ret=None):
        if key not in ret:
            ret[key] = value
        return ret

    g("a", 1)
    mtime_before = os.path.getmtime(fname)
    time.sleep(0.05)

    g("a", 99)  # no key change
    assert os.path.getmtime(fname) > mtime_before


def test_cache_me_if_you_can_incremental_suffix(tmp_path):
    """suffix produces distinct accumulator files per group."""
    fname = tmp_path / "y.pkl"

    @datanest.cache_me_if_you_can_incremental(
        fname,
        return_name="ret",
        return_default={},
        suffix=lambda *a, **kw: "_" + a[0],
    )
    def g(group, key, value, ret=None):
        if key not in ret:
            ret[key] = value
        return ret

    g("alpha", "k1", 1)
    g("beta", "k1", 100)

    assert (tmp_path / "y_alpha.pkl").exists()
    assert (tmp_path / "y_beta.pkl").exists()
    with open(tmp_path / "y_alpha.pkl", "rb") as f:
        assert dill.load(f) == {"k1": 1}
    with open(tmp_path / "y_beta.pkl", "rb") as f:
        assert dill.load(f) == {"k1": 100}


def test_cache_me_if_you_can_incremental_verbose(tmp_path, capsys):
    fname = tmp_path / "y.pkl"

    @datanest.cache_me_if_you_can_incremental(
        fname, return_name="ret", return_default={}, verbose=True
    )
    def g(key, value, ret=None):
        if key not in ret:
            ret[key] = value
        return ret

    g("a", 1)
    g("b", 2)
    out = capsys.readouterr().out
    assert "will create" in out
    assert "will add to" in out


def test_module_path():
    """Public API resolves to the cache submodule."""
    assert datanest.cache_me_if_you_can.__module__ == "datanest.cache"
    assert datanest.cache_me_if_you_can_incremental.__module__ == "datanest.cache"
