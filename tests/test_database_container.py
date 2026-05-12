"""Tests for ``datanest.DatabaseContainer`` using the wobble CSV fixtures.

The fixtures under ``tests/fixtures/wobble/`` are the same metadata CSVs that
``pn-projects/projects/wobble`` uses, so the test exercises the real id-tuple
patterns (subject_id, trial_id, action_id) without paying wobble's full
``__init__`` cost. Payloads are plain NumPy arrays — ``add_data_field`` is
modality-agnostic by design, so the synthesized payload type does not affect
what's under test.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import datanest

FIXTURES = Path(__file__).parent / "fixtures" / "wobble"


# ---------------------------------------------------------------------------
# Fixture loaders — mirror wobble's data prep but with zero side effects.
# ---------------------------------------------------------------------------
def _load_subject_df() -> pd.DataFrame:
    df = pd.read_csv(FIXTURES / "subject_database.csv")
    df["subject_id"] = df.apply(lambda row: (row.experiment, row.subject), axis=1)
    df["expert"] = df.category == "expert"
    df["nonexpert"] = df.category == "nonexpert"
    df["id"] = df["subject_id"]
    df.set_index("id", inplace=True)
    return df


def _load_trial_df() -> pd.DataFrame:
    df = pd.read_csv(FIXTURES / "trial_database.csv")
    df["id"] = df["id"].apply(eval)
    df["trial_id"] = df["id"]
    df.set_index("id", inplace=True)
    df.rename(columns={"type": "trial_type"}, inplace=True)
    return df


def _load_action_df() -> pd.DataFrame:
    df = pd.read_csv(FIXTURES / "action_database.csv")
    df["trial_id"] = df["trial_id"].apply(eval)
    df["action_id"] = df["action_id"].apply(eval)
    df["id"] = df["action_id"]
    df.set_index("id", inplace=True)
    return df


def _make_synthetic_signals(ids, n_samples: int = 100):
    """Generate one tiny NumPy array per id.

    Keyed by ``id`` (tuple) so it can be attached via ``add_data_field``.
    The payload type is intentionally a plain ``np.ndarray`` — ``datanest``
    is modality-agnostic, so the array stands in for any object a real
    project would attach (``pysampled.Data``, images, custom classes, ...).
    """
    rng = np.random.default_rng(seed=0)
    return {_id: rng.standard_normal(n_samples) for _id in ids}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def subject_db():
    return datanest.Database(_load_subject_df())


@pytest.fixture(scope="module")
def trial_db():
    return datanest.Database(_load_trial_df())


@pytest.fixture(scope="module")
def action_db():
    return datanest.Database(_load_action_df())


@pytest.fixture(scope="module")
def dbc(subject_db, trial_db, action_db):
    c = datanest.DatabaseContainer()
    c.add("subject", subject_db)
    c.add("trial", trial_db, "subject", lambda trial_id: trial_id[:2])
    c.add("action", action_db, "trial", lambda action_id: action_id[:3])
    return c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_hierarchy_structure(dbc):
    assert dbc.all_db_names == ["subject", "trial", "action"]
    assert dbc._db_level == {"subject": 0, "trial": 1, "action": 2}
    assert dbc.get_heritage("action") == ["trial", "subject"]
    assert dbc.get_heritage("trial") == ["subject"]
    assert dbc.get_heritage("subject") == []


def test_level_lookup(dbc):
    assert dbc.get_db_name_of_column("age") == "subject"
    assert dbc.get_db_name_of_column("action_phase") == "action"


def test_top_level_query(dbc):
    df = dbc(subject=3)
    assert len(df) >= 1
    # subject 3 only exists in experiment 1 in this fixture
    assert all(s == 3 for s in df["subject"])


def test_range_suffix_query(dbc):
    df = dbc(age_lim=(20, 25))
    assert not df.empty
    assert df["age"].between(20, 25).all()


def test_substring_suffix_query(dbc):
    df = dbc(activity_description_has="Tennis")
    assert not df.empty
    assert df["activity_description"].str.contains("Tennis").all()


def test_set_suffix_query(dbc):
    df = dbc(category_any=("expert", "nonexpert"))
    assert not df.empty
    assert df["category"].isin(("expert", "nonexpert")).all()


def test_bool_positional_arg(dbc):
    df = dbc("expert")
    assert not df.empty
    assert df["expert"].all()


def test_query_returns_records(dbc):
    recs = dbc.records(subject=3)
    assert isinstance(recs, list)
    assert recs and isinstance(recs[0], dict)
    assert all(rec["subject"] == 3 for rec in recs)


def test_cross_level_filter_propagates(dbc):
    # subject metadata predicate should narrow trial/action rows downstream
    expert_actions = dbc(level="action", category="expert")
    if expert_actions.empty:
        pytest.skip("no expert-action rows in fixture")
    assert all(a[:2] in set(dbc(category="expert").subject_id) for a in expert_actions.index)


def test_add_data_field_at_subject_level(dbc, subject_db):
    payload = _make_synthetic_signals(subject_db().index)
    subject_db.add_data_field("hr", payload, "subject_id")

    hr = dbc.hr(subject=3)
    assert isinstance(hr, dict)
    keys = list(hr.keys())
    assert len(keys) >= 1
    assert all(k[1] == 3 for k in keys)


def test_add_data_field_at_trial_level(dbc, trial_db):
    payload = _make_synthetic_signals(trial_db().index)
    trial_db.add_data_field("emg_rms", payload, "trial_id")

    rms = dbc.emg_rms(subject=1)
    assert isinstance(rms, dict)
    assert rms
    # every returned key should be a trial_id whose subject_id (first two) is (_, 1)
    assert all(k[1] == 1 for k in rms.keys())


# Pickle round-trip of a ``DatabaseContainer`` instance is intentionally
# not tested here. The class's ``__getattr__`` returns ``None`` for unknown
# keys (a long-standing convenience that predates datanest), which
# interferes with pickle's dunder probing. Pickle-compat for the migration
# is provided by the ``immersionlab.DatabaseContainer`` re-export shim
# itself (keeps the legacy module path resolvable); no DatabaseContainer
# instances are pickled directly in the downstream pn-projects code
# (caching there is on method outputs).


def test_help_runs(dbc, capsys):
    # help() should print without raising; we just smoke-test it
    dbc.help()
    out = capsys.readouterr().out
    assert "subject" in out
