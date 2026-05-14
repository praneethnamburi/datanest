# SPDX-FileCopyrightText: 2024-present praneeth <praneeth@mit.edu>
#
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

import datanest


@pytest.fixture(scope="session", autouse=True)
def db():
    return datanest.get_example_database()


@pytest.fixture(scope="session", autouse=True)
def data():
    return datanest.get_example_data()


@pytest.fixture(scope="session", autouse=True)
def example_csv_file(db, tmp_path_factory):
    ret = Path(tmp_path_factory.getbasetemp()) / "participant_metadata.csv"
    db().to_csv(ret, index=None)
    return ret


@pytest.fixture(scope="session", autouse=True)
def example_xls_file(db, tmp_path_factory):
    ret = Path(tmp_path_factory.getbasetemp()) / "participant_metadata.xls"
    db().to_excel(ret, index=None)
    return ret


@pytest.fixture(scope="session", autouse=True)
def example_xlsx_file(db, tmp_path_factory):
    ret = Path(tmp_path_factory.getbasetemp()) / "participant_metadata.xlsx"
    db().to_excel(ret, index=None)
    return ret


def test_examples(db, data, tmp_path_factory):
    assert isinstance(db, datanest.Database)
    assert isinstance(data, dict)
    assert len(data) == 99

    temp_dir = Path(tmp_path_factory.getbasetemp())
    assert os.path.exists(temp_dir / "participant_metadata.csv")
    assert os.path.exists(temp_dir / "participant_metadata.xls")
    assert os.path.exists(temp_dir / "participant_metadata.xlsx")


def test_database_initialization(
    db, example_csv_file, example_xls_file, example_xlsx_file
):
    assert len(datanest.Database(db())()) == 100
    assert len(datanest.Database(example_csv_file)()) == 100
    assert len(datanest.Database(example_xls_file)()) == 100
    assert len(datanest.Database(example_xlsx_file)()) == 100


def test_database_getitem(db):
    assert isinstance(db["participant_id"], pd.Series)


def test_database_call(db):
    x = db()
    assert isinstance(x, pd.DataFrame)
    assert len(x) == 100

    x = db(participant_id=3)
    assert isinstance(x, pd.DataFrame)
    assert len(x) == 1

    x = db(age_lim=(50, 60), surgery_performed=True)
    assert isinstance(x, pd.DataFrame)
    assert len(x) == 8

    x = db(notes_has="interesting")
    assert isinstance(x, pd.DataFrame)
    assert len(x) == 20

    x = db(participant_id_any=(3, 4))
    assert isinstance(x, pd.DataFrame)
    assert len(x) == 2

    x = db(participant_id_any=(3, 143356))  # participant 143356 does not exist
    assert len(x) == 1

    x = db(kwarg_does_not_exist_lim=12)
    assert len(x) == len(db())

    # test arg
    assert len(db("surgery_performed")) == 51


# moving forward, db will have the heart_rate field in all the tests below
def test_database_add_data_field(db, data):
    db.add_data_field("heart_rate", data, "participant_id")
    assert hasattr(db, "heart_rate")
    assert hasattr(db, "_heart_rate")
    assert isinstance(db.heart_rate, Callable)
    assert isinstance(db._heart_rate, dict)
    with pytest.raises(AssertionError):
        db.add_data_field("heart_rate", data, "participant_id")

    x = db.heart_rate()
    assert isinstance(x, dict)
    assert len(x) == 99

    # test behavior of ret_type and isolate_single for methods generated using add_data_field
    x = db.heart_rate(participant_id=3)
    assert x.__class__.__name__ == "HRData"

    x = db.heart_rate(participant_id=3, ret_type=list, isolate_single=False)
    assert isinstance(x, list)
    assert len(x) == 1
    x = db.heart_rate(participant_id=3, ret_type=dict, isolate_single=False)
    assert isinstance(x, dict)
    assert len(x) == 1

    x = db.heart_rate(participant_id=3, ret_type=list, isolate_single=True)
    assert x.__class__.__name__ == "HRData"
    x = db.heart_rate(participant_id=3, ret_type=dict, isolate_single=True)
    assert x.__class__.__name__ == "HRData"

    x = db.heart_rate(age_lim=(50, 60), surgery_performed=True)
    assert isinstance(x, dict)
    assert len(x) == 8

    x = db.heart_rate(notes_has="interesting")
    assert isinstance(x, dict)
    assert len(x) == 20

    x = db.heart_rate(participant_id_any=(3, 4))
    assert isinstance(x, dict)
    assert len(x) == 2

    x = db.heart_rate(
        participant_id_any=(3, 143356)
    )  # participant 143356 does not exist
    assert x.__class__.__name__ == "HRData"

    x = db.heart_rate(kwarg_does_not_exist_lim=12)
    assert len(x) == len(db.heart_rate())

    assert (
        len(db.heart_rate("surgery_performed")) == 51
    )  # caught a bug here in the datanest.Database.get method
    assert (
        len(db.heart_rate(surgery_performed=False)) == 48
    )  # missing entry for participant 23


def test_database_get(db, example_csv_file, capsys):
    db2 = datanest.Database(example_csv_file)
    field_name = "heart_rate"
    assert db2.get(field_name) is None
    captured = capsys.readouterr()
    assert (
        captured.out.strip()
        == f"{field_name} not found. Use db.add_data_field({field_name}, data)."
    )

    db2.add_data_field(field_name, {}, data_key_name="participant_id")
    assert db2.heart_rate() is None
    captured = capsys.readouterr()
    assert captured.out.strip() == f"db._{field_name} is empty. Nothing to return."

    x = db.heart_rate(
        surgery_performed=False, ret_type=list
    )  # missing entry for participant 23
    captured = capsys.readouterr()
    assert (
        captured.out.strip()
        == f"WARNING: Missing values in {field_name}, use ret_type=dict to reduce errors."
    )

    x = db.get("heart_rate", "surgery_performed", id_column_name="participant_id")
    assert isinstance(x, list)
    assert len(x) == 51

    x = db.get("heart_rate", None, "surgery_performed", id_column_name="participant_id")
    assert isinstance(x, list)
    assert len(x) == 51

    x = db.get(
        "heart_rate", None, dict, "surgery_performed", id_column_name="participant_id"
    )
    assert isinstance(x, dict)
    assert len(x) == 51


# ---------------------------------------------------------------------------
# Reserved-suffix collision detection (1.2.0)
# ---------------------------------------------------------------------------
def _df_with_collision():
    return pd.DataFrame(
        {
            "age": [20, 30, 40],
            "age_lim": [1, 2, 3],
            "notes": ["a", "b", "c"],
        }
    )


def test_warns_on_reserved_suffix_collision():
    with pytest.warns(datanest.ReservedSuffixCollisionWarning):
        datanest.Database(_df_with_collision())


def test_raises_on_reserved_suffix_collision_strict():
    with pytest.raises(ValueError, match="reserved query suffixes"):
        datanest.Database(_df_with_collision(), on_reserved_suffix="raise")


def test_ignore_silences_reserved_suffix_check(recwarn):
    datanest.Database(_df_with_collision(), on_reserved_suffix="ignore")
    assert not any(
        issubclass(w.category, datanest.ReservedSuffixCollisionWarning) for w in recwarn
    )


def test_no_warning_when_only_suffixed_column_present(recwarn):
    # `xyz_has` alone (no `xyz`) is not a collision — the suffix branch in
    # __call__ never fires because the would-be base isn't in the DataFrame.
    df = pd.DataFrame({"xyz_has": [1, 2], "notes": ["a", "b"]})
    datanest.Database(df)
    assert not any(
        issubclass(w.category, datanest.ReservedSuffixCollisionWarning) for w in recwarn
    )


def test_invalid_on_reserved_suffix_value():
    with pytest.raises(ValueError, match="on_reserved_suffix must be one of"):
        datanest.Database(_df_with_collision(), on_reserved_suffix="bogus")


def test_rename_mode_renames_lim():
    df = _df_with_collision()
    with pytest.warns(datanest.ReservedSuffixCollisionWarning, match="Auto-renamed"):
        db = datanest.Database(df, on_reserved_suffix="rename")
    cols = list(db().columns)
    assert "age_lim" not in cols
    assert "age_limits" in cols
    # Original user DataFrame should not be mutated.
    assert "age_lim" in df.columns


def test_rename_mode_covers_all_three_suffixes():
    df = pd.DataFrame(
        {
            "age": [1, 2],
            "age_lim": [10, 20],
            "notes": ["x", "y"],
            "notes_has": ["a", "b"],
            "level": [3, 4],
            "level_any": [5, 6],
        }
    )
    with pytest.warns(datanest.ReservedSuffixCollisionWarning):
        db = datanest.Database(df, on_reserved_suffix="rename")
    cols = set(db().columns)
    assert {"age_limits", "notes_contains", "level_options"} <= cols
    assert {"age_lim", "notes_has", "level_any"}.isdisjoint(cols)


def test_rename_mode_query_uses_suffix_branch_after_rename():
    # Before rename: age_lim shadows the suffix branch. After rename, the
    # suffix branch on `age` works normally.
    df = _df_with_collision()
    with pytest.warns(datanest.ReservedSuffixCollisionWarning):
        db = datanest.Database(df, on_reserved_suffix="rename")
    result = db(age_lim=(25, 35))
    assert len(result) == 1
    assert result.iloc[0]["age"] == 30


def test_rename_mode_raises_on_target_collision():
    # If the rename target already exists, refuse to clobber.
    df = pd.DataFrame(
        {
            "age": [1, 2],
            "age_lim": [10, 20],
            "age_limits": [100, 200],
        }
    )
    with pytest.raises(ValueError, match="target column already exists"):
        datanest.Database(df, on_reserved_suffix="rename")


def test_database_records(db, example_csv_file):
    r = db.records()
    assert len(r) == 100
    assert "heart_rate" in r[0]
    assert r[0]["heart_rate"].__class__.__name__ == "HRData"

    r = db.records("surgery_performed")
    assert len(r) == 51
    r = db.records(surgery_performed=False)
    assert len(r) == 49

    hdr = db("surgery_performed")
    assert isinstance(hdr, pd.DataFrame)
    assert len(db.records(hdr)) == 51

    db2 = datanest.Database(example_csv_file)
    assert len(db2.records()) == 100
    assert "heart_rate" not in db2.records()[0]
