"""A simple, intuitive, pandas-based database.

Perfect for handling data such as time series, images, or any
Python objects alongside their metadata. This tool encapsulates a pandas
DataFrame containing metadata and Python objects. It provides an intuitive
data and metadata retrieval syntax through keyword-arguments.

The :py:class:`Database` class initializes a database with a pandas DataFrame containing metadata and various data types such as time series or images.
Use :py:meth:`Database.add_data_field` to incorporate a new dictionary mapping metadata columns in the DataFrame to arbitrary Python objects.
Utilize :py:meth:`Database.__call__` for metadata retrieval, specifying criteria in Python keyword arguments."

"""
# SPDX-FileCopyrightText: 2024-present Praneeth Namburi <praneeth.namburi@gmail.com>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import re
import functools
from pathlib import Path
from typing import Any, Hashable, MutableMapping, Union, Callable

import numpy as np
import pandas as pd

__version__ = "1.0.0"
__all__ = ["Database"]


class Database:
    """Manipulate data (signals/images/arbitrary Python objects) and their metadata stored in a CSV/pandas DataFrame.

    A `Database` instance encapsulates a pandas DataFrame, and is designed for use with dictionary-like storage of data structures, such as time series data.
    It facilitates working with data (e.g., time series/images, etc.) alongside the metadata stored in a pandas DataFrame.
    This tool provides an intuitive and flexible way to retrieve relevant data from data structures based on metadata organized in the DataFrame.
    It was developed to address performance issues when storing arbitrary data objects, including NumPy arrays, in a pandas DataFrame.

    Args:
        data (Union[pd.DataFrame, str, Path]):
            - (str) Path to a CSV or Excel file. It is read as a pandas DataFrame. Make sure openpyxl is installed when working with excel files.
            - (pd.DataFrame) Pass an already loaded pandas DataFrame.

    Attributes:
        data_fields (list): Names of data dictionaries added using the :py:meth:`Database.add_data_field` method.
        data_key_names (dict): Mapping from the names of the data fields contained in `Database.data_fields` to the names of columns.
            Each data field, e.g., heart_rate_data, maps a column from the DataFrame `data`, e.g., participant_id, to the time series containing heart rate values.
            `data_key_names` stores {'heart_rate_data': 'participant_id'}, meaning that heart_rate_data is indexed using participant_id.

    Examples:
        ::

            import datanest
            # Load metadata from CSV file with columns: participant_id (int), age (float), surgery_performed (bool), notes (str)
            db = datanest.Database(r'C:\data\participant_data.csv')
            db = datanest.get_example_database()
            # Add heart rate data to the database, indexed by participant_id
            db.add_data_field('heart_rate', load_heart_rate_data(), 'participant_id')
            # Retrieve heart rate time series data for participant 3
            db.heart_rate(participant_id=3)
            # Retrieve heart rate time series data from participants aged 50 to 60
            db.heart_rate(age_lim=(50, 60))
            # Retrieve heart rate time series data from participants where the notes string contains the word interesting
            db.heart_rate(notes_has='interesting')
    """

    def __init__(self, data: Union[pd.DataFrame, str, Path]):
        if isinstance(data, pd.DataFrame):
            self._data = data
        elif isinstance(data, (str, Path)):
            fname = Path(data)
            assert os.path.exists(fname)
            assert fname.suffix in (".xls", ".xlsx", ".csv")
            if fname.suffix == ".csv":
                loader = pd.read_csv
            else:
                loader = pd.read_excel
            self._data = loader(fname)
        else:
            raise TypeError

        self.data_fields = []
        self.data_key_names = {}

    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        """Select rows from the metadata in the DataFrame.
        It provides an intuitive python kwargs (keyword arguments) based syntax.

        Keywords in kwargs can be any column name in the underlying DataFrame.
        Special keywords have the format <column_name>_<suffix>, where the suffix can be either 'any', 'lim', or 'has'.

        - The *any* suffix is useful to to specify *or* conditions, for example, `participant_id_any=(1,3)` retrieves rows whose participant_id matches either 1 or 3
        - The *lim* suffix is useful to specify limits, for example, `age_lim=(40,60)` retrieves rows where age is between 40 and 60, both included.
        - The *has* suffix is useful when working with entries that have strings, such as `notes_has='interesting'`, which will retrieve all rows where the word *interesting* is present in the notes entry.

        Arguments can be any column name of the underlying DataFrame containing boolean values.
        For example, passing an argument `'surgery_performed'` is equivalent to passing a keyword argument `surgery_performed=True`.

        Returns:
            pd.DataFrame: A DataFrame containing relevant rows.

        Examples:
            ::

                # Returns row where participant_id is 1
                db(participant_id=1)
                # Returns rows for participants with id 1 and 4
                db(participant_id_any=(1, 4))
                # Returns rows of participants between ages 40 and 60
                db(age_lim=(40,60))
                # Returns rows of participants between ages 40 and 60 who have had surgery
                db(age_lim=(40,60), surgery_performed=True)
        """
        df = self.get_df()
        sel = []
        for k in args:
            if k in df and df[k].dtype == bool:
                sel.append(df[k])

        for k, v in kwargs.items():
            if k.endswith("_lim") and (_removesuffix(k, "_lim") in df):
                assert len(v) == 2
                tk = _removesuffix(k, "_lim")
                sel.append(df[tk] >= v[0])
                sel.append(df[tk] <= v[1])
            elif k.endswith("_any") and (_removesuffix(k, "_any") in df):
                tk = _removesuffix(k, "_any")
                this_sel = []
                for this_val in v:
                    this_sel.append(df[tk] == this_val)
                sel.append(pd.Series(np.logical_or.reduce(this_sel)))
            elif k.endswith("_has") and (
                _removesuffix(k, "_has") in df
            ):  # e.g. notes_has='eyes closed'
                tk = _removesuffix(k, "_has")
                sel.append([v in x for x in df[tk].values])
            if k in df:
                sel.append((df[k] == v))

        if not sel:
            return df
        return df[np.logical_and.reduce(sel)]

    def get_df(self) -> pd.DataFrame:
        """Get the underlying DataFrame."""
        return self._data

    def __getitem__(self, key: Any):
        """Convenience method to access DataFrame columns.
        Use the square brackets on a database object as if you would use them on the underlying DataFrame containing the metadata.
        """
        return self()[key]

    def get(
        self,
        data_field_name: str,
        hdr: pd.DataFrame = None,
        ret_type=list,
        isolate_single: bool = False,
        *args,
        **kwargs,
    ) -> Union[list, dict, Any]:
        """Core method to retrieve data structures stored with the :py:meth:`Database.add_data_field` method.
        In practice, methods generated by the :py:meth:`Database.add_data_field` method will use this method.
        Note that the defaults used by the :py:meth:`Database.add_data_field` for `ret_type` is `dict` and `isolate_single` is `True`.

        Args:
            data_field_name (str): Name of the data field. Example - heart_rate
            hdr (pd.DataFrame, optional): A DataFrame containing the rows of interest. Typically the output of :py:meth:`Database.__call__`. Defaults to None.
            ret_type (type, optional): Either `list` or `dict`. For example, the former would return a list of heart rate data,
                and the latter would return a dictionary of {participant_id: participant_heart_rate_data} for the queried entries. Defaults to list.
            isolate_single (bool, optional): If the query results in only one data entry, then return just that data entry. Defaults to False.

        Returns:
            Union[list, dict, Any]: When `isolate_single` is set to `True`, then return type is Any because any data type can be stored in a data field.

        Example:
            Consider a database of motion capture data where the metadata contains values for cadence and percentage of preferred speed. ::

                hdr = db(cadence=160, speedp=100)
                db.get('ot', hdr)
                # OR, use the shorter version
                db.get('ot', cadence=160, speedp=100)
        """
        if not hasattr(self, f"_{data_field_name}"):
            print(
                f"{data_field_name} not found. Use db.add_data_field({data_field_name}, data)."
            )
            return

        if not getattr(self, f"_{data_field_name}"):  # if it is empty
            print(f"db._{data_field_name} is empty. Nothing to return.")
            return

        if isinstance(hdr, str):
            args = list(args) + [hdr]
            hdr = None  # revert to default

        if isinstance(ret_type, str):
            args = list(args) + [ret_type]
            ret_type = list  # revert to default
        assert ret_type in (list, dict)

        if isinstance(isolate_single, str):
            args = list(args) + [isolate_single]
            isolate_single = False  # revert to default

        if hdr is None:
            hdr = self(*args, **kwargs)

        field_data = getattr(self, f"_{data_field_name}")
        id_column_name = self.data_key_names[data_field_name]
        if ret_type == list:
            data = [field_data[k] for k in hdr[id_column_name] if k in field_data]
            if len(data) != len(hdr[id_column_name]):
                print(
                    f"WARNING: Missing values in {data_field_name}, use ret_type=dict to reduce errors."
                )
            if isolate_single and len(data) == 1:
                data = data[0]
        else:  # dictionary is more error-tolerant - returns data files only when they are present in that trial
            data = {k: field_data[k] for k in hdr[id_column_name] if k in field_data}
            if isolate_single and len(data) == 1:
                data = list(data.values())[0]

        return data

    def add_data_field(self, name: str, data: dict, data_key_name: str = "id"):
        """**Add a data field to the database.**

        Example:
            Consider the following example: ::

                db.add_data_field(name="heart_rate", data=heart_rate_data, data_key_name="participant_id")
                # retrieve heart rate data from participants between 40 and 60 years of age who have not had surgery.
                db.heart_rate(age_lim=(40,60), surgery_performed=False)

            This method will set `db._heart_rate = heart_rate_data`,
            and create a method `db.heart_rate` which can retrieve specific heart_rate_data entries
            based on queries related to the metadata stored in the header.
            See :py:meth:`Database.__call__` to learn more about query construction.

        Args:
            name (str): Name of the data field (e.g. heart_rate). Should not be present in the database, and it should not be 'records'
            data (dict): _description_
            data_key_name (str, optional): _description_. Defaults to 'id'.
        """
        assert not hasattr(self, f"_{name}")
        assert not hasattr(self, name)
        setattr(self, f"_{name}", data)
        setattr(
            self,
            name,
            lambda hdr=None, ret_type=dict, isolate_single=True, *args, **kwargs: self.get(
                data_field_name=name,
                hdr=hdr,
                ret_type=ret_type,
                isolate_single=isolate_single,
                *args,
                **kwargs,
            ),
        )
        self.data_fields.append(name)
        self.data_key_names[name] = data_key_name

    def records(
        self, hdr: pd.DataFrame = None, *args, **kwargs
    ) -> list[MutableMapping[Hashable, Any]]:
        """Returns records similar to `pandas.DataFrame.to_dict(orient='records')`.
        It will include all the entries from the fields in `Database.data_fields`.

        Args:
            hdr (pd.DataFrame, optional): A DataFrame containing the rows of interest. Typically the output of :py:meth:`Database.__call__`. Defaults to None.

        Returns:
            list[MutableMapping[Hashable, Any]]
        """
        if isinstance(hdr, str):
            args = list(args) + [hdr]
            hdr = None  # revert to default

        if hdr is None:
            hdr = self(*args, **kwargs).to_dict(orient="records")

        if isinstance(hdr, pd.DataFrame):
            hdr = hdr.to_dict(orient="records")

        for rec in hdr:
            for data_field_name, data_key_name in self.data_key_names.items():
                this_field_data = getattr(self, f"_{data_field_name}")
                if rec[data_key_name] in this_field_data:
                    rec[data_field_name] = this_field_data[rec[data_key_name]]
                else:
                    rec[data_field_name] = None
        return hdr


class DatabaseContainer:
    """Build hierarchical relationships between databases to enable flexible data retrieval based on metadata stored at multiple levels.
    Attributes:
        _db (dict) - database_name (str) : database (Database)
        _parents (dict) - stores the relationships between databases. It is a mapping from database_name : parent_name

    Example:
        dbc = DatabaseContainer()
        dbc.add("subject", subject_db) # add the top level database first
        dbc.add("trial", trial_db, "subject", lambda trial_id: trial_id[:2]) # last argument is a function that converts trial_id to subject_id

    Each database has columns (metadata) and data_fields.
    Each database is identified by a name <db_name>, and must have a column <db_name>_id
        (e.g. a database with name trial must have a column called trial_id)
    Within each container, there can only be one top-level database, and this should be added first.
    Each parent database can have multiple child databases, and each child in turn can be a parent to other databases.

    TODO:
        Make a plan for column names that conflict with special cases of keywords, i.e. <column_name>_lim / _has / _any.
    """

    def __init__(self) -> None:
        self._db: dict[str, Database] = {}
        self._parents: dict[str, str] = {}

    @property
    def all_db_names(self) -> list[str]:
        """List of database names in the container."""
        return list(self._db.keys())

    @property
    def all_column_names(self) -> list[str]:
        """Names of all columns in all databases."""
        return [column for db in self._db.values() for column in db().columns]

    @property
    def all_data_fields(self) -> list[str]:
        """Names of data fields in all databases.
        For example, "heart_rate", added through Database.add_data_field()
        """
        return [data_field for db in self._db.values() for data_field in db.data_fields]

    @property
    def _column_name_to_db_name(self) -> dict[str, str]:
        """A mapping to know which column belongs in which database."""
        ret = {}
        for db_name in self._db:
            for col_name in self._db[db_name]().columns:
                if col_name in [
                    x + "_id" for x in self._db
                ]:  # special cases of 'subject_id', 'trial_id', 'action_id' where the same column can be present in multiple databases
                    ret[col_name] = _removesuffix(col_name, "_id")
                else:
                    ret[col_name] = db_name
        return ret

    @property
    def _data_field_to_db_name(self) -> dict[str, str]:
        """A mapping to know which data field belongs in which database."""
        ret = {}
        for db_name, db in self._db.items():
            for data_field in db.data_fields:
                if data_field not in ret:
                    ret[data_field] = db_name
        return ret

    def get_db_name_of_column(self, column_name: str) -> str:
        """Get which database a column is in."""
        c2d = self._column_name_to_db_name
        assert column_name in c2d
        return c2d[column_name]

    def get_db_name_of_data_field(self, data_field: str) -> str:
        """Get the database name containing the data field."""
        assert data_field in self.all_data_fields
        return self._data_field_to_db_name[data_field]

    def get_heritage(self, db_name: str) -> list:
        """Return parent, grandparent, ...
        For example,
            "trial" -> ["subject"]
            "action" -> ["trial", "subject"]
        """
        assert db_name in self._parents
        ret = []
        current_parent = self._parents[db_name]
        while current_parent is not None:
            ret.append(current_parent)
            current_parent = self._parents[current_parent]
        return ret

    @property
    def _db_level(self) -> dict:
        """The level of a database is the number of ancestors it has.
        Top-level is 0.
        Return the level of each database within the container.
        """
        return {name: len(self.get_heritage(name)) for name in self._db}

    def add(
        self,
        child_name: str,
        db: Database,
        parent_name: str = None,
        child_to_parent_id: Callable = None,
    ) -> None:
        """Add a database to the container.

        Args:
            child_name (str): Name of the database inside the container. Use a singular word, e.g. "trial" instead of "trials"
            db (Database): The database to be added to the container.
            parent_name (str): Name of the parent in the container. Set this to None for the top level database (default).
            child_to_parent_id (Callable): A function that maps a row in the child to that in a parent.
                For example, if subject_id = (1,1), and trial_id = (1, 1, 4),
                child_to_parent_id = lambda trial_id: trial_id[:2]
        """
        assert child_name not in self._db

        assert f"{child_name}_id" in db().columns
        # A trial database MUST have trial_id

        overlapping_columns = {
            col_name: f"{child_name}_{col_name}"
            for col_name in db().columns
            if col_name in self.all_column_names
        }
        if len(overlapping_columns) > 0:
            print(
                f"WARNING: Found common column names across databases. Renaming columns:"
            )
            print(overlapping_columns)
        db._data = db._data.rename(columns=overlapping_columns)

        overlapping_data_fields = [
            data_field
            for data_field in db.data_fields
            if data_field in self.all_data_fields
        ]
        if len(overlapping_data_fields) > 0:
            print("Found overlapping data field names across databases.")
            print(overlapping_data_fields)

        if parent_name is not None:
            assert parent_name in self._parents
            assert parent_name in self._db
            assert child_to_parent_id is not None
        else:
            assert len(self._db) == 0  # only one top level database

        self._db[child_name] = db
        self._parents[child_name] = parent_name

        # add the column that maps every row in the child dataframe to a row in the parent dataframe
        for current_parent in self.get_heritage(child_name):
            child_df = db()
            child_df[f"{current_parent}_id"] = child_df[f"{child_name}_id"].apply(
                child_to_parent_id
            )

    def __getattr__(self, key: str) -> Callable | Database | pd.Series:
        """Flexible retrieval. Most useful when working with data fields.
        Checks data fields, database names, and column names, in that order

        1. Provides direct access to data fields. For example, instead of
            `dbc.trials.heart_rate(subject_id=(1,2))`
            it is possible to use
            `dbc.heart_rate(subject_id=(1,2))`

        2. If the key corresponds to database name e.g. "trials",
            then return the database.

        3. If the key corresponds to metadata (column) name,
            then return that columns from the dataframe.

        Returns:
            Callable | Database | pd.Series: Callable for data_field, Database for db_name, and pd.Series for column_name
        """
        if key in self.all_data_fields:
            # for doing dbc.ot('expert') instead of dbc('expert', data_field='ot')
            return functools.partial(self.__call__, data_field=key)

        if key in self._db:
            return self._db[key]

        if key in self.all_column_names:
            return self._db[self.get_db_name_of_column(key)]()[key]

        return None

    def _cast_column_to_db(self, column_name: str, db_name: str = None) -> pd.Series:
        """Cast a column into a different database.
        For example, if the "height" column is in the database "subject",
        and we want the "height" column for every "trial", use this method
        to cast height from subject to trial.
        """
        column_db_name = self.get_db_name_of_column(column_name)
        column = self._db[column_db_name]()[column_name]

        if db_name is None:  # default behavior
            return column

        assert db_name in self._db

        assert self._db_level[column_db_name] < self._db_level[db_name]
        # for example, to get experiment (attr_level_idx) at the level of trial (level)

        out_level_id = self._db[db_name]()[f"{column_db_name}_id"]
        # for example, self.trials()['subject_id']

        return out_level_id.map(column.to_dict())

    def help(self, key=None):
        if key is None:
            for db_name, db in self._db.items():
                print(f"Query params at db_name={db_name}")
                print([x for x in db().columns if not (x.endswith("_id") or x == "id")])
                print()
            print("These are the data fields (modalities) present at each level: ")
            print({db_name: db.data_fields for db_name, db in self._db.items()})
            return

        if isinstance(key, str) and key in self._column_name_to_db_name:
            print(
                f"{key} is an attribute at the {self.get_db_name_of_column(key)} level."
            )
            print(f"Unique values of {key} are:")
            print(np.unique(list(getattr(self, key))))

    def __call__(self, *args, **kwargs):
        """
        Generalize data search and retrieval across databases created at different levels - e.g. subject, trial, action
        create a temporary database to execute a search from attributes across multiple levels

        See class docstring for examples.
        """
        if "level" in kwargs:
            # for backwards compatibility when level meant db_name, which made sense when there was only one database per level
            # Old usage: dbc("expert", level="trial")
            # Recommended usage: dbc("expert", db_name="trial")
            assert isinstance(kwargs["level"], str) and kwargs["level"] in self._db
            assert db_name not in kwargs
            kwargs["db_name"] = kwargs.pop("level")

        db_name = kwargs.pop("db_name", None)  # in database names
        data_field = kwargs.pop("data_field", None)  # in data fields
        return_records = kwargs.pop("return_records", False)

        db_level = 0
        if db_name is not None:
            assert db_name in self._db
            db_level = self._db_level[db_name]

        data_field_level = 0
        if data_field is not None:
            assert isinstance(data_field, str)
            data_field_level = self._db_level[
                self.get_db_name_of_data_field(data_field)
            ]

        column_list = []
        all_column_names_modified = {
            item: c
            for c in self.all_column_names
            for item in (c + "_lim", c + "_has", c + "_any")
        }
        for inp in list(args) + list(kwargs.keys()):
            if inp in self.all_column_names:
                column_list.append(inp)
            elif inp in all_column_names_modified:
                column_list.append(all_column_names_modified[inp])

        column_level_list = [
            self._db_level[self._column_name_to_db_name[col]] for col in column_list
        ]

        query_level_num = max(db_level, *column_level_list, data_field_level)

        query_level_name = self.all_db_names[query_level_num]
        query_level_db = self._db[query_level_name]
        df = query_level_db().copy(deep=False)
        for column_name, column_level in zip(column_list, column_level_list):
            if column_level < query_level_num:
                df[column_name] = self._cast_column_to_db(
                    column_name, db_name=self.all_db_names[query_level_num]
                )
        
        db = Database(df)
        df_queried = db(*args, **kwargs)

        if return_records:
            return query_level_db.records(hdr=df_queried)

        if data_field is None:
            return df_queried

        assert data_field in query_level_db.data_fields
        return query_level_db.get(
            data_field_name=data_field,
            hdr=df_queried,
            ret_type=dict,
            isolate_single=True,
        )

    def records(self, *args, **kwargs):
        return self.__call__(*args, **kwargs, return_records=True)


def _removesuffix(s: str, suffix: Union[str, list]) -> str:
    """Remove specified suffix in a string s.
    For python 3.7 and 3.8 compatibility.
    Note that this works for use cases in this module,
    and not in generally for all strings and suffixes.

    Args:
        s (str): e.g. "trial_lim"
        suffix (str): e.g. "_lim"

    Returns:
        str: e.g. "trial"
    """
    if isinstance(suffix, str):
        suffix = [suffix]
    return functools.reduce(
        lambda ts, tsuffix: re.sub(f"\{tsuffix}$", "", ts), [s] + suffix
    )


def get_example_database() -> Database:
    """Generate an example database.

    Returns:
        Database
    """
    # code generated with chat-GPT 3.5, then modified
    # Set seed for reproducibility
    np.random.seed(42)

    # Generate data for each column
    participant_ids = np.arange(1, 101)
    ages = np.random.uniform(20, 80, size=100)
    surgery_performed = np.random.choice([True, False], size=100)
    notes_choice = [""] * 4 + [
        "HRV is interesting",
        "QRS complex is interesting",
        "review data",
    ]
    notes = np.random.choice(notes_choice, size=100)

    # Create DataFrame
    data = pd.DataFrame(
        {
            "participant_id": participant_ids,
            "age": ages,
            "surgery_performed": surgery_performed,
            "notes": notes,
        }
    )
    ret = Database(data)
    return ret


def get_example_data() -> MutableMapping[int, Any]:
    """Get example data for adding to a data field.

    Example:
        ::

            db = datanest.get_example_database()
            db.add_data_field('heart_rate', datanest.get_example_data(), 'participant_id')
            db.heart_rate(age_lim=(40,50), surgery_performed=False)

    Returns:
        MutableMapping[int, Any]: Fake time and heart rate values encapsulated in a python object
    """
    np.random.seed(42)
    participant_ids = np.arange(1, 101)
    hr_means = np.random.uniform(50, 120, size=100)
    hr_variances = np.random.uniform(5, 40, size=100)

    class HRData:
        def __init__(self, time, hr):
            assert len(time) == len(hr)
            self.time = time
            self.hr = hr

    ret = {}
    time_minutes = np.arange(1, 201)
    for participant_id, hr_mean, hr_variance in zip(
        participant_ids, hr_means, hr_variances
    ):
        hr_values = np.random.randn(200) * hr_variance + hr_mean
        ret[participant_id] = HRData(time_minutes, hr_values)

    # create a missing entry for participant 23
    del ret[23]

    return ret
