"""Encapsulate pandas DataFrame and arbitrary python objects.

:py:class:`Database` class initializes a Database that encapsulates a pandas DataFrame containing metadata, and data such as time series or images.
:py:meth:`Database.add_data_field` is used to add a new dictionary mapping a metadata column in the DataFrame to arbitrary python objects.
:py:meth:`Database.__call__` is used to retrieve metadata based on criteria that can be specified in a python keyword arguments format.
"""
# SPDX-FileCopyrightText: 2024-present Praneeth Namburi <praneeth.namburi@gmail.com>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Union, MutableMapping, Hashable, Any

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["Database"]


class Database:
    """Manipulate data (signals/images/arbitrary Python objects) and their metadata stored in a CSV/pandas DataFrame.

    A `Database` instance encapsulates a pandas DataFrame, and is designed for use with dictionary-like storage of data structures, such as time series data.
    It facilitates working with data (e.g., time series/images, etc.) alongside the metadata stored in a pandas DataFrame.
    This tool provides an intuitive and flexible way to retrieve relevant data from data structures based on metadata organized in the DataFrame.
    It was developed to address performance issues when storing arbitrary data objects, including NumPy arrays, in a pandas DataFrame.

    Args:
        data (Union[pd.DataFrame, str]):
            - (str) Path to a CSV or Excel file. It is read as a pandas DataFrame.
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

    def __init__(self, data: Union[pd.DataFrame, str] = None):
        if data is None:
            self._data = pd.DataFrame()
        elif isinstance(data, pd.DataFrame):
            self._data = data
        elif isinstance(data, str):
            fname = data
            assert os.path.exists(fname)
            assert Path(fname).suffix in (".xls", ".xlsx", ".csv")
            if Path(fname).suffix == ".csv":
                loader = pd.read_csv
            else:
                loader = pd.read_excel
            self._data = loader(fname)

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

        def removesuffix(s, suffix):  # for python 3.7 and 3.8
            return re.sub(f"\{suffix}$", "", s)

        df = self.get_df()
        sel = []
        for k in args:
            if k in df and df[k].dtype == bool:
                sel.append(df[k])

        for k, v in kwargs.items():
            if k.endswith("_lim") and (removesuffix(k, "_lim") in df):
                assert len(v) == 2
                tk = removesuffix(k, "_lim")
                sel.append(df[tk] >= v[0])
                sel.append(df[tk] <= v[1])
            elif k.endswith("_any") and (removesuffix(k, "_any") in df):
                tk = removesuffix(k, "_any")
                this_sel = []
                for this_val in v:
                    this_sel.append(df[tk] == this_val)
                sel.append(pd.Series(np.logical_or.reduce(this_sel)))
            elif k.endswith("_has") and (
                removesuffix(k, "_has") in df
            ):  # e.g. notes_has='eyes closed'
                tk = removesuffix(k, "_has")
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
        id_column_name: str = "id",
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
            id_column_name (str, optional): Column name in the metadata DataFrame used to map the entries in `data_field_name`. Defaults to 'id'.

        Returns:
            Union[list, dict, Any]: When `isolate_single` is set to `True`, then return type is Any because any data type can be stored in a data field.

        Example:
            Consider a database of motion capture data where the metadata contains values for cadence and percentage of preferred speed. ::

                hdr = db(cadence=160, speedp=100)
                db.get('ot', hdr)
                # OR, use the shorter version
                db.get('ot', cadence=160, speedp=100)
        """
        assert ret_type in (list, dict)
        if not hasattr(self, f"_{data_field_name}"):
            print(
                f"Data not loaded into memory. Use db.add_data_field({data_field_name}, data) to add the {data_field_name} data field."
            )
            return
        if not getattr(self, f"_{data_field_name}"):  # if it is empty
            print(f"db._{data_field_name} is empty. Nothing to return.")
            return
        if hdr is None:
            hdr = self(*args, **kwargs)
        elif isinstance(hdr, (list, tuple)):
            assert len(hdr) == 2
            hdr = self(id=hdr)
        if ret_type == list:
            data = [
                getattr(self, f"_{data_field_name}")[k] for k in hdr[id_column_name]
            ]
            if isolate_single and len(data) == 1:
                data = data[0]
        else:  # dictionary is more error-tolerant - returns data files only when they are present in that trial
            data = {
                k: getattr(self, f"_{data_field_name}")[k]
                for k in hdr[id_column_name]
                if k in getattr(self, f"_{data_field_name}")
            }
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
            name (str): Name of the data field (e.g. heart_rate).
            data (dict): _description_
            data_key_name (str, optional): _description_. Defaults to 'id'.
        """
        assert not hasattr(self, f"_{name}")
        assert not hasattr(self, name)
        setattr(self, f"_{name}", data)
        setattr(
            self,
            name,
            lambda hdr=None, *args, **kwargs: self.get(
                data_field_name=name,
                hdr=hdr,
                ret_type=dict,
                isolate_single=True,
                id_column_name=data_key_name,
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
        if hdr is None:
            hdr = self(*args, **kwargs).to_dict(orient="records")
        if isinstance(hdr, pd.DataFrame):
            hdr = hdr.to_dict(orient="records")
        for rec in hdr:
            for mod, data_key_name in self.data_key_names.items():
                this_modality_data = getattr(self, f"_{mod}")
                if rec[data_key_name] in this_modality_data:
                    rec[mod] = this_modality_data[rec[data_key_name]]
                else:
                    rec[mod] = None
        return hdr


def get_example_database(has_data_field=False) -> Database:
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
    notes_choice = (
        "",
        "",
        "",
        "",
        "HRV is interesting",
        "QRS complex is interesting",
        "review data",
    )
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
    if has_data_field:
        ret.add_data_field("heart_rate", get_example_data(), "participant_id")
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

    return ret
