# datanest

[![PyPI - Version](https://img.shields.io/pypi/v/datanest.svg)](https://pypi.org/project/datanest)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/datanest.svg)](https://pypi.org/project/datanest)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/praneethnamburi/datanest/main/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/datanest/badge/?version=latest)](https://datanest.readthedocs.io)

Encapsulate pandas DataFrame and arbitratry python objects, such as time series data.

-----

## Installation

```console
pip install datanest
```

## Usage

`datanest.Database` is the core class that wraps a `pandas.DataFrame` object. Even before adding any data fields using the `add_data_field` method, the database can already be used to query rows from the encapsulated DataFrame with an intuitive keyword argument syntax.

```python
import datanest

# Load example DataFrame with columns: 
# participant_id (int), age (float), surgery_performed (bool), notes (str)
db = datanest.get_example_database()

# Retrieve all metadata
db()

# Retrieve metadata for participant 3
db(participant_id=3)

# Retrieve metadata for participants aged 50 to 60 who have not had surgery
db(age_lim=(50, 60), surgery_performed=True)

# Retrieve metadata for participants where the notes string contains the word interesting
db(notes_has='interesting')
```

The `add_data_field` method can be used to add arbitrary python objects to the database, and we can retrieve relevant data entries using the same keyword argument syntax.

```python
# Add heart rate data to the database, indexed by participant_id
db.add_data_field('heart_rate', datanest.get_example_data(), 'participant_id')

# Retrieve all heart rate time series data
db.heart_rate()

# Retrieve heart rate time series data for participant 3
db.heart_rate(participant_id=3)

# Retrieve heart rate time series for participants aged 50 to 60
db.heart_rate(age_lim=(50, 60))

# Retrieve heart rate time series for participants where the notes string contains the word interesting
db.heart_rate(notes_has='interesting')
```

## License

`datanest` is distributed under the terms of the [MIT license](LICENSE).

## Contact

[Praneeth Namburi](https://praneethnamburi.com)

Project Link: [https://github.com/praneethnamburi/datanest](https://github.com/praneethnamburi/datanest)


## Acknowledgments

This tool was developed as part of the ImmersionToolbox initiative at the [MIT.nano Immersion Lab](https://immersion.mit.edu). Thanks to NCSOFT for supporting this initiative.
