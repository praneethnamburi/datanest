# datanest

[![src](https://img.shields.io/badge/src-github-blue)](https://github.com/praneethnamburi/datanest)
[![PyPI - Version](https://img.shields.io/pypi/v/datanest.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/datanest/)
[![Documentation Status](https://readthedocs.org/projects/datanest/badge/?version=latest)](https://datanest.readthedocs.io)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/praneethnamburi/datanest/main/LICENSE)

*A simple, intuitive, pandas-based database.*

Perfect for handling data such as time series, images, or any
Python objects alongside their metadata. This tool encapsulates a pandas
DataFrame containing metadata and Python objects. It provides an intuitive
data and metadata retrieval syntax through keyword-arguments.

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

### Payload types — anything goes

The value side of `add_data_field` is unconstrained: datanest does not
inspect the objects you attach, only the keys that index them. So a
payload dict can hold NumPy arrays, images, custom dataclasses,
[`pysampled.Data`][pysampled] signals — anything that makes sense for
your project.

```python
import numpy as np

db = datanest.get_example_database()
db.add_data_field(
    "eeg",
    {pid: np.random.randn(1000) for pid in db()["participant_id"]},
    "participant_id",
)
db.eeg(age_lim=(50, 60))   # {participant_id: ndarray} for the matching rows
```

In the lab where datanest originated, time-series payloads are
typically [`pysampled.Data`][pysampled] objects, but this is a *use
convention*, not a hard dependency — `datanest` does not import
`pysampled` and works with whatever payload type you choose.

[pysampled]: https://github.com/praneethnamburi/pysampled

### Hierarchical data: `DatabaseContainer`

When metadata lives at multiple levels (e.g. *subject* → *trial* → *action*), wrap a set of `Database` instances in a `DatabaseContainer`. Each level is added with a key-derivation function that maps a child id to its parent id. The container provides the same keyword-argument query syntax as `Database`, resolving the right level automatically.

```python
import datanest

dbc = datanest.DatabaseContainer()
dbc.add("subject", subject_db)
dbc.add("trial", trial_db, "subject", lambda trial_id: trial_id[:2])
dbc.add("action", action_db, "trial", lambda action_id: action_id[:3])

# Query at any level — subject metadata filters trials and actions too
dbc(subject=3)                       # all subject-3 trials/actions
dbc(action_phase='extension')        # subset of action rows
dbc.heart_rate(age_lim=(50, 60))     # data field added at any level
```

`DatabaseContainer` uses the same `_lim` / `_has` / `_any` suffix conventions as `Database`. Add data fields to the child databases directly (`trial_db.add_data_field(...)`); the container makes them queryable at any level via `dbc.<field>(...)`.

### Caching expensive computations

`datanest.cache_me_if_you_can` and `cache_me_if_you_can_incremental` are
dill-backed file-cache decorators. Use them to skip recomputation when
loading or summarizing data fields is expensive. Both accept an optional
`suffix` callable that receives the wrapped function's `(*args, **kwargs)`
and returns a string inserted between the cache file's stem and extension —
handy for one cache file per input.

```python
from datanest import cache_me_if_you_can, cache_me_if_you_can_incremental

# One cache file per subject (e.g. heart_rate_s03.pkl)
@cache_me_if_you_can("heart_rate.pkl", suffix=lambda subject_id: f"_s{subject_id:02d}")
def load_heart_rate(subject_id):
    return expensive_load(subject_id)            # runs once per subject_id

# Build up a {trial_id: metrics} dict across many calls, persisting on disk
@cache_me_if_you_can_incremental("trial_metrics.pkl", return_name="ret", return_default={})
def summarize_trial(trial_id, ret=None):
    if trial_id not in ret:
        ret[trial_id] = compute_metrics(trial_id)
    return ret
```

## License

`datanest` is distributed under the terms of the [MIT license](LICENSE).

## Contact

[Praneeth Namburi](https://praneethnamburi.com)

Project Link: [https://github.com/praneethnamburi/datanest](https://github.com/praneethnamburi/datanest)


## Acknowledgments

This tool was developed as part of the ImmersionToolbox initiative at the [MIT.nano Immersion Lab](https://immersion.mit.edu). Thanks to NCSOFT for supporting this initiative.
