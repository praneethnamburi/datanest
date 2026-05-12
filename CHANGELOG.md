# Change Log
All notable changes to this project will be documented in this file.

## [1.1.0] - 2026-05-12

### Added
- `DatabaseContainer` — manage hierarchical relationships between `Database` instances (e.g. subject → trial → action) with key-derivation lambdas linking levels. Supports `add`, `add_data_field` (via the contained `Database` instances), the same `_lim` / `_has` / `_any` keyword-suffix query syntax as `Database`, `records`, `help`, and parent/child column casting (`_cast_column_to_db`).
- `DatabaseContainer` is now exported (`from datanest import DatabaseContainer`).
- `cache_me_if_you_can` / `cache_me_if_you_can_incremental` — dill-backed file-cache decorators, migrated from `immersionToolbox/immersionlab/__init__.py`. The `append_str: str` (eval'd at call time) parameter is replaced by `suffix: Callable[..., str]`; pn-projects callers updated in the same release.
- GitHub Actions CI (`.github/workflows/test.yml`) — pytest matrix across Python 3.8–3.12 on Ubuntu 22.04 / macOS latest / Windows latest, triggered on push and PRs.
- `[project.optional-dependencies] test` group (`pip install -e ".[test]"`) — installs `pytest` and `openpyxl` (the latter exercises `Database`'s Excel I/O path in the test suite without bloating the runtime dep list).
- README "Payload types — anything goes" section documenting the modality-agnostic design: payloads can be any Python object, and `pysampled.Data` is a *use convention* rather than a hard dependency.

### Dependencies
- `dill` is now a required dependency (used by the new `cache_me_if_you_can*` decorators).
- Python floor raised from 3.7 to 3.8 (3.7 is EOL and unsupported by the new CI matrix).

### Tests
- `test_database_container.py` — `_make_synthetic_signals` now returns plain `np.ndarray` payloads instead of `pysampled.Data`. The two affected tests exercise `add_data_field`'s modality-agnostic contract (key→object dict), which the payload type does not influence; removing the `pysampled` test-only import aligns the suite with datanest's design philosophy and drops a test-only dep.

### Notes
- `DatabaseContainer`: this is its new home — previously implemented in `immersionToolbox/immersionlab/__init__.py`. The legacy classes there become a re-export shim of the `datanest` originals (mirroring the `immersionlab/delsys.py` pattern); existing `immersionlab.DatabaseContainer` subclasses continue to work unchanged.
- `cache_me_if_you_can*`: **no re-export shim** is left in `immersionlab` (in contrast with `DatabaseContainer`). pn-projects callers were flipped to import directly from `datanest` in lockstep with the migration; `immersionlab.cache_me_if_you_can` no longer exists.

### Deferred to 1.2.0
- `break_signals_into_actions` stays in `immersionlab` (couples to pysampled / event semantics — does not belong in datanest's modality-agnostic core). `immersionlab.DatabaseContainer` keeps the method via a thin subclass.
- Kwarg-suffix collision detection (`_lim` / `_has` / `_any` reservations vs. real column names) — TODO at `datanest/__init__.py` `DatabaseContainer` docstring.


## [1.0.0] - 2024-02-04

First major release after thorough testing, 100% coverage, and formatting.
