# Change Log
All notable changes to this project will be documented in this file.

## [1.1.0] - 2026-05-12

### Added
- `DatabaseContainer` — manage hierarchical relationships between `Database` instances (e.g. subject → trial → action) with key-derivation lambdas linking levels. Supports `add`, `add_data_field` (via the contained `Database` instances), the same `_lim` / `_has` / `_any` keyword-suffix query syntax as `Database`, `records`, `help`, and parent/child column casting (`_cast_column_to_db`).
- `DatabaseContainer` is now exported (`from datanest import DatabaseContainer`).

### Notes
- This is the new home for `DatabaseContainer` — previously implemented in `immersionToolbox/immersionlab/__init__.py`. The legacy classes there become a re-export shim of the `datanest` originals (mirroring the `immersionlab/delsys.py` pattern); existing `immersionlab.DatabaseContainer` subclasses continue to work unchanged.

### Deferred to 1.2.0
- `break_signals_into_actions` stays in `immersionlab` (couples to pysampled / event semantics — does not belong in datanest's modality-agnostic core). `immersionlab.DatabaseContainer` keeps the method via a thin subclass.
- Kwarg-suffix collision detection (`_lim` / `_has` / `_any` reservations vs. real column names) — TODO at `datanest/__init__.py` `DatabaseContainer` docstring.
- `cache_me_if_you_can` / `cache_me_if_you_can_incremental` migration from `immersionlab`.


## [1.0.0] - 2024-02-04

First major release after thorough testing, 100% coverage, and formatting.
