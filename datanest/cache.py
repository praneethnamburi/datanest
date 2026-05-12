"""dill-backed file-cache decorators.

Two decorators that skip recomputation by storing the wrapped
function's return value on disk:

- :py:func:`cache_me_if_you_can` — bypass the wrapped function entirely
  if its cache file already exists.
- :py:func:`cache_me_if_you_can_incremental` — accumulate results across
  calls (e.g. building up a per-trial dictionary one trial at a time).

Both vary the cache filename per call via an optional ``suffix``
callable, which receives the wrapped function's ``(*args, **kwargs)``
and returns a string inserted between the file's stem and extension.
"""
# SPDX-FileCopyrightText: 2024-present Praneeth Namburi <praneeth.namburi@gmail.com>
#
# SPDX-License-Identifier: MIT

import copy
import os
from pathlib import Path
from typing import Any, Callable, Union

import dill


def cache_me_if_you_can(
        cache_fname: Union[str, Path],
        *,
        suffix: Union[Callable[..., str], None] = None,
        verbose: bool = False,
        ):
    """Decorator: skip recomputation if cache_fname exists on disk.

    Args:
        cache_fname: Path to the cache file (dill-serialized).
        suffix: Optional callable invoked with the wrapped function's
            ``(*args, **kwargs)`` at call time; its return value is
            inserted between the file's stem and its extension. Use
            this to vary the cache file by call-time inputs.
        verbose: If True, print create/load messages.

    Example:
        >>> @cache_me_if_you_can("results.pkl", suffix=lambda *a, **kw: "_" + a[0])
        ... def expensive(tag):
        ...     return heavy_computation(tag)
        >>> expensive("alpha")  # computes, writes results_alpha.pkl
        >>> expensive("alpha")  # loads from results_alpha.pkl
        >>> expensive("beta")   # computes, writes results_beta.pkl
    """
    def wrapper(func):
        def inner_func(*args, **kwargs):
            extra = suffix(*args, **kwargs) if suffix is not None else ''
            p = Path(cache_fname)
            cache_path = str(p.with_name(p.stem + extra + p.suffix))
            if not os.path.exists(cache_path):
                if verbose:
                    print(f"{func.__qualname__} is creating {cache_path}")
                ret = func(*args, **kwargs)
                with open(cache_path, 'wb') as f:
                    dill.dump(ret, f)
                return ret
            with open(cache_path, 'rb') as f:
                if verbose:
                    print(f"{func.__qualname__} is loading data from {cache_path}")
                ret = dill.load(f)
            return ret
        return inner_func
    return wrapper


def cache_me_if_you_can_incremental(
        cache_fname: Union[str, Path],
        return_name: str,
        return_default: Any,
        *,
        suffix: Union[Callable[..., str], None] = None,
        verbose: bool = False,
        force_save: bool = False,
        ):
    """Decorator: incrementally accumulate results in a dill-cached object.

    Save the current state, and avoid repeating computations (e.g. when
    adding files to a database one after the other and extracting
    metrics from them). The wrapped function receives the running
    accumulator via the keyword named in ``return_name``; it returns
    the (mutated) accumulator.

    Args:
        cache_fname: Path to the cache file (dill-serialized).
        return_name: Name of the keyword argument injected into the
            wrapped function carrying the running accumulator.
        return_default: Initial accumulator value when no cache exists.
        suffix: Optional callable invoked with the wrapped function's
            ``(*args, **kwargs)`` at call time; its return value is
            inserted between the file's stem and its extension.
        verbose: If True, print create/add messages.
        force_save: If True, always rewrite the cache file even when
            the accumulator dict's key set is unchanged.

    Example:
        >>> @cache_me_if_you_can_incremental(
        ...     "trials.pkl", return_name="ret", return_default={})
        ... def process(trial_id, ret=None):
        ...     if trial_id not in ret:
        ...         ret[trial_id] = expensive_compute(trial_id)
        ...     return ret
    """
    def wrapper(func):
        def inner_func(*args, **kwargs):
            extra = suffix(*args, **kwargs) if suffix is not None else ''
            p = Path(cache_fname)
            cache_path = str(p.with_name(p.stem + extra + p.suffix))

            if not os.path.exists(cache_path):
                if verbose:
                    print(f"{func.__qualname__} will create {cache_path}")
                current_return = copy.copy(return_default)
            else:
                if verbose:
                    print(f"{func.__qualname__} will add to {cache_path}")
                with open(cache_path, 'rb') as f:
                    current_return = dill.load(f)

            if isinstance(current_return, dict):
                current_keys = set(current_return.keys())

            ret = func(*args, **{**kwargs, **{return_name: current_return}})
            if not force_save:
                if isinstance(current_return, dict) and current_keys == set(ret.keys()):
                    return ret

            # try to save only if there is a change, or if force_save is True
            with open(cache_path, 'wb') as f:
                dill.dump(ret, f)
            return ret

        return inner_func
    return wrapper
