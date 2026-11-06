# Copyright © 2026 Pathway

from __future__ import annotations

import abc
import functools
import inspect
import os
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ParamSpec, TypeVar, overload

import async_lru
import diskcache

from pathway.internals import api, trace
from pathway.internals.runtime_type_check import check_arg_types

T = TypeVar("T")
P = ParamSpec("P")


@dataclass(frozen=True)
class CacheStats:
    """Read-only, atomic snapshot of a named UDF cache statistics.

    The snapshot never contains UDF arguments or return values: only integer
    counters are recorded.

    Attributes:
        name: explicit name of the cache.
        hits: number of cache lookups served from the cache. Every completed
            lookup is counted exactly once, either as a hit or as a miss.
        misses: number of cache lookups that did not find an entry and caused
            the UDF to be called. Exceptions raised by the UDF are never cached,
            so a repeated call with the same arguments is counted as another
            miss.
        write_failures: number of failed attempts to store a UDF result in the
            cache (for example, when a value exceeds the disk cache size limit).
            The exception raised by the failed write is propagated to the
            caller. Always 0 for in-memory caches.
        entries: number of entries kept in the cache at the moment of taking
            the snapshot. Resetting the counters does not change this value.
    """

    name: str
    hits: int
    misses: int
    write_failures: int
    entries: int


class _CacheCounters:
    """Thread-safe integer counters of a single cache.

    No arguments or return values of the cached UDF are ever stored here.

    Counting model: every cache lookup increments ``_lookups``; a lookup that
    makes the caching engine call the underlying UDF additionally increments
    ``_misses``. Hence ``hits = lookups - misses`` at every point in time once
    the in-flight lookup reached the caching engine.
    """

    __slots__ = ("_lock", "_lookups", "_misses", "_write_failures")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lookups = 0
        self._misses = 0
        self._write_failures = 0

    def record_lookup(self) -> None:
        with self._lock:
            self._lookups += 1

    def record_miss(self) -> None:
        """Record a lookup that misses (atomic lookup + miss accounting)."""
        with self._lock:
            self._lookups += 1
            self._misses += 1

    def record_backend_miss(self) -> None:
        """Record the caching engine calling the UDF (a miss decided elsewhere)."""
        with self._lock:
            self._misses += 1

    def record_write_failure(self) -> None:
        with self._lock:
            self._write_failures += 1

    def snapshot(self, name: str, entry_count: Callable[[], int]) -> CacheStats:
        with self._lock:
            lookups = self._lookups
            misses = self._misses
            write_failures = self._write_failures
            # Reading the live entry count under the same lock makes the whole
            # snapshot an atomic point-in-time observation.
            entries = entry_count()
        return CacheStats(
            name=name,
            hits=lookups - misses,
            misses=misses,
            write_failures=write_failures,
            entries=entries,
        )

    def reset(self) -> None:
        with self._lock:
            self._lookups = 0
            self._misses = 0
            self._write_failures = 0


_NAMED_CACHES: dict[str, CacheStrategy] = {}
_NAMED_CACHES_LOCK = threading.Lock()


def _register_named_cache(name: str, cache_strategy: CacheStrategy) -> None:
    with _NAMED_CACHES_LOCK:
        if name in _NAMED_CACHES:
            raise ValueError(f"cache name `{name}` used more than once")
        _NAMED_CACHES[name] = cache_strategy


def get_cache_stats() -> dict[str, CacheStats]:
    """Return atomic statistics snapshots of all explicitly named UDF caches.

    Only caches created with an explicit ``name`` argument (``DiskCache`` and
    ``InMemoryCache``) are reported. The snapshots contain integer counters
    only; UDF arguments and return values are never recorded.

    Returns:
        Mapping from cache name to its :class:`CacheStats` snapshot.
    """
    with _NAMED_CACHES_LOCK:
        named_caches = list(_NAMED_CACHES.items())
    return {
        name: cache_strategy.get_stats()
        for name, cache_strategy in sorted(named_caches)
    }


def reset_cache_stats(name: str | None = None) -> None:
    """Reset the statistics counters of UDF caches.

    Only the counters (hits, misses, write failures) are reset. The entries
    stored in the caches are not removed.

    Args:
        name: Name of the cache whose counters are reset. When set to ``None``
            (the default), the counters of every named cache are reset.
    """
    with _NAMED_CACHES_LOCK:
        if name is None:
            cache_strategies = list(_NAMED_CACHES.values())
        else:
            try:
                cache_strategies = [_NAMED_CACHES[name]]
            except KeyError:
                raise ValueError(f"no named cache `{name}`") from None
    for cache_strategy in cache_strategies:
        cache_strategy.reset_stats()


class CacheStrategy(abc.ABC):
    """Base class used to represent caching strategy."""

    _name: str | None
    _stats: _CacheCounters | None

    def __init__(self, name: str | None = None) -> None:
        self._name = name
        if name is None:
            self._stats = None
        else:
            self._stats = _CacheCounters()
            _register_named_cache(name, self)

    @abc.abstractmethod
    def wrap_async(
        self, func: Callable[P, Awaitable[T]]
    ) -> Callable[P, Awaitable[T]]: ...

    @abc.abstractmethod
    def wrap_sync(self, func: Callable[P, T]) -> Callable[P, T]: ...

    @abc.abstractmethod
    def _entry_count(self) -> int: ...

    def get_stats(self) -> CacheStats:
        """Return an atomic snapshot of the cache statistics.

        Only caches created with an explicit ``name`` keep statistics.

        Raises:
            ValueError: when the cache has no explicit name.
        """
        if self._stats is None:
            raise ValueError(
                "cache statistics are only kept for caches with an explicit name"
            )
        assert self._name is not None
        return self._stats.snapshot(self._name, self._entry_count)

    def reset_stats(self) -> None:
        """Reset the hits, misses and write failure counters of this cache.

        The entries stored in the cache are not removed.

        Raises:
            ValueError: when the cache has no explicit name.
        """
        if self._stats is None:
            raise ValueError(
                "cache statistics are only kept for caches with an explicit name"
            )
        self._stats.reset()


class DiskCache(CacheStrategy):
    """On disk cache."""

    _cache: diskcache.Cache | None
    _name: str | None
    _size_limit: int

    @trace.trace_user_frame
    def __init__(self, name: str | None = None, size_limit=2**30) -> None:
        """
        Args:
            name: name of the cache. When multiple caches have the same name, they share a storage.
                Only a cache with an explicit name keeps statistics available through
                ``get_cache_stats`` or ``get_stats``.
            size_limit: a memory limit of the cache in bytes.
        """
        super().__init__(name)
        self._cache = None
        self._size_limit = size_limit

    def make_key(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        return str(api.ref_scalar(args, tuple(kwargs.items())))

    def wrap_async(self, func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            cache = self._get_cache(func)
            key = self.make_key(args, kwargs)
            if cache is None:
                # The cache is disabled: the call bypasses the cache and is not
                # counted as either a hit or a miss.
                return await func(*args, **kwargs)
            if key in cache:
                if self._stats is not None:
                    self._stats.record_lookup()
                return cache[key]
            if self._stats is not None:
                self._stats.record_miss()
            result = await func(*args, **kwargs)
            return self._store(cache, key, result)

        return wrapper

    def wrap_sync(self, func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            cache = self._get_cache(func)
            key = self.make_key(args, kwargs)
            if cache is None:
                # The cache is disabled: the call bypasses the cache and is not
                # counted as either a hit or a miss.
                return func(*args, **kwargs)
            if key in cache:
                if self._stats is not None:
                    self._stats.record_lookup()
                return cache[key]
            if self._stats is not None:
                self._stats.record_miss()
            result = func(*args, **kwargs)
            return self._store(cache, key, result)

        return wrapper

    def _store(self, cache: diskcache.Cache, key: str, result: Any) -> Any:
        try:
            cache[key] = result
            # Read the value back: depending on the diskcache version, an
            # entry larger than the size limit is either rejected while being
            # written (KeyError from the assignment) or silently evicted
            # inside the write transaction (the following lookup misses).
            # Both outcomes are write failures.
            return cache[key]
        except Exception:
            # A failed write is counted and then propagated; nothing is
            # cached, so a retry misses and writes again.
            if self._stats is not None:
                self._stats.record_write_failure()
            raise

    def _entry_count(self) -> int:
        if self._cache is None:
            return 0
        return len(self._cache)

    def _get_cache(self, func: Callable) -> diskcache.Cache | None:
        if self._cache is None:
            if self._name is None:
                func = inspect.unwrap(func)
                self._name = f"{func.__module__}_{func.__qualname__}"
            storage_root = os.environ.get("PATHWAY_PERSISTENT_STORAGE")
            if storage_root is None:
                raise RuntimeError(
                    "no persistent storage configured for the disk cache"
                )
            cache_dir = Path(storage_root) / "runtime_calls"
            self._cache = diskcache.Cache(
                cache_dir / self._name, size_limit=self._size_limit
            )
        return self._cache


class DefaultCache(DiskCache):
    """
    The default caching strategy.
    Persistence layer will be used if enabled. Otherwise, cache will be disabled.
    """

    def _get_cache(self, func: Callable) -> diskcache.Cache | None:
        if "PATHWAY_PERSISTENT_STORAGE" not in os.environ:
            return None
        return super()._get_cache(func)


class InMemoryCache(CacheStrategy):
    """In-memory LRU cache. It is not persisted between runs."""

    max_size: int | None
    _cached: Any

    def __init__(self, max_size: int | None = None, *, name: str | None = None) -> None:
        """
        Args:
            max_size: Maximum size of the cache (the number of entries).
            If set to None, it is unlimited.
            name: Explicit name of the cache. Only a cache with an explicit name
                keeps statistics available through ``get_cache_stats`` or
                ``get_stats``.
        """
        super().__init__(name)
        self.max_size = max_size
        self._cached = None

    def _entry_count(self) -> int:
        if self._cached is None:
            return 0
        return self._cached.cache_info().currsize

    def wrap_async(self, func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        # The LRU engine (async_lru) calls the wrapped function only on a miss.
        # The outer wrapper accounts for the lookup itself, so a hit is derived
        # as ``lookups - misses``. Concurrent calls with the same arguments are
        # coalesced by async_lru: the function runs once (one miss) and every
        # concurrent caller awaiting its result is counted as a hit. Raised
        # exceptions are not cached (async_lru evicts the failed entry), so
        # retrying the same arguments causes another miss.
        @functools.wraps(func)
        async def miss_tracker(*args: P.args, **kwargs: P.kwargs) -> T:
            if self._stats is not None:
                self._stats.record_backend_miss()
            return await func(*args, **kwargs)

        cached = async_lru.alru_cache(self.max_size)(miss_tracker)
        self._cached = cached

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if self._stats is not None:
                self._stats.record_lookup()
            return await cached(*args, **kwargs)

        return wrapper

    def wrap_sync(self, func: Callable[P, T]) -> Callable[P, T]:
        # ``functools.lru_cache`` calls the wrapped function only on a miss.
        # The outer wrapper accounts for the lookup itself, so a hit is derived
        # as ``lookups - misses``. A raised exception is not cached, so retrying
        # the same arguments causes another miss.
        @functools.wraps(func)
        def miss_tracker(*args: P.args, **kwargs: P.kwargs) -> T:
            if self._stats is not None:
                self._stats.record_backend_miss()
            return func(*args, **kwargs)

        cached = functools.lru_cache(self.max_size)(miss_tracker)
        self._cached = cached

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if self._stats is not None:
                self._stats.record_lookup()
            return cached(*args, **kwargs)  # type: ignore[arg-type]

        return wrapper


@overload
def with_cache_strategy(
    func: Callable[P, T], cache_strategy: CacheStrategy
) -> Callable[P, T]: ...


@overload
def with_cache_strategy(
    func: Callable[P, Awaitable[T]], cache_strategy: CacheStrategy
) -> Callable[P, Awaitable[T]]: ...


@check_arg_types
def with_cache_strategy(
    func: Callable[P, Awaitable[T]] | Callable[P, T], cache_strategy: CacheStrategy
):
    """
    Returns a function with applied cache strategy.

    Args:
        cache_strategy: Defines the caching mechanism.
    Returns:
        Callable/Coroutine
    """

    if inspect.iscoroutinefunction(func):
        return cache_strategy.wrap_async(func)
    else:
        return cache_strategy.wrap_sync(func)
