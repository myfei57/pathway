# Copyright © 2026 Pathway

"""Pausable checkpoints for Pathway dataflows.

This module implements the control-plane part of the pausable checkpoint
mechanism.  It is deliberately written against the standard library only, so
the state machine can be unit-tested without the native engine and can be
embedded into both :class:`pathway.internals.graph_runner.state.ScopeState`
and :class:`pathway.internals.table.Table`.

The lifecycle exposed to users is::

    RUNNING --request_pause()--> PAUSING --safe frontier--> PAUSED
                                   |                         |
                                cancel()                  resume()
                                   v                         v
                                CANCELLED                  RUNNING

* pause requests only take effect at a *safe frontier*: all changes already
  admitted into the dataflow have been emitted to, and acknowledged by, the
  output sinks (no in-flight nor pending changes); data arriving while the
  dataflow is ``PAUSING`` is held back and released only after resume;
* every transition into ``PAUSED`` (and every explicit
  :meth:`CheckpointController.persist_checkpoint` call) atomically writes a new
  checkpoint version - failed writes never overwrite the last valid version;
* on resume the previously confirmed changes are restored and outputs for
  them are suppressed, so replayed sources never produce duplicate output;
* multiple inputs are tracked independently but the safe frontier is global;
* late data (event time below the current watermark) is accepted, accounted
  for and never moves the watermark backwards nor reopens a finalized
  checkpoint;
* schema fingerprints are recorded per input: incompatible schema changes on
  resume raise :class:`IncompatibleSchemaError` unless explicitly allowed.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import enum
import hashlib
import json
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Hashable

# How many late changes are remembered inside a checkpoint for observability.
_MAX_LATE_IN_CHECKPOINT = 64


class CheckpointState(enum.Enum):
    """State of a pausable dataflow checkpoint.

    RUNNING: the dataflow admits and processes new data.
    PAUSING: a pause was requested; new data is held while admitted changes
        are drained towards a safe frontier.
    PAUSED: the dataflow is stopped at a safe frontier and its state was
        durably persisted.
    PERSISTING: a checkpoint version is being written.
    RESUMING: the last valid checkpoint version is being restored.
    CANCELLED: terminal state, the dataflow does not accept any more data.
    """

    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    PERSISTING = "persisting"
    RESUMING = "resuming"
    CANCELLED = "cancelled"

    def __str__(self) -> str:
        return self.value


class CheckpointError(RuntimeError):
    """Base class for checkpoint related errors."""


class InvalidCheckpointTransition(CheckpointError):
    """Raised when an operation is not allowed in the current state."""


class CheckpointPersistError(CheckpointError):
    """Raised when a checkpoint cannot be persisted.

    The previously committed (last valid) checkpoint version is always left
    untouched when this error is raised.
    """


class IncompatibleSchemaError(CheckpointError):
    """Raised on resume when an input schema changed in an incompatible way."""

    def __init__(self, input_id: str, old_fingerprint: str, new_fingerprint: str):
        self.input_id = input_id
        self.old_fingerprint = old_fingerprint
        self.new_fingerprint = new_fingerprint
        super().__init__(
            f"schema of input {input_id!r} changed: "
            f"{old_fingerprint!r} -> {new_fingerprint!r}; resume with"
            " allow_schema_change=True to accept the new schema"
        )


class CheckpointCancelled(InvalidCheckpointTransition):
    """Raised when data is fed to a cancelled dataflow."""


@dataclass(frozen=True)
class LateChange:
    """A change received below the current input watermark."""

    input_id: str
    change_id: Hashable
    event_time: int
    watermark: int


@dataclass(frozen=True)
class Checkpoint:
    """An immutable, versioned snapshot of the dataflow progress.

    Attributes:
        version: monotonically increasing checkpoint version.
        positions: per-input number of admitted changes (source offset).
        watermarks: per-input event-time watermark (``None`` if unset).
        schemas: per-input schema fingerprint (``None`` if never announced).
        schema_versions: per-input schema revision (bumped on schema change).
        confirmed: changes whose output was already acknowledged downstream.
            Resumed runs suppress outputs for these changes.
        late_changes: a bounded sample of the late changes seen so far.
        created_at: wall-clock timestamp of the checkpoint.
        name: optional dataflow name.
    """

    version: int
    positions: Mapping[str, int] = field(default_factory=dict)
    watermarks: Mapping[str, int | None] = field(default_factory=dict)
    schemas: Mapping[str, str | None] = field(default_factory=dict)
    schema_versions: Mapping[str, int] = field(default_factory=dict)
    confirmed: frozenset[tuple[str, Hashable]] = frozenset()
    late_changes: tuple[LateChange, ...] = ()
    created_at: float = 0.0
    name: str | None = None

    _FORMAT_VERSION: int = 1

    @property
    def frontier(self) -> Mapping[str, int]:
        """The safe frontier (per-input source positions) of this version."""
        return dict(self.positions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self._FORMAT_VERSION,
            "version": self.version,
            "name": self.name,
            "created_at": self.created_at,
            "positions": dict(self.positions),
            "watermarks": dict(self.watermarks),
            "schemas": dict(self.schemas),
            "schema_versions": dict(self.schema_versions),
            "confirmed": [[input_id, change_id] for input_id, change_id in self.confirmed],
            "late_changes": [dataclasses.asdict(late) for late in self.late_changes],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Checkpoint:
        if not isinstance(payload, Mapping):
            raise ValueError("checkpoint payload must be a JSON object")
        format_version = payload.get("format_version")
        if format_version != cls._FORMAT_VERSION:
            raise ValueError(
                f"unsupported checkpoint format version: {format_version!r}"
            )
        try:
            version = int(payload["version"])
            confirmed: list[tuple[str, Hashable]] = [
                (str(entry[0]), entry[1]) for entry in payload["confirmed"]
            ]
            late_changes = tuple(
                LateChange(
                    input_id=str(entry["input_id"]),
                    change_id=entry["change_id"],
                    event_time=int(entry["event_time"]),
                    watermark=int(entry["watermark"]),
                )
                for entry in payload["late_changes"]
            )
            return cls(
                version=version,
                name=payload.get("name"),
                created_at=float(payload.get("created_at", 0.0)),
                positions={k: int(v) for k, v in payload["positions"].items()},
                watermarks={
                    k: (None if v is None else int(v))
                    for k, v in payload["watermarks"].items()
                },
                schemas={k: v for k, v in payload["schemas"].items()},
                schema_versions={
                    k: int(v) for k, v in payload["schema_versions"].items()
                },
                confirmed=frozenset(confirmed),
                late_changes=late_changes,
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"malformed checkpoint payload: {e}") from e


def fingerprint_schema(schema: Mapping[str, Any] | None) -> str | None:
    """Compute a stable fingerprint of a column name -> dtype mapping."""
    if schema is None:
        return None
    if not isinstance(schema, Mapping):
        raise TypeError("schema must be a mapping of column name to dtype")
    materialized = {name: str(dtype) for name, dtype in schema.items()}
    payload = json.dumps(materialized, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class CheckpointStore:
    """Storage backend for atomic, versioned checkpoints."""

    def save(self, checkpoint: Checkpoint) -> None:
        """Persist ``checkpoint`` as a new valid version.

        Implementations must be atomic: a failure leaves the previously
        stored version available through :meth:`load_latest`.
        """
        raise NotImplementedError

    def load_latest(self) -> Checkpoint | None:
        """Return the newest valid checkpoint version, if any."""
        raise NotImplementedError

    def versions(self) -> list[int]:
        """Sorted list of the versions currently held by the store."""
        raise NotImplementedError


class InMemoryCheckpointStore(CheckpointStore):
    """Checkpoint store keeping serialized versions in memory (tests/demos)."""

    def __init__(self) -> None:
        self._versions: dict[int, bytes] = {}
        self._latest_version: int | None = None

    def save(self, checkpoint: Checkpoint) -> None:
        # Serialize first: a serialization failure must not touch the store.
        payload = json.dumps(checkpoint.to_dict(), sort_keys=True).encode("utf-8")
        self._versions[checkpoint.version] = payload
        self._latest_version = checkpoint.version

    def load_latest(self) -> Checkpoint | None:
        if self._latest_version is None:
            return None
        payload = self._versions[self._latest_version]
        return Checkpoint.from_dict(json.loads(payload.decode("utf-8")))

    def versions(self) -> list[int]:
        return sorted(self._versions)


class FileSystemCheckpointStore(CheckpointStore):
    """Directory-backed checkpoint store with atomic version replacement.

    Layout::

        <path>/checkpoint-000000000001.json
        <path>/checkpoint-000000000002.json
        <path>/latest.json

    Data files and the pointer are written to a temporary file and atomically
    renamed, so a crash (or a write failure) can never expose a partially
    written version: the pointer keeps referencing the previous valid
    version.  Only the newest and the previous valid version are retained.
    """

    _PREFIX = "checkpoint-"
    _SUFFIX = ".json"
    _POINTER = "latest.json"

    def __init__(self, path: str | os.PathLike[str], *, keep_versions: int = 2):
        self._path = os.fspath(path)
        if keep_versions < 1:
            raise ValueError("keep_versions must be at least 1")
        self._keep_versions = keep_versions
        os.makedirs(self._path, exist_ok=True)

    @property
    def path(self) -> str:
        return self._path

    def _version_file(self, version: int) -> str:
        return os.path.join(
            self._path, f"{self._PREFIX}{version:012d}{self._SUFFIX}"
        )

    def _write_payload(self, destination: str, payload: bytes) -> None:
        """Atomically write ``payload`` to ``destination``.

        Factored out so that failure injection (subclasses / monkeypatching)
        can simulate broken writes without touching the valid pointer.
        """
        tmp_file = f"{destination}.tmp.{os.getpid()}"
        with open(tmp_file, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, destination)

    def save(self, checkpoint: Checkpoint) -> None:
        data_path = self._version_file(checkpoint.version)
        pointer_path = os.path.join(self._path, self._POINTER)
        # Serialize before touching the filesystem.
        payload = json.dumps(checkpoint.to_dict(), sort_keys=True).encode("utf-8")
        pointer = json.dumps(
            {"format_version": Checkpoint._FORMAT_VERSION, "version": checkpoint.version},
            sort_keys=True,
        ).encode("utf-8")

        previous = self._pointer_version()
        try:
            self._write_payload(data_path, payload)
            # The pointer is replaced only after the data file landed.
            self._write_payload(pointer_path, pointer)
        except BaseException:
            # Remove the half-written data file, keep the previous pointer.
            tmp_file = f"{data_path}.tmp.{os.getpid()}"
            if os.path.exists(tmp_file):
                with contextlib.suppress(OSError):
                    os.remove(tmp_file)
            raise
        self._prune(preserve=checkpoint.version)
        # Sanity check: the version we just advertised must be readable.
        restored = self.load_latest()
        if restored is None or restored.version != checkpoint.version:
            if previous is not None:
                self._write_payload(
                    pointer_path,
                    json.dumps(
                        {
                            "format_version": Checkpoint._FORMAT_VERSION,
                            "version": previous,
                        },
                        sort_keys=True,
                    ).encode("utf-8"),
                )
            raise CheckpointPersistError(
                f"checkpoint version {checkpoint.version} failed verification"
            )

    def _pointer_version(self) -> int | None:
        pointer_path = os.path.join(self._path, self._POINTER)
        if not os.path.exists(pointer_path):
            return None
        try:
            with open(pointer_path, "rb") as f:
                payload = json.loads(f.read().decode("utf-8"))
            return int(payload["version"])
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _load_version(self, version: int) -> Checkpoint | None:
        path = self._version_file(version)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                payload = json.loads(f.read().decode("utf-8"))
            return Checkpoint.from_dict(payload)
        except (OSError, ValueError):
            return None

    def load_latest(self) -> Checkpoint | None:
        version = self._pointer_version()
        if version is not None:
            checkpoint = self._load_version(version)
            if checkpoint is not None and checkpoint.version == version:
                return checkpoint
        # Pointer missing or corrupt: fall back to the newest valid file.
        candidates = self._discover_versions()
        for candidate in reversed(candidates):
            checkpoint = self._load_version(candidate)
            if checkpoint is not None:
                return checkpoint
        return None

    def _discover_versions(self) -> list[int]:
        versions: list[int] = []
        for name in os.listdir(self._path):
            if name.startswith(self._PREFIX) and name.endswith(self._SUFFIX):
                middle = name[len(self._PREFIX) : -len(self._SUFFIX)]
                if middle.isdigit():
                    versions.append(int(middle))
        return sorted(versions)

    def versions(self) -> list[int]:
        return self._discover_versions()

    def _prune(self, preserve: int) -> None:
        candidates = self._discover_versions()
        stale = candidates[: -self._keep_versions] if self._keep_versions else candidates
        for version in stale:
            if version == preserve:
                continue
            path = self._version_file(version)
            try:
                os.remove(path)
            except OSError:
                pass


@dataclass
class _HeldBatch:
    """Data that arrived while PAUSING, to be released on resume."""

    input_id: str
    change_ids: list[Hashable]
    event_times: list[int | None]
    watermark: int | None
    schema: str | None


class CheckpointController:
    """Coordinates pause / persist / resume / cancel around safe frontiers.

    Args:
        inputs: iterable of input names or mapping ``name -> schema
            fingerprint``.  Inputs may also be registered later with
            :meth:`register_input`.
        store: checkpoint storage backend (defaults to an in-memory store).
        name: optional dataflow name recorded in every checkpoint.
        on_pause: optional callback invoked with the checkpoint once the
            dataflow reaches the PAUSED state.
    """

    def __init__(
        self,
        inputs: Iterable[str] | Mapping[str, Any] | None = None,
        *,
        store: CheckpointStore | None = None,
        name: str | None = None,
        on_pause: Callable[[Checkpoint], None] | None = None,
    ):
        self._lock = threading.RLock()
        self._store = store if store is not None else InMemoryCheckpointStore()
        self._name = name
        self._on_pause = on_pause
        self._state = CheckpointState.RUNNING
        self._inputs: dict[str, dict[str, Any]] = {}
        # change identity: (input_id, change_id)
        self._inflight: set[tuple[str, Hashable]] = set()
        self._pending: set[tuple[str, Hashable]] = set()
        self._confirmed: set[tuple[str, Hashable]] = set()
        self._held: list[_HeldBatch] = []
        self._late: collections.deque[LateChange] = collections.deque(
            maxlen=_MAX_LATE_IN_CHECKPOINT
        )
        self._late_counts: dict[str, int] = {}
        self._last_checkpoint: Checkpoint | None = None
        if inputs is not None:
            if isinstance(inputs, Mapping):
                for input_id, schema in inputs.items():
                    self.register_input(input_id, schema)
            else:
                for input_id in inputs:
                    self.register_input(input_id)

    # ------------------------------------------------------------------ #
    # Registration and state introspection
    # ------------------------------------------------------------------ #

    def register_input(
        self, input_id: str, schema: Mapping[str, Any] | str | None = None
    ) -> None:
        """Register a dataflow input, optionally with its schema fingerprint."""
        fingerprint = (
            schema if isinstance(schema, str) else fingerprint_schema(schema)
        )
        with self._lock:
            if input_id in self._inputs:
                if fingerprint is not None:
                    self._announce_schema(input_id, fingerprint)
                return
            self._inputs[input_id] = {
                "position": 0,
                "watermark": None,
                "schema": fingerprint,
                "schema_version": 0,
            }
            self._late_counts.setdefault(input_id, 0)

    @property
    def state(self) -> CheckpointState:
        return self._state

    @property
    def checkpoint_state(self) -> CheckpointState:
        """Alias of :attr:`state` mirroring the user-facing vocabulary."""
        return self._state

    @property
    def is_paused(self) -> bool:
        return self._state is CheckpointState.PAUSED

    @property
    def is_cancelled(self) -> bool:
        return self._state is CheckpointState.CANCELLED

    @property
    def last_checkpoint(self) -> Checkpoint | None:
        return self._last_checkpoint

    @property
    def frontier(self) -> dict[str, int]:
        """Per-input admitted positions (the current global frontier)."""
        with self._lock:
            return {
                input_id: state["position"] for input_id, state in self._inputs.items()
            }

    @property
    def watermarks(self) -> dict[str, int | None]:
        with self._lock:
            return {
                input_id: state["watermark"]
                for input_id, state in self._inputs.items()
            }

    @property
    def safe_frontier(self) -> bool:
        """``True`` iff the dataflow is currently at a safe frontier.

        A frontier is safe when every admitted change has been emitted to an
        output and acknowledged by its sink: there are neither in-flight nor
        pending (unacknowledged) changes.  The condition is global, i.e. it
        takes all registered inputs into account.
        """
        with self._lock:
            return not self._inflight and not self._pending

    @property
    def pending_changes(self) -> frozenset[tuple[str, Hashable]]:
        with self._lock:
            return frozenset(self._pending)

    @property
    def held_change_count(self) -> int:
        with self._lock:
            return sum(len(batch.change_ids) for batch in self._held)

    @property
    def late_changes(self) -> tuple[LateChange, ...]:
        with self._lock:
            return tuple(self._late)

    def late_change_count(self, input_id: str | None = None) -> int:
        with self._lock:
            if input_id is None:
                return sum(self._late_counts.values())
            return self._late_counts.get(input_id, 0)

    def is_confirmed(self, input_id: str, change_id: Hashable) -> bool:
        """Whether output for ``change_id`` was already confirmed."""
        with self._lock:
            return (input_id, change_id) in self._confirmed

    def filter_unconfirmed(
        self, changes: Iterable[tuple[str, Hashable] | Hashable], input_id: str | None = None
    ) -> list:
        """Drop changes that must not be re-emitted after resume."""
        result: list = []
        with self._lock:
            for change in changes:
                if isinstance(change, tuple) and input_id is None:
                    key = (change[0], change[1])
                    value = change
                else:
                    if input_id is None:
                        raise ValueError(
                            "input_id is required for plain change ids"
                        )
                    key = (input_id, change)
                    value = change
                if key not in self._confirmed:
                    result.append(value)
        return result

    # ------------------------------------------------------------------ #
    # Pause / cancel / resume control
    # ------------------------------------------------------------------ #

    def request_pause(self) -> CheckpointState:
        """Request a pause.

        The dataflow leaves ``RUNNING`` immediately, but it becomes PAUSED
        only once the next safe frontier is reached.  New data arriving in
        the meantime is held until resume.  Returns the resulting state
        (``PAUSED`` if the frontier was already safe, ``PAUSING`` otherwise).
        """
        with self._lock:
            if self._state is CheckpointState.CANCELLED:
                raise CheckpointCancelled("cannot pause a cancelled dataflow")
            if self._state in (CheckpointState.PAUSED, CheckpointState.PAUSING):
                self._maybe_finalize_pause()
                return self._state
            if self._state is not CheckpointState.RUNNING:
                raise InvalidCheckpointTransition(
                    f"cannot request pause from state {self._state}"
                )
            self._state = CheckpointState.PAUSING
            self._maybe_finalize_pause()
            return self._state

    # Friendly alias used by the Table-level API.
    pause = request_pause

    def cancel(self) -> CheckpointState:
        """Cancel the dataflow. Terminal; held data is discarded."""
        with self._lock:
            if self._state is CheckpointState.CANCELLED:
                return self._state
            self._held.clear()
            self._pending.clear()
            self._inflight.clear()
            self._state = CheckpointState.CANCELLED
            return self._state

    def persist_checkpoint(self) -> Checkpoint:
        """Persist the current safe frontier without pausing.

        Only callable in ``RUNNING`` and only at a safe frontier - persisting
        an unsafe frontier would make resumed runs skip or duplicate changes.
        """
        with self._lock:
            if self._state is CheckpointState.CANCELLED:
                raise CheckpointCancelled("cannot persist a cancelled dataflow")
            if self._state is not CheckpointState.RUNNING:
                raise InvalidCheckpointTransition(
                    f"cannot persist checkpoint from state {self._state}"
                )
            if not self.safe_frontier:
                raise InvalidCheckpointTransition(
                    "refusing to persist an unsafe frontier: "
                    f"{len(self._inflight)} in-flight, {len(self._pending)} pending"
                )
            checkpoint = self._write_checkpoint(previous_state=self._state)
            return checkpoint

    def resume(
        self,
        *,
        schemas: Mapping[str, Mapping[str, Any] | str | None] | None = None,
        allow_schema_change: bool = False,
    ) -> Checkpoint:
        """Restore the last valid checkpoint and keep processing.

        Usable from PAUSED or on a freshly constructed controller whose store
        contains a checkpoint (cold restart).  Returns the restored
        checkpoint.  Raises :class:`IncompatibleSchemaError` when a registered
        input announces a different schema and ``allow_schema_change`` is
        False.
        """
        with self._lock:
            if self._state is CheckpointState.PAUSED:
                entry_state = CheckpointState.PAUSED
            elif self._state is CheckpointState.RUNNING:
                # Cold restart: a new controller reusing an existing store.
                cold_restart = all(
                    state["position"] == 0 for state in self._inputs.values()
                ) and not self._inflight and not self._pending
                if not cold_restart:
                    raise InvalidCheckpointTransition(
                        "resume() is only valid while PAUSED or on a fresh"
                        " controller with no local progress"
                    )
                entry_state = CheckpointState.RUNNING
            else:
                raise InvalidCheckpointTransition(
                    f"cannot resume from state {self._state}"
                )

            checkpoint = self._store.load_latest()
            if checkpoint is None:
                raise CheckpointError("no valid checkpoint found in the store")

            # Schemas announced while constructing this controller must also
            # agree with the checkpointed ones (relevant for cold restarts).
            announced = {
                input_id: state["schema"]
                for input_id, state in self._inputs.items()
                if state["schema"] is not None
            }
            announced.update(schemas or {})

            self._state = CheckpointState.RESUMING
            try:
                self._restore_checkpoint(
                    checkpoint, announced, allow_schema_change
                )
            except IncompatibleSchemaError:
                self._state = entry_state
                raise

            held = self._held
            self._held = []
            self._state = CheckpointState.RUNNING
            # Release data held during PAUSING, preserving arrival order.
            for batch in held:
                self._admit(
                    batch.input_id,
                    batch.change_ids,
                    batch.event_times,
                    batch.watermark,
                    batch.schema,
                )
            return checkpoint

    # ------------------------------------------------------------------ #
    # Data plane
    # ------------------------------------------------------------------ #

    def ingest(
        self,
        input_id: str,
        change_ids: Iterable[Hashable] = (),
        *,
        event_times: Iterable[int | None] | None = None,
        watermark: int | None = None,
        schema: Mapping[str, Any] | str | None = None,
    ) -> list[Hashable]:
        """Admit changes from ``input_id`` into the dataflow.

        While PAUSING, changes are held and only admitted after resume, so
        the pause frontier stays stable.  Returns the list of change ids that
        were admitted immediately (held changes are therefore excluded).
        """
        change_ids = list(change_ids)
        event_times_list = (
            list(event_times) if event_times is not None else [None] * len(change_ids)
        )
        if len(event_times_list) != len(change_ids):
            raise ValueError("change_ids and event_times must have equal length")
        with self._lock:
            self._ensure_known_input(input_id)
            fingerprint = (
                schema if isinstance(schema, str) else fingerprint_schema(schema)
            )
            if self._state is CheckpointState.CANCELLED:
                raise CheckpointCancelled("dataflow is cancelled")
            if self._state in (
                CheckpointState.PAUSED,
                CheckpointState.PERSISTING,
                CheckpointState.RESUMING,
            ):
                raise InvalidCheckpointTransition(
                    f"cannot ingest data while {self._state}"
                )
            if self._state is CheckpointState.PAUSING:
                # Keep the pause frontier stable: nothing is applied until
                # resume, when the whole batch (data, watermark, schema) is
                # admitted atomically in arrival order.
                self._held.append(
                    _HeldBatch(
                        input_id=input_id,
                        change_ids=list(change_ids),
                        event_times=list(event_times_list),
                        watermark=watermark,
                        schema=fingerprint,
                    )
                )
                return []
            admitted = self._admit(
                input_id, change_ids, event_times_list, watermark, fingerprint
            )
            return admitted

    def advance_watermark(self, input_id: str, watermark: int) -> None:
        """Advance (never lower) the event-time watermark of an input."""
        with self._lock:
            self._ensure_known_input(input_id)
            if self._state is CheckpointState.CANCELLED:
                raise CheckpointCancelled("dataflow is cancelled")
            if self._state in (
                CheckpointState.PAUSED,
                CheckpointState.PERSISTING,
                CheckpointState.RESUMING,
            ):
                raise InvalidCheckpointTransition(
                    f"cannot advance watermark while {self._state}"
                )
            self._advance_watermark(input_id, watermark)

    def emit(
        self,
        input_id: str,
        change_ids: Iterable[Hashable],
    ) -> list[Hashable]:
        """Mark changes as emitted to a sink and return the ones to send.

        Already confirmed or pending changes are filtered out, so sources
        replayed after resume never produce duplicate output.  The returned
        ids are considered pending until :meth:`acknowledge`.
        """
        with self._lock:
            self._ensure_known_input(input_id)
            if self._state is CheckpointState.CANCELLED:
                raise CheckpointCancelled("dataflow is cancelled")
            if self._state not in (
                CheckpointState.RUNNING,
                CheckpointState.PAUSING,
            ):
                raise InvalidCheckpointTransition(
                    f"cannot emit changes while {self._state}"
                )
            emitted: list[Hashable] = []
            for change_id in change_ids:
                key = (input_id, change_id)
                if key in self._confirmed or key in self._pending:
                    continue
                if key not in self._inflight:
                    raise ValueError(
                        f"change {change_id!r} of input {input_id!r} was never"
                        " ingested (or is held behind a pause frontier)"
                    )
                self._inflight.remove(key)
                self._pending.add(key)
                emitted.append(change_id)
            return emitted

    def acknowledge(
        self,
        input_id: str,
        change_ids: Iterable[Hashable] = (),
    ) -> CheckpointState | None:
        """Confirm that outputs for ``change_ids`` were delivered.

        Confirmation is what makes a frontier safe.  If a pause is pending,
        reaching a safe frontier finalizes the checkpoint and transitions the
        dataflow to PAUSED; the resulting state is returned in that case.
        """
        with self._lock:
            self._ensure_known_input(input_id)
            if self._state is CheckpointState.CANCELLED:
                raise CheckpointCancelled("dataflow is cancelled")
            if self._state not in (
                CheckpointState.RUNNING,
                CheckpointState.PAUSING,
            ):
                raise InvalidCheckpointTransition(
                    f"cannot acknowledge changes while {self._state}"
                )
            for change_id in change_ids:
                key = (input_id, change_id)
                self._pending.discard(key)
                self._inflight.discard(key)
                self._confirmed.add(key)
            if self._state is CheckpointState.PAUSING:
                self._maybe_finalize_pause()
                return self._state
            return None

    def settle(
        self,
        input_id: str,
        change_ids: Iterable[Hashable],
        *,
        event_times: Iterable[int | None] | None = None,
        watermark: int | None = None,
        schema: Mapping[str, Any] | str | None = None,
    ) -> list[Hashable]:
        """Convenience: ingest, emit and acknowledge a batch of changes.

        Returns the ids that were emitted for the first time - identical to
        what the sink would have observed.
        """
        admitted = self.ingest(
            input_id,
            change_ids,
            event_times=event_times,
            watermark=watermark,
            schema=schema,
        )
        emitted = self.emit(input_id, admitted)
        if emitted:
            self.acknowledge(input_id, emitted)
        return emitted

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _ensure_known_input(self, input_id: str) -> None:
        if input_id not in self._inputs:
            raise ValueError(f"unknown input: {input_id!r}")

    def _admit(
        self,
        input_id: str,
        change_ids: list[Hashable],
        event_times: list[int | None],
        watermark: int | None,
        fingerprint: str | None,
    ) -> list[Hashable]:
        state = self._inputs[input_id]
        if fingerprint is not None:
            self._announce_schema(input_id, fingerprint)
        current_watermark = state["watermark"]
        new_changes: list[Hashable] = []
        for change_id, event_time in zip(change_ids, event_times):
            key = (input_id, change_id)
            if event_time is not None and current_watermark is not None:
                if event_time < current_watermark:
                    late_change = LateChange(
                        input_id=input_id,
                        change_id=change_id,
                        event_time=event_time,
                        watermark=current_watermark,
                    )
                    self._late.append(late_change)
                    self._late_counts[input_id] = (
                        self._late_counts.get(input_id, 0) + 1
                    )
            if key in self._confirmed:
                # At-least-once source re-delivering an already confirmed
                # change after restart: count it neither as new work nor as
                # in-flight, otherwise the frontier could never be safe and
                # the source position would be double-counted.
                continue
            # Late data is still admitted and processed normally.
            self._inflight.add(key)
            new_changes.append(change_id)
        if watermark is not None:
            self._advance_watermark(input_id, watermark)
        state["position"] += len(new_changes)
        return list(change_ids)

    def _advance_watermark(self, input_id: str, watermark: int) -> None:
        state = self._inputs[input_id]
        if state["watermark"] is None or watermark > state["watermark"]:
            state["watermark"] = watermark
        # A smaller watermark is ignored: watermarks never move backwards.

    def _announce_schema(self, input_id: str, fingerprint: str) -> None:
        state = self._inputs[input_id]
        if state["schema"] is None:
            state["schema"] = fingerprint
        elif state["schema"] != fingerprint:
            # The input changed its schema mid-run.  The revision is bumped;
            # the next checkpoint records both the new fingerprint and the
            # bumped revision, and the previous checkpoint stays valid.
            state["schema"] = fingerprint
            state["schema_version"] += 1

    def _maybe_finalize_pause(self) -> None:
        if self._state is not CheckpointState.PAUSING:
            return
        if not self.safe_frontier:
            return
        self._write_checkpoint(previous_state=CheckpointState.PAUSING)
        self._state = CheckpointState.PAUSED
        if self._on_pause is not None:
            try:
                self._on_pause(self._last_checkpoint)
            except Exception:
                # A misbehaving callback must not corrupt the state machine.
                pass

    def _build_checkpoint(self, version: int) -> Checkpoint:
        return Checkpoint(
            version=version,
            name=self._name,
            created_at=time.time(),
            positions={
                input_id: state["position"]
                for input_id, state in self._inputs.items()
            },
            watermarks={
                input_id: state["watermark"]
                for input_id, state in self._inputs.items()
            },
            schemas={
                input_id: state["schema"] for input_id, state in self._inputs.items()
            },
            schema_versions={
                input_id: state["schema_version"]
                for input_id, state in self._inputs.items()
            },
            confirmed=frozenset(self._confirmed),
            late_changes=tuple(self._late),
        )

    def _write_checkpoint(self, previous_state: CheckpointState) -> Checkpoint:
        version = self._last_checkpoint.version + 1 if self._last_checkpoint else 1
        checkpoint = self._build_checkpoint(version)
        self._state = CheckpointState.PERSISTING
        try:
            self._store.save(checkpoint)
        except Exception as e:
            # Failed checkpoint: revert the state, keep the previous version.
            self._state = previous_state
            if not isinstance(e, CheckpointPersistError):
                raise CheckpointPersistError(str(e)) from e
            raise
        self._last_checkpoint = checkpoint
        self._state = previous_state
        return checkpoint

    def _restore_checkpoint(
        self,
        checkpoint: Checkpoint,
        schema_updates: Mapping[str, Mapping[str, Any] | str | None],
        allow_schema_change: bool,
    ) -> None:
        # Ensure every checkpointed input exists locally (cold restart).
        for input_id in checkpoint.positions:
            if input_id not in self._inputs:
                self.register_input(input_id)
        for input_id in schema_updates:
            if input_id not in self._inputs:
                raise ValueError(f"unknown input: {input_id!r}")

        # Validate everything against the immutable checkpoint first: a
        # rejected schema change must leave the controller untouched, so a
        # cold restart can be retried.
        resolved: dict[str, dict[str, Any]] = {}
        for input_id, state in self._inputs.items():
            if input_id in checkpoint.positions:
                resolved[input_id] = {
                    "position": checkpoint.positions[input_id],
                    "watermark": checkpoint.watermarks.get(input_id),
                    "schema": checkpoint.schemas.get(input_id),
                    "schema_version": checkpoint.schema_versions.get(input_id, 0),
                }
            else:
                resolved[input_id] = {
                    "position": state["position"],
                    "watermark": state["watermark"],
                    "schema": state["schema"],
                    "schema_version": state["schema_version"],
                }

        for input_id, schema in schema_updates.items():
            fingerprint = (
                schema if isinstance(schema, str) else fingerprint_schema(schema)
            )
            if fingerprint is None:
                continue
            old_fingerprint = resolved[input_id]["schema"]
            if old_fingerprint is not None and old_fingerprint != fingerprint:
                if not allow_schema_change:
                    raise IncompatibleSchemaError(
                        input_id, old_fingerprint, fingerprint
                    )
                resolved[input_id]["schema"] = fingerprint
                resolved[input_id]["schema_version"] += 1
            elif old_fingerprint is None:
                resolved[input_id]["schema"] = fingerprint

        # Commit the resolved state only after successful validation.
        for input_id, values in resolved.items():
            self._inputs[input_id].update(values)

        self._confirmed = set(checkpoint.confirmed)
        self._pending.clear()
        self._inflight.clear()
        self._late.clear()
        self._late.extend(checkpoint.late_changes)
        self._late_counts = {input_id: 0 for input_id in self._inputs}
        for late_change in checkpoint.late_changes:
            self._late_counts[late_change.input_id] = (
                self._late_counts.get(late_change.input_id, 0) + 1
            )
        self._last_checkpoint = checkpoint

    def snapshot(self) -> Checkpoint:
        """Build (but do not persist) a checkpoint view of current progress."""
        with self._lock:
            version = (
                self._last_checkpoint.version + 1 if self._last_checkpoint else 1
            )
            return self._build_checkpoint(version)
