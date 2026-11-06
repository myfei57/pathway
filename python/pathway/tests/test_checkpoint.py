# Copyright © 2026 Pathway

"""Tests for pausable dataflow checkpoints.

Covers:
* pause taking effect only at safe frontiers (and held data on resume);
* no duplicate output for already confirmed changes after resume;
* failed checkpoints retaining the previous valid version;
* multiple inputs sharing one global frontier;
* late data below the watermark;
* schema changes on resume;
* the terminal cancelled state;
* the Table-level API in ``table.py``.
"""

import json

import pytest

import pathway as pw
from pathway.internals.checkpoint import (
    Checkpoint,
    CheckpointCancelled,
    CheckpointPersistError,
    CheckpointState,
    FileSystemCheckpointStore,
    InMemoryCheckpointStore,
    IncompatibleSchemaError,
    InvalidCheckpointTransition,
    fingerprint_schema,
)

SCHEMA_V1 = {"value": "int"}
SCHEMA_V2 = {"value": "int", "label": "str"}


# --------------------------------------------------------------------------- #
# Basic lifecycle
# --------------------------------------------------------------------------- #


def test_initial_state_is_running():
    controller = pw.CheckpointController(["input"])
    assert controller.state is CheckpointState.RUNNING
    assert str(controller.state) == "running"
    assert controller.safe_frontier
    assert controller.last_checkpoint is None


def test_pause_at_already_safe_frontier_persists_immediately():
    controller = pw.CheckpointController(["input"])
    state = controller.request_pause()
    assert state is CheckpointState.PAUSED
    assert controller.is_paused
    checkpoint = controller.last_checkpoint
    assert checkpoint is not None
    assert checkpoint.version == 1
    assert checkpoint.positions == {"input": 0}


def test_full_pause_resume_lifecycle():
    controller = pw.CheckpointController(["input"])
    controller.settle("input", [1, 2, 3])
    assert controller.request_pause() is CheckpointState.PAUSED
    assert controller.last_checkpoint.version == 1

    resumed = controller.resume()
    assert resumed.version == 1
    assert controller.state is CheckpointState.RUNNING
    controller.settle("input", [4, 5])
    checkpoint = controller.persist_checkpoint()
    assert checkpoint.version == 2
    assert checkpoint.positions == {"input": 5}
    assert len(checkpoint.confirmed) == 5


# --------------------------------------------------------------------------- #
# Pause takes effect only at a safe frontier
# --------------------------------------------------------------------------- #


def test_pause_waits_for_safe_frontier():
    controller = pw.CheckpointController(["input"])
    controller.ingest("input", [1, 2])
    emitted = controller.emit("input", [1, 2])
    assert emitted == [1, 2]

    state = controller.request_pause()
    # Outputs are not acknowledged yet: the frontier is unsafe, so the pause
    # stays pending and no checkpoint is written.
    assert state is CheckpointState.PAUSING
    assert controller.last_checkpoint is None
    assert not controller.safe_frontier

    with pytest.raises(InvalidCheckpointTransition):
        controller.persist_checkpoint()

    state = controller.acknowledge("input", [1])
    assert state is CheckpointState.PAUSING
    state = controller.acknowledge("input", [2])
    assert state is CheckpointState.PAUSED
    assert controller.last_checkpoint.version == 1
    assert controller.last_checkpoint.positions == {"input": 2}


def test_data_arriving_while_pausing_is_held_and_released_on_resume():
    controller = pw.CheckpointController(["input"])
    controller.ingest("input", [1])
    controller.emit("input", [1])
    assert controller.request_pause() is CheckpointState.PAUSING

    # New data while PAUSING is held: not admitted, not positioned.
    assert controller.ingest("input", [2, 3]) == []
    assert controller.frontier == {"input": 1}
    assert controller.held_change_count == 2

    # Draining the admitted change finalizes the pause at the old frontier.
    assert controller.acknowledge("input", [1]) is CheckpointState.PAUSED
    assert controller.last_checkpoint.positions == {"input": 1}

    controller.resume()
    assert controller.state is CheckpointState.RUNNING
    assert controller.held_change_count == 0
    assert controller.frontier == {"input": 3}
    # The released changes flow through the pipeline normally.
    assert set(controller.emit("input", [2, 3])) == {2, 3}


def test_on_pause_callback_receives_checkpoint():
    events = []
    controller = pw.CheckpointController(
        ["input"], on_pause=lambda checkpoint: events.append(checkpoint.version)
    )
    controller.request_pause()
    assert events == [1]


# --------------------------------------------------------------------------- #
# No duplicate output after resume
# --------------------------------------------------------------------------- #


def test_resume_does_not_reemit_confirmed_changes():
    store = InMemoryCheckpointStore()
    first = pw.CheckpointController({"input": SCHEMA_V1}, store=store)
    first.settle("input", [1, 2, 3])
    first.request_pause()
    assert first.last_checkpoint.version == 1

    # Cold restart: the source replays everything through offsets 1..5.
    second = pw.CheckpointController({"input": SCHEMA_V1}, store=store)
    checkpoint = second.resume()
    assert checkpoint.positions == {"input": 3}
    replayed = [1, 2, 3, 4, 5]
    second.ingest("input", replayed)
    emitted = second.emit("input", replayed)
    assert emitted == [4, 5]
    assert second.filter_unconfirmed([1, 2, 3, 4, 5], input_id="input") == [4, 5]
    assert second.is_confirmed("input", 1)
    assert not second.is_confirmed("input", 4)
    # Re-delivered confirmed changes are not double-counted in the position.
    assert second.frontier == {"input": 5}
    # A safe frontier remains reachable, and the next checkpoint de-duplicates.
    second.acknowledge("input", [4, 5])
    assert second.safe_frontier
    assert second.request_pause() is CheckpointState.PAUSED
    latest = store.load_latest()
    assert latest.version == 2
    assert latest.positions == {"input": 5}
    assert len(latest.confirmed) == 5


def test_repeated_emit_is_idempotent():
    controller = pw.CheckpointController(["input"])
    controller.settle("input", [1])
    assert controller.emit("input", [1]) == []
    controller.ingest("input", [2])
    assert controller.emit("input", [2, 2]) == [2]


# --------------------------------------------------------------------------- #
# Multiple inputs
# --------------------------------------------------------------------------- #


def test_multi_input_shares_one_global_safe_frontier():
    controller = pw.CheckpointController({"left": None, "right": None})
    controller.settle("left", ["a"])
    controller.ingest("right", ["b"])
    controller.emit("right", ["b"])

    assert controller.request_pause() is CheckpointState.PAUSING
    # Acknowledging only one input is not enough: the frontier is global.
    controller.acknowledge("left", ["a"])
    assert controller.state is CheckpointState.PAUSING
    controller.acknowledge("right", ["b"])
    assert controller.state is CheckpointState.PAUSED

    checkpoint = controller.last_checkpoint
    assert checkpoint.positions == {"left": 1, "right": 1}


def test_resume_restores_all_input_positions():
    store = InMemoryCheckpointStore()
    first = pw.CheckpointController(["left", "right"], store=store)
    first.settle("left", [1])
    first.settle("right", [10, 20])
    first.request_pause()

    second = pw.CheckpointController(["left", "right"], store=store)
    checkpoint = second.resume()
    assert checkpoint.positions == {"left": 1, "right": 2}


# --------------------------------------------------------------------------- #
# Late data
# --------------------------------------------------------------------------- #


def test_late_data_is_accepted_but_flagged():
    controller = pw.CheckpointController(["input"])
    controller.advance_watermark("input", 10)

    controller.ingest("input", ["on-time"], event_times=[12])
    assert controller.late_change_count("input") == 0

    controller.ingest("input", ["late"], event_times=[5])
    assert controller.late_change_count("input") == 1
    late = controller.late_changes[0]
    assert late.input_id == "input"
    assert late.change_id == "late"
    assert (late.event_time, late.watermark) == (5, 10)

    # The late change is still processed and output normally.
    assert controller.emit("input", ["on-time", "late"]) == ["on-time", "late"]


def test_watermark_never_moves_backwards():
    controller = pw.CheckpointController(["input"])
    controller.advance_watermark("input", 10)
    controller.advance_watermark("input", 3)
    assert controller.watermarks["input"] == 10
    controller.request_pause()
    assert controller.last_checkpoint.watermarks == {"input": 10}


def test_late_data_does_not_reopen_checkpoint_nor_duplicate_after_resume():
    store = InMemoryCheckpointStore()
    first = pw.CheckpointController(["input"], store=store)
    first.settle("input", [1, 2])
    first.request_pause()

    second = pw.CheckpointController(["input"], store=store)
    second.resume()
    second.advance_watermark("input", 100)
    # A replay of change 1 below the watermark is late AND already confirmed.
    assert second.ingest("input", [1], event_times=[5]) == [1]
    assert second.late_change_count("input") == 1
    assert second.emit("input", [1]) == []


# --------------------------------------------------------------------------- #
# Schema changes
# --------------------------------------------------------------------------- #


def test_resume_with_unchanged_schema_succeeds():
    store = InMemoryCheckpointStore()
    first = pw.CheckpointController({"input": SCHEMA_V1}, store=store)
    first.settle("input", [1])
    first.request_pause()

    second = pw.CheckpointController({"input": SCHEMA_V1}, store=store)
    checkpoint = second.resume()
    assert checkpoint.schemas["input"] == fingerprint_schema(SCHEMA_V1)
    assert checkpoint.schema_versions["input"] == 0


def test_resume_with_changed_schema_raises_and_keeps_state():
    store = InMemoryCheckpointStore()
    first = pw.CheckpointController({"input": SCHEMA_V1}, store=store)
    first.settle("input", [1])
    first.request_pause()

    second = pw.CheckpointController({"input": SCHEMA_V2}, store=store)
    with pytest.raises(IncompatibleSchemaError):
        second.resume()
    # The failed resume does not destroy the checkpoint nor start processing.
    assert second.state is CheckpointState.RUNNING
    assert store.load_latest().version == 1

    resumed = second.resume(allow_schema_change=True)
    # The immutable stored version is unchanged; the bumped revision is
    # visible in the runtime snapshot and in the next persisted version.
    assert resumed.version == 1
    assert second.snapshot().schema_versions["input"] == 1
    assert second.frontier == {"input": 1}
    # The next checkpoint records the new schema revision while remaining a
    # new, valid version on top of the old one.
    second.settle("input", [2])
    second.request_pause()
    latest = store.load_latest()
    assert latest.version == 2
    assert latest.schemas["input"] == fingerprint_schema(SCHEMA_V2)
    assert latest.schema_versions["input"] == 1


def test_warm_resume_rejects_schema_change_unless_allowed():
    controller = pw.CheckpointController({"input": SCHEMA_V1})
    controller.request_pause()
    with pytest.raises(IncompatibleSchemaError):
        controller.resume(schemas={"input": SCHEMA_V2})
    assert controller.state is CheckpointState.PAUSED
    controller.resume(schemas={"input": SCHEMA_V2}, allow_schema_change=True)
    assert controller.snapshot().schema_versions["input"] == 1


# --------------------------------------------------------------------------- #
# Failed checkpoints retain the previous valid version
# --------------------------------------------------------------------------- #


class _FailingPointerStore(FileSystemCheckpointStore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_pointer_writes = False

    def _write_payload(self, destination, payload):
        if self.fail_pointer_writes and destination.endswith(self._POINTER):
            raise OSError("simulated disk failure")
        super()._write_payload(destination, payload)


def test_failed_persist_keeps_previous_valid_version(tmp_path):
    store = _FailingPointerStore(tmp_path / "ckpt")
    controller = pw.CheckpointController(["input"], store=store)
    controller.settle("input", [1, 2])
    controller.request_pause()
    assert store.load_latest().version == 1

    controller.resume()
    controller.settle("input", [3, 4])
    store.fail_pointer_writes = True
    with pytest.raises(CheckpointPersistError):
        controller.persist_checkpoint()

    # The pointer still references the previous valid version...
    assert store.load_latest().version == 1
    # ...the controller kept running and the failed version was not adopted.
    assert controller.state is CheckpointState.RUNNING
    assert controller.last_checkpoint.version == 1

    store.fail_pointer_writes = False
    controller.request_pause()
    assert store.load_latest().version == 2
    assert store.load_latest().positions == {"input": 4}


def test_failed_pause_persist_returns_to_pausing(tmp_path):
    store = _FailingPointerStore(tmp_path / "ckpt")
    controller = pw.CheckpointController(["input"], store=store)
    controller.settle("input", [1])
    controller.request_pause()
    controller.resume()
    controller.settle("input", [2])

    store.fail_pointer_writes = True
    with pytest.raises(CheckpointPersistError):
        controller.request_pause()
    # The failed finalization keeps the pause pending and the old version.
    assert controller.state is CheckpointState.PAUSING
    assert store.load_latest().version == 1
    store.fail_pointer_writes = False
    assert controller.request_pause() is CheckpointState.PAUSED
    assert store.load_latest().version == 2


def test_corrupt_pointer_falls_back_to_newest_valid_file(tmp_path):
    store = FileSystemCheckpointStore(tmp_path / "ckpt")
    controller = pw.CheckpointController(["input"], store=store)
    controller.settle("input", [1])
    controller.request_pause()
    controller.resume()
    controller.settle("input", [2])
    controller.request_pause()

    pointer = tmp_path / "ckpt" / "latest.json"
    pointer.write_text("{not json")
    assert store.load_latest().version == 2

    pointer.write_text(json.dumps({"format_version": 1, "version": 999}))
    # Version 999 does not exist: scan for the newest readable file.
    assert store.load_latest().version == 2


def test_old_versions_are_pruned_but_one_previous_is_kept(tmp_path):
    store = FileSystemCheckpointStore(tmp_path / "ckpt", keep_versions=2)
    controller = pw.CheckpointController(["input"], store=store)
    for i in range(4):
        controller.settle("input", [i])
        controller.request_pause()
        controller.resume()
    controller.settle("input", [4])
    controller.request_pause()
    assert store.versions() == [4, 5]
    assert store.load_latest().version == 5


def test_checkpoint_serialization_roundtrip():
    checkpoint = Checkpoint(
        version=3,
        positions={"input": 7},
        watermarks={"input": 42},
        schemas={"input": "abc123"},
        schema_versions={"input": 1},
        confirmed=frozenset({("input", 1), ("input", 2)}),
    )
    restored = Checkpoint.from_dict(checkpoint.to_dict())
    assert restored == checkpoint


def test_malformed_payload_is_rejected():
    with pytest.raises(ValueError):
        Checkpoint.from_dict({"format_version": 1})


# --------------------------------------------------------------------------- #
# Cancelling
# --------------------------------------------------------------------------- #


def test_cancel_is_terminal_and_keeps_last_checkpoint():
    store = InMemoryCheckpointStore()
    controller = pw.CheckpointController(["input"], store=store)
    controller.settle("input", [1])
    controller.request_pause()
    controller.resume()

    controller.ingest("input", [2])
    controller.emit("input", [2])
    assert controller.request_pause() is CheckpointState.PAUSING
    assert controller.cancel() is CheckpointState.CANCELLED
    assert controller.is_cancelled

    with pytest.raises(CheckpointCancelled):
        controller.ingest("input", [3])
    with pytest.raises(CheckpointCancelled):
        controller.request_pause()
    # Cancellation does not destroy the last valid checkpoint.
    assert store.load_latest().version == 1
    # Idempotent.
    assert controller.cancel() is CheckpointState.CANCELLED


# --------------------------------------------------------------------------- #
# Invalid transitions
# --------------------------------------------------------------------------- #


def test_invalid_transitions():
    from pathway.internals.checkpoint import CheckpointError

    controller = pw.CheckpointController(["input"])
    with pytest.raises(CheckpointError):
        pw.CheckpointController(["input"]).resume()

    controller.ingest("input", [1])
    controller.emit("input", [1])
    with pytest.raises(InvalidCheckpointTransition):
        controller.persist_checkpoint()

    with pytest.raises(ValueError):
        controller.ingest("unknown", [1])
    with pytest.raises(ValueError):
        controller.emit("input", [99])


def test_emitting_while_paused_is_rejected():
    controller = pw.CheckpointController(["input"])
    controller.request_pause()
    with pytest.raises(InvalidCheckpointTransition):
        controller.ingest("input", [1])
    with pytest.raises(InvalidCheckpointTransition):
        controller.emit("input", [1])


# --------------------------------------------------------------------------- #
# Table-level API
# --------------------------------------------------------------------------- #


def test_table_checkpoint_api(tmp_path):
    table = pw.debug.table_from_markdown(
        """
        value
        1
        2
        """
    )
    assert table.checkpoint_state is CheckpointState.RUNNING

    controller = table.enable_checkpointing(
        store=FileSystemCheckpointStore(tmp_path / "ckpt")
    )
    input_name = table._default_checkpoint_input_name()

    controller.settle(input_name, ["r1", "r2"])
    assert table.checkpoint_state is CheckpointState.RUNNING

    state = table.request_pause()
    assert state is CheckpointState.PAUSED
    assert table.checkpoint_state is CheckpointState.PAUSED
    assert table.persist_checkpoint is not None

    resumed = table.resume_checkpoint()
    assert resumed.version == 1
    assert table.checkpoint_state is CheckpointState.RUNNING
    assert table.filter_unconfirmed(["r1", "r2", "r3"]) == ["r3"]

    assert table.cancel_checkpoint() is CheckpointState.CANCELLED
    assert table.checkpoint_state is CheckpointState.CANCELLED


def test_multi_input_table_api(tmp_path):
    left = pw.debug.table_from_markdown("v\n1")
    right = pw.debug.table_from_markdown("w\n2")

    store = FileSystemCheckpointStore(tmp_path / "ckpt")
    controller = left.enable_checkpointing(store=store, input_name="left")
    right.attach_checkpoint(controller, input_name="right")
    assert left.checkpoint_controller is right.checkpoint_controller

    controller.settle("left", [1])
    controller.ingest("right", [2])
    controller.emit("right", [2])
    assert left.request_pause() is CheckpointState.PAUSING
    assert controller.acknowledge("right", [2]) is CheckpointState.PAUSED
    assert controller.last_checkpoint.positions == {"left": 1, "right": 1}


def test_table_schema_fingerprint_is_stable():
    table = pw.debug.table_from_markdown("v\n1")
    fingerprint = table._schema_fingerprint()
    assert isinstance(fingerprint, str) and len(fingerprint) == 16
    assert fingerprint == table._schema_fingerprint()
