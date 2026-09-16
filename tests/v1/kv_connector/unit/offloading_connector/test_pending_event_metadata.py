# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.kv_events import BlockRemoved
from tests.v1.kv_connector.unit.offloading_connector.test_events import (
    _group_config,
    _hash,
    _lookup_chunk,
    _record_chunks,
    _removed_event,
    _request,
    _stored_event,
    _tracker,
    _wire_hash,
)


def test_old_removal_keeps_metadata_for_pending_store():
    tracker, req, group, key = _lookup_chunk()
    tracker.record_store(req, group, 0, key)

    list(tracker.take_events([_removed_event([key])], pending_store_keys={key}))
    list(tracker.take_events([], pending_store_keys={key}))
    [event] = tracker.take_events([_stored_event([key], removal_expected=True)])

    assert event.block_size == 4
    assert event.token_ids == [1, 2, 3, 4]
    assert event.block_hashes == [_wire_hash(_hash(0))]
    assert not tracker._deferred_removals
    list(tracker.take_events([_removed_event([key])]))
    assert key not in tracker._pending_event_metadata


@pytest.mark.parametrize("reset", [False, True])
def test_pending_metadata_is_cleaned_when_store_hold_ends(reset):
    tracker, req, group, key = _lookup_chunk()
    tracker.record_store(req, group, 0, key)
    list(tracker.take_events([_removed_event([key])], pending_store_keys={key}))
    assert key in tracker._pending_event_metadata

    if reset:
        tracker.reset()
    else:
        # A cancelled hold can end without any subsequent inventory event.
        list(tracker.take_events([]))

    assert not tracker._pending_event_metadata
    assert not tracker._deferred_removals
    tracker.record_store(req, group, 0, key)
    [event] = tracker.take_events([_stored_event([key], removal_expected=True)])
    assert event.block_size == 4
    assert event.token_ids == [1, 2, 3, 4]


def test_pending_store_does_not_retain_unrelated_removed_metadata():
    tracker = _tracker()
    req = _request(block_hashes=[_hash(0), _hash(1)], token_count=8)
    kept, removed = _record_chunks(tracker, req, _group_config(), num_chunks=2)

    list(
        tracker.take_events(
            [_removed_event([kept, removed])], pending_store_keys={kept}
        )
    )
    assert kept in tracker._pending_event_metadata
    assert removed not in tracker._pending_event_metadata
    assert tracker._deferred_removals == {kept}


def test_same_batch_secondary_store_keeps_metadata_after_primary_removal():
    tracker, req, group, key = _lookup_chunk()
    events = list(
        tracker.take_events(
            [
                _removed_event([key]),
                _stored_event([key], ownership="kvcr", removal_expected=True),
            ]
        )
    )

    assert isinstance(events[0], BlockRemoved)
    assert events[1].block_size == 4
    assert events[1].token_ids == [1, 2, 3, 4]
    assert events[1].ownership == "kvcr"
    list(tracker.take_events([_removed_event([key], ownership="kvcr")]))
    assert key not in tracker._pending_event_metadata
