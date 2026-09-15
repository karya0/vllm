# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapter contract: forward access metadata, never request a transfer."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.kv_offload.base import ReqContext, make_offload_key
from vllm.v1.kv_offload.tiering.kvcr.manager import (
    KVCRSecondaryTierManager,
    _VllmKeyAdapter,
)


@pytest.mark.parametrize("key_count", [0, 1, 4096])
def test_touch_batches_encoded_keys_without_data_operations(key_count):
    # Constructor/resource coverage is in the companion real-tiering test.
    core = SimpleNamespace(touch=Mock())
    adapter = object.__new__(KVCRSecondaryTierManager)
    adapter._key_adapter = _VllmKeyAdapter()
    adapter._kvcr = core
    keys = [make_offload_key(i.to_bytes(8, "big"), i % 2) for i in range(key_count)]

    adapter.touch(keys, ReqContext(req_id="native-prefix-hit"))

    # Missing query/fetch/deposit methods intentionally make accidental I/O fail.
    core.touch.assert_called_once_with(tuple(keys))


def test_touch_preserves_kv_group_identity():
    core = SimpleNamespace(touch=Mock())
    adapter = object.__new__(KVCRSecondaryTierManager)
    adapter._key_adapter = _VllmKeyAdapter()
    adapter._kvcr = core
    keys = [make_offload_key(b"same-hash", group) for group in (0, 1)]

    adapter.touch(keys, ReqContext(req_id="two-groups"))

    forwarded = core.touch.call_args.args[0]
    assert forwarded == tuple(keys)
    assert forwarded[0] != forwarded[1]
