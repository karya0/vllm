# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real Scheduler/native-cache regression; model and GPU completion are synthetic."""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from tests.v1.kv_connector.unit.offloading_connector import utils
from tests.v1.kv_connector.unit.utils import create_model_runner_output
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.v1.kv_offload.base import LookupResult, ReqContext
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)


@pytest.fixture(params=[1, 4])
def engine(request, monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "opt",
                "architectures": ["OPTForCausalLM"],
                "hidden_size": 64,
                "ffn_dim": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "vocab_size": 50272,
                "max_position_embeddings": 2048,
            }
        )
    )
    create_config = utils.create_vllm_config
    monkeypatch.setattr(
        utils,
        "create_vllm_config",
        lambda **kw: create_config(
            model=str(tmp_path),
            **(kw | {"disable_hybrid_kv_cache_manager": True}),
        ),
    )
    runner = utils.RequestRunner(
        block_size=64,
        num_gpu_blocks=16,
        async_scheduling=False,
        worker_count=request.param,
        extra_config_overrides={"offload_prompt_only": True},
    )
    offload = runner.connector_scheduler
    # Real two-block native cache, with NO secondary availability mock.
    region = MagicMock()
    region.create_kv_memoryview.return_value = memoryview(np.zeros((2, 16), np.int8))
    primary = CPUPrimaryTierOffloadingManager(
        2,
        region,
        cache_policy="lru",
        enable_events=True,
    )
    manager = TieringOffloadingManager(primary, [])
    offload.manager = manager
    core = runner.scheduler
    core.connector.connector_scheduler = offload
    published = []
    monkeypatch.setattr(
        core.kv_event_publisher, "publish", lambda batch: published.extend(batch.events)
    )
    yield runner, core, offload, primary, published
    offload.reset_cache()
    primary.shutdown()


def update(core, scheduled, *, completed=(), count=None, eos=False):
    if count is None:
        count = core.connector.connector_scheduler.config.num_workers
    output = create_model_runner_output(
        reqs=core.running,
        use_eos=eos,
        kv_connector_worker_meta=OffloadingWorkerMetadata(
            completed_jobs={job: count for job in completed},
        ),
    )
    core.update_from_output(scheduled, output)


def finish_prompt(engine, tokens):
    runner, core, offload, primary, published = engine
    runner.new_request(tokens)
    scheduled = core.schedule()
    jobs = scheduled.kv_connector_metadata.store_jobs
    assert len(jobs) == 1
    job_id = next(iter(jobs))
    key = next(iter(offload._jobs[job_id].keys))
    update(core, scheduled, completed=[job_id], eos=True)
    for _ in range(2):
        step = core.schedule()
        update(core, step)
    assert not core.requests
    assert not offload._jobs
    return key


def test_real_core_gpu_hit_repopulation_keeps_pending_metadata(engine):
    runner, core, offload, primary, published = engine
    tokens = list(range(65))
    later_tokens = list(range(1000, 1065))
    key = finish_prompt(engine, tokens)
    finish_prompt(engine, later_tokens)
    published.clear()
    # Admission touches K then L. New J evicts native K during store-building;
    # GPU-hit K then repopulates CPU by evicting L. Old removal drains later.
    runner.new_request(list(range(2000, 2065)))
    runner.new_request(tokens)
    key_req_id = str(runner.req_id)
    runner.new_request(later_tokens)
    scheduled = core.schedule()
    assert scheduled.num_scheduled_tokens[key_req_id] == 1  # 64 GPU-hit tokens.
    assert not scheduled.kv_connector_metadata.load_jobs
    jobs = scheduled.kv_connector_metadata.store_jobs
    key_job = next(j for j in jobs if key in offload._jobs[j].keys)
    assert primary.lookup(key, ReqContext(req_id="check")) is LookupResult.HIT_PENDING
    workers = offload.config.num_workers
    # Four-rank case: one worker completes, but the job must keep metadata alive
    # until the remaining three finish. One-rank case: completion is deferred.
    update(core, scheduled, completed=jobs if workers > 1 else (), count=1, eos=True)
    assert offload._jobs[key_job].pending_count == max(1, workers - 1)
    survived = key in offload._events_tracker._pending_event_metadata
    pending_step = core.schedule()
    update(core, pending_step, completed=jobs, count=max(1, workers - 1))
    stores = [
        e
        for e in published
        if getattr(e, "medium", None) == "CPU" and hasattr(e, "block_size")
    ]
    print(
        {
            "real_core_schedule": True,
            "workers": workers,
            "gpu_hit_tokens": 64,
            "new_gpu_to_cpu_store": key_job,
            "metadata_survived": survived,
            "native_stored_block_sizes": [e.block_size for e in stores],
        }
    )
    assert any(
        event.block_size == 64 and event.token_ids == tokens[:64] for event in stores
    )


def test_real_core_native_hit_loads_instead_of_storing_same_key(engine):
    runner, core, offload, primary, published = engine
    tokens = list(range(65))
    key = finish_prompt(engine, tokens)
    assert primary.lookup(key, ReqContext(req_id="check")) is LookupResult.HIT
    # Explicit negative control: clear only GPU prefix index, retain CPU K.
    assert core.kv_cache_manager.reset_prefix_cache()
    runner.new_request(tokens)
    scheduled = core.schedule()
    meta = scheduled.kv_connector_metadata
    assert not meta.store_jobs
    assert len(meta.load_jobs) == 1
    job_id = next(iter(meta.load_jobs))
    assert offload._jobs[job_id].keys == {key}
    assert str(runner.req_id) not in scheduled.num_scheduled_tokens
    print(
        {
            "real_core_schedule": True,
            "native_hit": True,
            "gpu_load_jobs": 1,
            "gpu_store_jobs": 0,
        }
    )
