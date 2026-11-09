"""CPU regression tests for configured block sizes.

Run with pytest, torch, numpy and xxhash installed. CUDA/model initialization
is mocked; Sequence, Scheduler, BlockManager and model input preparation are
real. The package initializer is bypassed to avoid importing FlashAttention.
"""
import importlib
import os
from pathlib import Path
import pickle
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.fixture
def runtime(monkeypatch):
    root = Path(os.environ.get("NANOVLLM_TEST_ROOT", Path(__file__).parents[1]))
    for name, directory in [("nanovllm", "nanovllm"),
                            ("nanovllm.engine", "nanovllm/engine"),
                            ("nanovllm.utils", "nanovllm/utils")]:
        module = types.ModuleType(name)
        module.__path__ = [str(root / directory)]
        monkeypatch.setitem(sys.modules, name, module)
    for name, attr in [("nanovllm.config", "Config"),
                       ("nanovllm.models.qwen3", "Qwen3ForCausalLM"),
                       ("nanovllm.layers.sampler", "Sampler"),
                       ("nanovllm.utils.loader", "load_model")]:
        module = types.ModuleType(name)
        setattr(module, attr, MagicMock())
        monkeypatch.setitem(sys.modules, name, module)
    module_names = ["nanovllm.sampling_params", "nanovllm.engine.sequence",
                    "nanovllm.engine.block_manager", "nanovllm.engine.scheduler",
                    "nanovllm.utils.context", "nanovllm.engine.model_runner"]
    saved = {name: sys.modules.pop(name, None) for name in module_names}
    try:
        modules = [importlib.import_module(name) for name in module_names]
        yield SimpleNamespace(params=modules[0].SamplingParams,
                              sequence=modules[1].Sequence,
                              scheduler=modules[3].Scheduler,
                              context=modules[4], runner=modules[5])
    finally:
        for name, module in saved.items():
            sys.modules.pop(name, None)
            if module is not None:
                sys.modules[name] = module


def make_runner(runtime, block_size, rank=0):
    config = SimpleNamespace(kvcache_block_size=block_size, enforce_eager=True,
                             tensor_parallel_size=rank + 1, model="unused",
                             hf_config=SimpleNamespace(torch_dtype=torch.float32),
                             chunked_prefill=True, max_model_len=8192,
                             max_num_seqs=8, max_num_batched_tokens=4096,
                             eos=-1, num_kvcache_blocks=32)
    model_runner = runtime.runner.ModelRunner
    # Execute the actual per-rank constructor, excluding hardware/model work.
    with patch.object(runtime.runner.dist, "init_process_group"), \
         patch.object(runtime.runner.dist, "barrier"), \
         patch.object(torch.cuda, "set_device"), \
         patch.object(torch, "set_default_device"), \
         patch.object(model_runner, "warmup_model"), \
         patch.object(model_runner, "allocate_kv_cache"), \
         patch.object(model_runner, "loop"), \
         patch.object(runtime.runner, "SharedMemory"):
        runner = model_runner(config, rank, [])
    return runner, config


def prepare_inputs(runtime, runner, seqs):
    tensor = torch.tensor
    def cpu_tensor(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return tensor(*args, **kwargs)
    with patch.object(torch, "tensor", side_effect=cpu_tensor), \
         patch.object(torch.Tensor, "cuda", lambda self, **kwargs: self):
        inputs, positions = runner.prepare_model_input(seqs)
    context = runtime.context.get_context()
    expected_slots = [seq.block_table[position // runner.block_size] * runner.block_size
                      + position % runner.block_size
                      for seq in seqs
                      for position in range(seq.num_cached_tokens, seq.num_context_tokens)]
    assert context.slot_mapping.tolist() == expected_slots
    assert context.slot_mapping.numel() == inputs.numel()
    assert inputs.tolist() == [token for seq in seqs
                              for token in seq[seq.num_cached_tokens:seq.num_context_tokens]]
    return inputs, positions, context


@pytest.mark.parametrize("block_size", [256, 512, 768, 1024])
@pytest.mark.parametrize("prompt_length", [300, 1023, 1024, 1025])
def test_prefill_to_decode(runtime, block_size, prompt_length):
    runner, config = make_runner(runtime, block_size)
    scheduler = runtime.scheduler(config)
    seq = runtime.sequence(list(range(prompt_length)), runtime.params(max_tokens=3))
    scheduler.add(seq)
    for step in range(3):
        seqs = scheduler.schedule()
        prepare_inputs(runtime, runner, seqs)
        scheduler.postprocess(seqs, [10000 + step], [0])
    assert scheduler.is_finished()
    assert len(scheduler.block_manager.free_block_ids) == config.num_kvcache_blocks
    assert not scheduler.block_manager.used_block_ids


@pytest.mark.parametrize("block_size", [256, 512, 768, 1024])
def test_worker_initialization_and_serialized_sequence(runtime, block_size):
    # Emulate a fresh spawn import: a parent process's class assignment would
    # not be inherited. Rank 1 must configure its own Sequence class.
    runtime.sequence.block_size = 256
    runner, _ = make_runner(runtime, block_size, rank=1)
    seq = runtime.sequence(list(range(300)))
    seq.num_cached_tokens = 300
    seq.num_new_tokens = 1
    seq.append_token(1234)
    seq.block_table = list(range((301 + block_size - 1) // block_size))
    worker_seq = pickle.loads(pickle.dumps(seq))
    prepare_inputs(runtime, runner, [worker_seq])


@pytest.mark.parametrize("block_size", [256, 512, 768, 1024])
def test_prefix_cache_reuse(runtime, block_size):
    runner, config = make_runner(runtime, block_size)
    scheduler = runtime.scheduler(config)
    for _ in range(2):
        seq = runtime.sequence(list(range(block_size + 44)), runtime.params(max_tokens=1))
        scheduler.add(seq)
        seqs = scheduler.schedule()
        expected_cached = 0 if _ == 0 else block_size
        assert seq.num_cached_tokens == expected_cached
        prepare_inputs(runtime, runner, seqs)
        scheduler.postprocess(seqs, [10000], [0])
        assert scheduler.is_finished()
        assert len(scheduler.block_manager.free_block_ids) == config.num_kvcache_blocks


@pytest.mark.parametrize("block_size", [256, 512, 768, 1024])
@pytest.mark.parametrize("preempt_after_first_chunk", [False, True])
def test_chunked_prefill_and_preemption(runtime, block_size, preempt_after_first_chunk):
    runner, config = make_runner(runtime, block_size)
    # Full-block chunks isolate this regression from the separately reported
    # partial-block hash bug in PR #4.
    config.max_num_batched_tokens = block_size
    scheduler = runtime.scheduler(config)
    seq = runtime.sequence(list(range(2 * block_size + 44)), runtime.params(max_tokens=3))
    scheduler.add(seq)
    steps = 0
    while not scheduler.is_finished():
        seqs = scheduler.schedule()
        _, _, context = prepare_inputs(runtime, runner, seqs)
        indices = context.seq_need_compute_logits.tolist()
        scheduler.postprocess(seqs, [10000 + steps] * len(indices), indices)
        if steps == 0 and preempt_after_first_chunk:
            scheduler.running.remove(seq)
            scheduler.preempt(seq)
            assert not scheduler.block_manager.used_block_ids
        steps += 1
        assert steps <= 6
    assert seq.num_completion_tokens == 3
    assert len(scheduler.block_manager.free_block_ids) == config.num_kvcache_blocks
    assert all(block.ref_count == 0 for block in scheduler.block_manager.blocks)


def test_failed_second_runner_preserves_active_block_size(runtime):
    runner, config = make_runner(runtime, 256)
    scheduler = runtime.scheduler(config)
    seq = runtime.sequence(list(range(300)), runtime.params(max_tokens=2))
    scheduler.add(seq)
    seqs = scheduler.schedule()
    prepare_inputs(runtime, runner, seqs)
    scheduler.postprocess(seqs, [10000], [0])

    other_config = SimpleNamespace(**vars(config))
    other_config.kvcache_block_size = 512
    failure = ValueError("trying to initialize the default process group twice!")
    with patch.object(runtime.runner.dist, "init_process_group", side_effect=failure):
        with pytest.raises(ValueError) as caught:
            runtime.runner.ModelRunner(other_config, 0, [])
    assert caught.value is failure
    assert runtime.sequence.block_size == 256
    seqs = scheduler.schedule()
    prepare_inputs(runtime, runner, seqs)
    scheduler.postprocess(seqs, [10001], [0])
    assert scheduler.is_finished()


@pytest.mark.parametrize("block_sizes", [(512, 256), (1024, 512)])
def test_successive_runner_initializations_reset_block_size(runtime, block_sizes):
    for block_size in block_sizes:
        runner, config = make_runner(runtime, block_size)
        scheduler = runtime.scheduler(config)
        seq = runtime.sequence(list(range(300)), runtime.params(max_tokens=2))
        scheduler.add(seq)
        for step in range(2):
            seqs = scheduler.schedule()
            prepare_inputs(runtime, runner, seqs)
            scheduler.postprocess(seqs, [10000 + step], [0])
        assert scheduler.is_finished()
        assert not scheduler.block_manager.used_block_ids
