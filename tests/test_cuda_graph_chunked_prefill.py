"""
tests/test_cuda_graph_chunked_prefill.py

Unit tests for CUDA graph + chunked prefill compatibility (PR #1).

Strategy: avoid the nanovllm/__init__.py -> llm -> llm_engine -> transformers
-> accelerate -> torch.distributed chain entirely. Instead:

  1. Inline the three tiny helper classes (SamplingParams, Context, Sequence)
     directly in this file -- they have zero external dependencies beyond stdlib.
  2. Inline _is_decode_only and the use_graph gate as pure Python so the
     logic tests need no mocking at all.
  3. For the ModelRunner integration tests, load model_runner.py directly via
     importlib AFTER patching every heavy dependency, so the real
     torch.distributed is never replaced.

All 16 tests run with no GPU, no CUDA, no flash-attn, no transformers.
"""

import sys
import types
import unittest
import importlib.util
import pathlib
from copy import copy
from dataclasses import dataclass
from enum import Enum, auto
from itertools import count
from unittest.mock import MagicMock

import torch


# ---------------------------------------------------------------------------
# 1. Inline minimal helpers (zero imports from nanovllm)
# ---------------------------------------------------------------------------

@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False


@dataclass
class Context:
    cu_seqlens_q: object = None
    cu_seqlens_k: object = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: object = None
    context_lens: object = None
    block_tables: object = None
    seq_need_compute_logits: object = None


_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0,
                max_seqlen_k=0, slot_mapping=None, context_lens=None,
                block_tables=None, seq_need_compute_logits=None):
    global _CONTEXT
    _CONTEXT = Context(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                       slot_mapping, context_lens, block_tables,
                       seq_need_compute_logits)


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids, sampling_params=None):
        if sampling_params is None:
            sampling_params = SamplingParams()
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_new_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def num_context_tokens(self):
        return self.num_cached_tokens + self.num_new_tokens

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size


# ---------------------------------------------------------------------------
# 2. The two methods under test, inlined verbatim from model_runner.py
# ---------------------------------------------------------------------------

def _is_decode_only(seqs) -> bool:
    return all(seq.num_new_tokens == 1 for seq in seqs)


def _use_graph(enforce_eager: bool, input_ids_size: int, seqs) -> bool:
    return (
        not enforce_eager
        and input_ids_size <= 512
        and seqs is not None
        and _is_decode_only(seqs)
    )


# ---------------------------------------------------------------------------
# 3. Load ModelRunner without triggering the heavy import chain
# ---------------------------------------------------------------------------

def _load_model_runner():
    repo_root = pathlib.Path(__file__).parent.parent

    # Build stub modules for everything model_runner imports at module level.
    # Critically: we do NOT stub torch.distributed -- the real one stays.
    ctx_stub = types.ModuleType("nanovllm.utils.context")
    ctx_stub.Context = Context
    ctx_stub.get_context = get_context
    ctx_stub.set_context = set_context
    ctx_stub.reset_context = reset_context

    seq_stub = types.ModuleType("nanovllm.engine.sequence")
    seq_stub.Sequence = Sequence
    seq_stub.SequenceStatus = SequenceStatus

    stubs = {
        "flash_attn":                  MagicMock(),
        "triton":                      MagicMock(),
        "triton.language":             MagicMock(),
        "nanovllm.utils.context":      ctx_stub,
        "nanovllm.engine.sequence":    seq_stub,
        "nanovllm.layers.sampler":     MagicMock(),
        "nanovllm.models.qwen3":       MagicMock(),
        "nanovllm.utils.loader":       MagicMock(),
        "nanovllm.config":             MagicMock(),
    }

    saved = {name: sys.modules.get(name) for name in stubs}
    for name, stub in stubs.items():
        sys.modules[name] = stub

    try:
        spec = importlib.util.spec_from_file_location(
            "nanovllm.engine.model_runner",
            repo_root / "nanovllm" / "engine" / "model_runner.py",
        )
        mod = importlib.util.module_from_spec(spec)
        # Inject a mock dist directly into the module's namespace so the
        # import at the top of model_runner.py ("import torch.distributed as dist")
        # is shadowed before any method calls it.
        mod.dist = MagicMock()
        sys.modules["nanovllm.engine.model_runner"] = mod
        spec.loader.exec_module(mod)
        return mod.ModelRunner
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


ModelRunner = _load_model_runner()


# ---------------------------------------------------------------------------
# 4. Shared test helpers
# ---------------------------------------------------------------------------

def make_seq(num_new_tokens: int, num_cached_tokens: int = 0) -> Sequence:
    token_ids = [0] * max(num_cached_tokens + num_new_tokens, 1)
    seq = Sequence(token_ids)
    seq.num_cached_tokens = num_cached_tokens
    seq.num_new_tokens = num_new_tokens
    seq.block_table = [0]
    return seq


def make_runner(enforce_eager: bool = False) -> ModelRunner:
    hidden = 64
    max_bs = 32
    max_num_blocks = 16

    runner = object.__new__(ModelRunner)
    runner.enforce_eager = enforce_eager
    runner.world_size = 1
    runner.rank = 0
    runner.block_size = 256

    mock_model = MagicMock()
    mock_model.return_value = torch.zeros(max_bs, hidden)
    mock_model.compute_logits = MagicMock(
        side_effect=lambda x: torch.zeros(x.size(0), 32000)
    )
    runner.model = mock_model
    runner.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    runner.graph_vars = dict(
        input_ids=torch.zeros(max_bs, dtype=torch.int64),
        positions=torch.zeros(max_bs, dtype=torch.int64),
        slot_mapping=torch.full((max_bs,), -1, dtype=torch.int32),
        context_lens=torch.zeros(max_bs, dtype=torch.int32),
        block_tables=torch.zeros(max_bs, max_num_blocks, dtype=torch.int32),
        outputs=torch.zeros(max_bs, hidden),
    )
    runner.graphs = {bs: MagicMock() for bs in runner.graph_bs}
    return runner


def set_decode_ctx(bs: int):
    reset_context()
    set_context(
        slot_mapping=torch.zeros(bs, dtype=torch.int32),
        context_lens=torch.ones(bs, dtype=torch.int32) * 5,
        block_tables=torch.zeros(bs, 4, dtype=torch.int32),
    )


# ===========================================================================
# Tests
# ===========================================================================

class TestIsDecodeOnly(unittest.TestCase):

    def test_single_decode_seq(self):
        self.assertTrue(_is_decode_only([make_seq(1, 10)]))

    def test_batch_all_decode(self):
        self.assertTrue(_is_decode_only([make_seq(1, i + 5) for i in range(8)]))

    def test_single_first_prefill(self):
        self.assertFalse(_is_decode_only([make_seq(50, 0)]))

    def test_chunked_prefill_mid_chunk(self):
        self.assertFalse(_is_decode_only([make_seq(32, 64)]))

    def test_mixed_prefill_and_decode(self):
        self.assertFalse(_is_decode_only([make_seq(1, 20), make_seq(1, 15), make_seq(16, 0)]))

    def test_last_prefill_chunk(self):
        self.assertFalse(_is_decode_only([make_seq(2, 62)]))

    def test_all_prefill_batch(self):
        self.assertFalse(_is_decode_only([make_seq(64, 0) for _ in range(4)]))


class TestUseGraphGate(unittest.TestCase):

    def test_enforce_eager_blocks_graph(self):
        self.assertFalse(_use_graph(True, 4, [make_seq(1, 10)] * 4))

    def test_decode_only_gets_graph(self):
        self.assertTrue(_use_graph(False, 4, [make_seq(1, 10)] * 4))

    def test_prefill_blocks_graph(self):
        self.assertFalse(_use_graph(False, 1, [make_seq(32, 0)]))

    def test_mixed_blocks_graph(self):
        self.assertFalse(_use_graph(False, 2, [make_seq(1, 10), make_seq(16, 0)]))

    def test_large_bs_blocks_graph(self):
        self.assertFalse(_use_graph(False, 513, [make_seq(1, 5)] * 4))

    def test_seqs_none_blocks_graph(self):
        self.assertFalse(_use_graph(False, 1, None))


class TestRunModelRouting(unittest.TestCase):

    def tearDown(self):
        reset_context()

    def test_enforce_eager_always_eager(self):
        runner = make_runner(enforce_eager=True)
        set_decode_ctx(4)
        runner.run_model(torch.zeros(4, dtype=torch.int64),
                         torch.zeros(4, dtype=torch.int64),
                         [make_seq(1, 10)] * 4)
        runner.model.assert_called_once()
        for g in runner.graphs.values():
            g.replay.assert_not_called()

    def test_decode_only_uses_graph(self):
        runner = make_runner(enforce_eager=False)
        bs = 4
        set_decode_ctx(bs)
        runner.run_model(torch.zeros(bs, dtype=torch.int64),
                         torch.zeros(bs, dtype=torch.int64),
                         [make_seq(1, 10)] * bs)
        runner.model.assert_not_called()
        bucket = next(x for x in runner.graph_bs if x >= bs)
        runner.graphs[bucket].replay.assert_called_once()

    def test_graph_bucket_selection(self):
        runner = make_runner(enforce_eager=False)
        bs = 5
        set_decode_ctx(bs)
        runner.run_model(torch.zeros(bs, dtype=torch.int64),
                         torch.zeros(bs, dtype=torch.int64),
                         [make_seq(1, 10)] * bs)
        runner.graphs[8].replay.assert_called_once()
        runner.graphs[4].replay.assert_not_called()

    def test_prefill_uses_eager(self):
        runner = make_runner(enforce_eager=False)
        set_decode_ctx(1)
        runner.run_model(torch.zeros(1, dtype=torch.int64),
                         torch.zeros(1, dtype=torch.int64),
                         [make_seq(32, 0)])
        runner.model.assert_called_once()
        for g in runner.graphs.values():
            g.replay.assert_not_called()

    def test_mixed_batch_uses_eager(self):
        runner = make_runner(enforce_eager=False)
        set_decode_ctx(2)
        runner.run_model(torch.zeros(2, dtype=torch.int64),
                         torch.zeros(2, dtype=torch.int64),
                         [make_seq(1, 20), make_seq(16, 0)])
        runner.model.assert_called_once()
        for g in runner.graphs.values():
            g.replay.assert_not_called()

    def test_large_batch_uses_eager(self):
        runner = make_runner(enforce_eager=False)
        set_decode_ctx(4)
        runner.run_model(torch.zeros(513, dtype=torch.int64),
                         torch.zeros(513, dtype=torch.int64),
                         [make_seq(1, 5)] * 513)
        runner.model.assert_called_once()
        for g in runner.graphs.values():
            g.replay.assert_not_called()

    def test_seqs_none_uses_eager(self):
        runner = make_runner(enforce_eager=False)
        set_decode_ctx(1)
        runner.run_model(torch.zeros(1, dtype=torch.int64),
                         torch.zeros(1, dtype=torch.int64),
                         seqs=None)
        runner.model.assert_called_once()
        for g in runner.graphs.values():
            g.replay.assert_not_called()


class TestBlockTablesZeroed(unittest.TestCase):

    def tearDown(self):
        reset_context()

    def _ctx(self, bs, block_val):
        reset_context()
        set_context(
            slot_mapping=torch.zeros(bs, dtype=torch.int32),
            context_lens=torch.ones(bs, dtype=torch.int32),
            block_tables=torch.full((bs, 2), block_val, dtype=torch.int32),
        )

    def test_stale_block_tables_are_cleared(self):
        runner = make_runner(enforce_eager=False)

        # Step 1: large batch (bs=8), block_id=99
        self._ctx(8, 99)
        runner.run_model(torch.zeros(8, dtype=torch.int64),
                         torch.zeros(8, dtype=torch.int64),
                         [make_seq(1, 5)] * 8)

        # Step 2: small batch (bs=2), block_id=7
        self._ctx(2, 7)
        runner.run_model(torch.zeros(2, dtype=torch.int64),
                         torch.zeros(2, dtype=torch.int64),
                         [make_seq(1, 5)] * 2)

        bt = runner.graph_vars["block_tables"]
        self.assertTrue((bt[:2, :2] == 7).all(),
                        "Active rows should hold current block ids")
        self.assertTrue((bt[2:] == 0).all(),
                        "Padded rows must be zeroed, not carry stale ids")


class TestSchedulingSimulation(unittest.TestCase):

    def test_eager_then_graph_transition(self):
        chunk_size = 4
        prompt_len = 8

        steps = [
            (chunk_size, [make_seq(chunk_size, 0)]),
            (chunk_size, [make_seq(chunk_size, chunk_size)]),
            (1,          [make_seq(1, prompt_len)]),
            (1,          [make_seq(1, prompt_len + 1)]),
        ]

        eager = sum(1 for bs, seqs in steps if not _use_graph(False, bs, seqs))
        graph = sum(1 for bs, seqs in steps if _use_graph(False, bs, seqs))

        self.assertEqual(eager, 2, "Prefill steps should be eager")
        self.assertEqual(graph, 2, "Decode steps should use CUDA graph")


if __name__ == "__main__":
    unittest.main(verbosity=2)