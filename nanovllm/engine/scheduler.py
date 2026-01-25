from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.enable_chunked = config.chunked_prefill
        self.max_model_len = config.max_model_len
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        assert len(seq) <= self.max_model_len - 1, "Sequence length exceeds max_model_len"
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        scheduled_running_seqs = []
        scheduled_new_reqs = []
        preempted_seqs = []
        token_budget = self.max_num_batched_tokens

        # schedule from the running queue
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            seq = self.running[req_index]
            num_new_tokens = len(seq) - seq.num_cached_tokens
            if self.enable_chunked:
                num_new_tokens = min(num_new_tokens, token_budget)
            num_new_tokens = min(
                num_new_tokens, self.max_model_len - 1 - seq.num_cached_tokens
            )
            assert num_new_tokens > 0
            while True:
                if self.block_manager.can_append(seq, num_new_tokens):
                    seq.num_new_tokens = num_new_tokens
                    self.block_manager.may_append(seq)
                    break
                preempted_seq = self.running.pop()
                self.preempt(preempted_seq)
                preempted_seqs.append(preempted_seq)
                if len(self.running) == req_index:
                    break
            if len(self.running) == req_index:
                break
            scheduled_running_seqs.append(seq)
            token_budget -= seq.num_new_tokens
            req_index += 1
        
        # schedule from the waiting queue
        if not preempted_seqs:
            while self.waiting and token_budget > 0 and len(self.running) < self.max_num_seqs:
                seq = self.waiting[0]
                assert not seq.block_table
                num_new_computed_tokens_in_used, num_new_computed_tokens_in_free, num_new_tokens = \
                    self.block_manager.get_token_layout(seq)
                if self.enable_chunked:
                    num_new_tokens = min(num_new_tokens, token_budget)
                assert num_new_tokens > 0
                if num_new_tokens > token_budget or \
                    not self.block_manager.can_allocate(num_new_computed_tokens_in_free + num_new_tokens):
                    break
                seq.num_new_tokens = num_new_tokens
                self.block_manager.allocate(seq)
                assert seq.num_cached_tokens == num_new_computed_tokens_in_free + \
                    num_new_computed_tokens_in_used
                token_budget -= num_new_tokens
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
                scheduled_new_reqs.append(seq)
        
        scheduled_seqs = scheduled_running_seqs + scheduled_new_reqs
        assert scheduled_seqs
        return scheduled_seqs


    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], seq_need_compute_logits) -> list[bool]:
        assert len(token_ids) == len(seq_need_compute_logits)
        for seq_index, token_id in zip(seq_need_compute_logits, token_ids):
            seq = seqs[seq_index]
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or \
                seq.num_completion_tokens == seq.max_tokens or \
                    len(seq) >= self.max_model_len:
                if len(seq) >= self.max_model_len:
                    print(f"Sequence {seq.seq_id} reached max_model_len {self.max_model_len}.")
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
        for seq in seqs:
            if seq.status != SequenceStatus.FINISHED:
                seq.num_cached_tokens = seq.num_cached_tokens + seq.num_new_tokens
                seq.num_new_tokens = 0
