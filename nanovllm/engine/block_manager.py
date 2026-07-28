from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """
    Blocks (or tokens) layout:
    
    ----------------------------------------------------------------------
    | < computed > | < new_computed > |       < new >       |
    ----------------------------------------------------------------------
    |     < Prefix-cached tokens >    |  < to be computed > |
    ----------------------------------------------------------------------
                                      | < to be allocated > |
    ----------------------------------------------------------------------
                                      |   < to be cached >  |
    ----------------------------------------------------------------------
    
    """
    
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if self.hash_to_block_id.get(block.hash) == block_id:
            self.hash_to_block_id.pop(block.hash, None)
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, num_tokens: int) -> bool:
        """
        Only for seq in the waiting queue.
        """
        return len(self.free_block_ids) >= (num_tokens + self.block_size - 1) // self.block_size

    def get_token_layout(self, seq: Sequence):
        """
        Only for seq in the waiting queue.
        """
        assert not seq.block_table
        num_new_tokens = 0
        num_new_computed_tokens_in_used = 0
        num_new_computed_tokens_in_free = 0
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids or i == seq.num_blocks - 1:
                cache_miss = True
            if cache_miss:
                num_new_tokens += len(token_ids)
            else:
                if block_id in self.used_block_ids:
                    num_new_computed_tokens_in_used += len(token_ids)
                else:
                    num_new_computed_tokens_in_free += len(token_ids)
        return num_new_computed_tokens_in_used, num_new_computed_tokens_in_free, num_new_tokens

    def allocate(self, seq: Sequence):
        """
        Only for seq in the waiting queue.
        """
        assert not seq.block_table
        h = -1
        # allocate new_computed_blocks
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids or i == seq.num_blocks - 1:
                break               # cache miss
            seq.num_cached_tokens += self.block_size
            if block_id in self.used_block_ids:
                block = self.blocks[block_id]
                block.ref_count += 1
            else:
                block = self._allocate_block(block_id)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)
        
        # allocate new_blocks
        for i in range(seq.num_cached_tokens, seq.num_cached_tokens + seq.num_new_tokens, self.block_size):
            token_ids = seq[i: min(i + self.block_size, seq.num_cached_tokens + seq.num_new_tokens)]
            if len(token_ids) != self.block_size:
                h = -1
            elif i != seq.num_cached_tokens:
                h = self.compute_hash(token_ids, h)
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)


    def deallocate(self, seq: Sequence):
        """
        For finished seq or preempted seq in the running queue.
        """
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.num_new_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence, num_new_tokens: int) -> bool:
        """
        Only for seq in the running queue.
        """
        last_computed_block_capacity = self.block_size - (seq.num_cached_tokens % self.block_size)
        if last_computed_block_capacity == self.block_size:
            last_computed_block_capacity = 0
        if (num_new_tokens - last_computed_block_capacity + self.block_size - 1) // self.block_size \
            <= len(self.free_block_ids):
            return True
        return False

    def may_append(self, seq: Sequence):
        """
        Only for seq in the running queue.
        """
        for i in range(
            seq.num_cached_blocks * self.block_size, 
            seq.num_cached_tokens + seq.num_new_tokens, 
            self.block_size
        ):  
            token_ids = seq[i: min(i + self.block_size, seq.num_cached_tokens + seq.num_new_tokens)]
            current_block_id = seq.block_table[i // self.block_size] \
                    if i // self.block_size < len(seq.block_table) else -1
            if current_block_id != -1:
                current_block = self.blocks[current_block_id]
                assert current_block.hash == -1
            if len(token_ids) % self.block_size == 0:
                previous_block_id = seq.block_table[i // self.block_size - 1] if i >= self.block_size else -1
                prefix = self.blocks[previous_block_id].hash if previous_block_id != -1 else -1
                h = self.compute_hash(token_ids, prefix)
                if current_block_id == -1:
                    block_id = self.free_block_ids[0]
                    current_block = self._allocate_block(block_id)
                    seq.block_table.append(block_id)
                current_block.update(h, token_ids)
                self.hash_to_block_id[h] = current_block.block_id
            elif current_block_id == -1:
                    block_id = self.free_block_ids[0]
                    self._allocate_block(block_id)
                    seq.block_table.append(block_id)
