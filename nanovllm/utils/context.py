from dataclasses import dataclass
import torch


@dataclass
class Context:
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    seq_need_compute_logits: torch.Tensor | None = None


    @property

    def is_pure_decode(self) -> bool:

        """True when this is a decode-only batch with no prefill tokens.

        During CUDA graph capture and replay, set_context() is called without

        cu_seqlens_q (it remains None). At runtime, _is_decode_only() in

        ModelRunner guarantees graph replay only happens for pure-decode batches,

        so cu_seqlens_q being None is a reliable signal for the attention layer

        to select the graph-compatible flash_attn_with_kvcache path.

        """

        return self.cu_seqlens_q is None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, seq_need_compute_logits=None):
    global _CONTEXT
    _CONTEXT = Context(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, seq_need_compute_logits)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
