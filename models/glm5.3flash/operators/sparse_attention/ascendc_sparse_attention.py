"""SparseFlashAttention autograd for GLM5-next's absorbed, shared KV.

The sparse indices already encode causality and packed-sequence boundaries.
Use sparse_mode=0 in both directions, preserving duplicate selected slots.
A3 CANN 9.2 KV-merge returns dK+dV in d_key and an empty d_value.
A5 uses explicit shared K/V and adds their separate gradients; masked selection
slots align the A5 backward width without changing real sparse choices.
A3 requires 64 auxiliary RoPE channels. Real keys have zero RoPE, preserving
no-RoPE model scores. Padding maps to a pool of zero KVs with large negative auxiliary scores,
because the A3 backward kernel cannot consume -1 indices. Distributing padding
indices spreads masked selections across KV rows.
"""

import math

import torch
import torch.nn.functional as F

# 512 masked rows require 512 KiB of BF16/FP16 KV storage, excluding RoPE.
_PADDING_KV_ROWS = 512


def uses_explicit_kv(device_name):
    """A5 SparseFlashAttention uses explicit K/V; A3 uses merged KV."""
    if device_name.startswith("Ascend950"):
        return True
    if device_name.startswith("Ascend910_93"):
        return False
    raise RuntimeError(f"GLM5-next sparse attention requires Ascend910_93 or Ascend950, got {device_name}")


def compact_sparse_indices(indices, kv_length):
    """Move valid slots to a prefix, retaining their order and multiplicity."""
    indices = indices.to(dtype=torch.int32)
    valid = (indices >= 0) & (indices < kv_length)
    position = torch.arange(indices.size(-1), device=indices.device, dtype=torch.int32)
    # Unique position keys preserve order without relying on stable-sort support.
    keys = torch.where(valid, position, position + indices.size(-1))
    # Sort position keys in FP32; sparse index values remain int32.
    # The current width is 2112, so every key in [0, 4223] is exact.
    keys = keys.float()
    order = keys.argsort(-1)
    return torch.where(valid, indices, -1).gather(-1, order).contiguous()


class _SparseFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, indices, scaling):
        import torch_npu

        if q.ndim != 3 or kv.ndim != 3 or indices.ndim != 3:
            raise ValueError("SparseFlashAttention expects Q/KV [T,N,D], indices [T,1,K]")
        if q.shape[-1] != 512 or kv.shape[-2:] != (1, 512):
            raise ValueError("SparseFlashAttention requires 512 channels and one shared KV head")
        if not q.shape[0] or not kv.shape[0] or indices.shape[:2] != (q.shape[0], 1) or not indices.shape[-1]:
            raise ValueError("SparseFlashAttention requires nonempty Q/KV and selections [T,1,K]")
        if not math.isfinite(float(scaling)) or scaling <= 0:
            raise ValueError("SparseFlashAttention requires a positive finite softmax scale")
        if q.dtype not in (torch.bfloat16, torch.float16) or kv.dtype != q.dtype:
            raise ValueError("SparseFlashAttention requires matching BF16 or FP16 Q/KV")
        with torch.npu.device(q.device):
            # On A5 use explicit K=V: the installed KV-merge backward returns
            # only dK. Summing the separate gradients preserves shared-KV math.
            explicit_value = uses_explicit_kv(torch.npu.get_device_name(q.device.index))
            query = q.contiguous()
            selected = compact_sparse_indices(indices, kv.size(0))
            # A5 SFAG requires a multiple of 1024 selected slots. Appended -1
            # slots use the masked KV pool below; real selections, including
            # duplicates, retain their order and probability mass.
            padding = (-selected.shape[-1]) % 1024
            if padding and explicit_value:
                selected = F.pad(selected, (0, padding), value=-1)
            # A pooled selection may begin with padding and end with valid keys.
            nonempty = selected.ge(0).any(dim=-1, keepdim=True)
            selected = torch.cat((torch.where(nonempty, selected[..., :1], 0), selected[..., 1:]), -1)
            # S2 must be strictly greater than K: otherwise A3 backward selects
            # a dense-prefix fast path that ignores nonmonotonic sparse indices.
            # Reserve a pool of masked padding rows, never real tokens.
            tokens = max(q.shape[0], kv.shape[0] + _PADDING_KV_ROWS, selected.shape[-1] + 1)
            shared = F.pad(kv, (0, 0, 0, 0, 0, tokens - kv.shape[0])).contiguous()
            q_rope = query.new_zeros((*query.shape[:-1], 64))
            k_rope = shared.new_zeros((*shared.shape[:-1], 64))
            # Avoid invalid GM reads in SFAG for -1 padding. Real key scores
            # remain unchanged; the sentinel probability underflows to zero.
            q_rope[..., 0] = 1
            k_rope[-_PADDING_KV_ROWS:, :, 0] = torch.finfo(kv.dtype).min
            padding_indices = (
                torch.arange(selected.shape[-1], device=selected.device, dtype=torch.int32) % _PADDING_KV_ROWS
                + tokens
                - _PADDING_KV_ROWS
            )
            selected = torch.where(selected < 0, padding_indices, selected)
            q_lengths = torch.tensor([query.shape[0]], dtype=torch.int32, device=q.device)
            kv_lengths = torch.tensor([tokens], dtype=torch.int32, device=q.device)
            output, softmax_max, softmax_sum = torch_npu.npu_sparse_flash_attention(
                query,
                shared,
                shared if explicit_value else None,
                selected,
                float(scaling),
                query_rope=q_rope,
                key_rope=k_rope,
                actual_seq_lengths_query=q_lengths,
                actual_seq_lengths_kv=kv_lengths,
                sparse_block_size=1,
                layout_query="TND",
                layout_kv="TND",
                sparse_mode=0,
                attention_mode=2,
                return_softmax_lse=True,
            )
            ctx.save_for_backward(
                query,
                shared,
                selected,
                output,
                softmax_max,
                softmax_sum,
                q_rope,
                k_rope,
                q_lengths,
                kv_lengths,
                nonempty,
            )
            ctx.explicit_value = explicit_value
            ctx.scaling = float(scaling)
            ctx.kv_tokens = kv.shape[0]
            return torch.where(nonempty, output, 0)

    @staticmethod
    def backward(ctx, dout):
        import torch_npu

        q, kv, selected, output, maximum, total, qr, kr, ql, kl, nonempty = ctx.saved_tensors
        with torch.npu.device(q.device):
            upstream = torch.where(nonempty, dout, 0).contiguous()
            dq, dkv, dv, *_ = torch_npu.npu_sparse_flash_attention_grad(
                q,
                kv,
                kv if ctx.explicit_value else None,
                selected,
                upstream,
                output,
                maximum,
                total,
                ctx.scaling,
                1,
                query_rope=qr,
                key_rope=kr,
                actual_seq_qlen=ql,
                actual_seq_kvlen=kl,
                layout="TND",
                sparse_mode=0,
                attention_mode=2,
            )
            if ctx.explicit_value:
                if dv.shape != dkv.shape:
                    raise RuntimeError("SparseFlashAttention explicit shared K/V requires equal gradient shapes")
                dkv = dkv + dv
            elif dv.numel() != 0:
                raise RuntimeError("SparseFlashAttention KV-merge must return an empty d_value")
            return dq, dkv[: ctx.kv_tokens], None, None


def sparse_attention(q, kv, indices, scaling):
    """Indices must contain global KV offsets with -1 padding in any slot."""
    return _SparseFlashAttention.apply(q, kv, indices, scaling)
