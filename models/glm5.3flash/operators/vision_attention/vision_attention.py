"""Inference-aligned visual attention with an explicit training backward."""

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from transformers.models.glm_ocr.modeling_glm_ocr import (
    GlmOcrVisionAttention,
    apply_rotary_pos_emb_vision,
)

from .packed_sequence import packed_boundaries


class _VisionFIA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, ends, scale):
        import torch_npu

        output = torch_npu.npu_fused_infer_attention_score(
            query=q,
            key=k,
            value=v,
            actual_seq_lengths=ends,
            actual_seq_lengths_kv=ends,
            scale=scale,
            num_heads=q.shape[1],
            num_key_value_heads=k.shape[1],
            sparse_mode=0,
            input_layout="TND",
        )[0]
        ctx.save_for_backward(q, k, v)
        ctx.ends = ends
        ctx.scale = scale
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # FIA has no autograd kernel. Recompute the same unmasked, noncausal,
        # zero-dropout attention using training SDPA, then use its backward.
        # This preserves the mathematical Q/K/V gradients without allocating
        # an explicit full-sequence attention matrix in this adapter.
        q, k, v = ctx.saved_tensors
        gradients = [[], [], []]
        start = 0
        for end in ctx.ends:
            with torch.enable_grad():
                inputs = [x[start:end].detach().transpose(0, 1).unsqueeze(0).requires_grad_() for x in (q, k, v)]
                output = F.scaled_dot_product_attention(
                    *inputs,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=ctx.scale,
                )
                grads = torch.autograd.grad(
                    output,
                    inputs,
                    grad_output[start:end].transpose(0, 1).unsqueeze(0),
                )
            for parts, grad in zip(gradients, grads):
                parts.append(grad.squeeze(0).transpose(0, 1))
            start = end
        return *(torch.cat(parts, dim=0) for parts in gradients), None, None


def vision_fia(q, k, v, cu_seqlens, scale):
    """Packed TND attention, using the same FIA arguments as SGLang vision."""
    if q.shape[0] == 0:
        raise ValueError("Vision attention requires at least one token")
    bounds = packed_boundaries(cu_seqlens, q.shape[0])
    ends = tuple(b for a, b in zip(bounds, bounds[1:]) if b > a)
    return _VisionFIA.apply(q.contiguous(), k.contiguous(), v.contiguous(), ends, scale)


class Glm5NextVisionAttention(GlmOcrVisionAttention):
    def forward(self, hidden_states, cu_seqlens, position_embeddings=None, max_seqlen=None, **kwargs):
        # FIA inference has no dropout; preserve the original training semantics
        # if a future configuration enables it, or uses another device/dtype.
        if (
            hidden_states.device.type != "npu"
            or hidden_states.dtype not in (torch.bfloat16, torch.float16)
            or (self.training and self.attention_dropout != 0)
        ):
            return super().forward(hidden_states, cu_seqlens, position_embeddings, max_seqlen, **kwargs)
        length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(length, 3, self.num_heads, -1).unbind(1)
        q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = position_embeddings
        # SGLang stores the visual RoPE cache in the activation dtype.
        # Round the tables at the same boundary before the FP32 rotation;
        # keep the differentiable Q/K path used by Transformers.
        cos, sin = cos.to(q.dtype), sin.to(k.dtype)
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        output = vision_fia(q, k, v, cu_seqlens, self.scaling)
        return self.proj(output.reshape(length, -1).contiguous())
