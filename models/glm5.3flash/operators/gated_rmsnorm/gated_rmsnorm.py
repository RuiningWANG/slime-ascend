"""Sigmoid-gated RMSNorm for GLM5-next on NPU."""

import torch
from torch.autograd.function import once_differentiable


class _GatedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate, weight, eps):
        from sglang.kernels.ops.attention.fla.fused_norm_gate import (
            layer_norm_gated_fwd,
        )

        if x.shape != gate.shape or weight.shape != (x.shape[-1],):
            raise ValueError("Gated RMSNorm requires matching x/gate and one weight per channel")
        shape = x.shape
        xx = x.reshape(-1, shape[-1]).contiguous()
        gg = gate.reshape_as(xx).contiguous()
        # Passing out_dtype allocates a separate output. The inference default
        # aliases x, which would corrupt saved inputs and checkpoint recomputation.
        y, _, rstd, _ = layer_norm_gated_fwd(
            xx,
            gg,
            weight,
            None,
            activation="sigmoid",
            eps=eps,
            out_dtype=x.dtype,
            is_rms_norm=True,
        )
        ctx.save_for_backward(xx, gg, weight, rstd)
        ctx.shape = shape
        return y.reshape(shape)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, gate, weight, rstd = ctx.saved_tensors
        xhat = x.float() * rstd[:, None]
        sigmoid = gate.float().sigmoid()
        dy = grad_output.reshape_as(x).float()
        weighted_dy = dy * weight.float() * sigmoid
        dx = (weighted_dy - xhat * (weighted_dy * xhat).mean(-1, keepdim=True)) * rstd[:, None]
        dgate = dy * xhat * weight.float() * sigmoid * (1.0 - sigmoid)
        dweight = (dy * xhat * sigmoid).sum(dim=0)
        return (
            dx.to(x.dtype).reshape(ctx.shape),
            dgate.to(gate.dtype).reshape(ctx.shape),
            dweight.to(weight.dtype),
            None,
        )


def gated_rms_norm(x, gate, weight, eps=1e-5):
    """Use only the inference fused forward on NPU.

    Supports first-order training gradients and noncontiguous input views.
    Higher-order derivatives of the NPU custom backward are not supported.
    """
    if any(t.device.type != "npu" for t in (x, gate, weight)):
        raise ValueError("Fused gated RMSNorm requires NPU inputs and weight")
    if not x.numel():
        raise ValueError("Fused gated RMSNorm requires nonempty input")
    return _GatedRMSNorm.apply(x, gate, weight, eps)


class FusedRMSNormGated(torch.nn.Module):
    """NPU fused sigmoid-gated RMSNorm with first-order training gradients."""

    def __init__(self, hidden_size, eps, activation="sigmoid"):
        super().__init__()
        if activation != "sigmoid":
            raise ValueError("GLM5-next gated norm expects sigmoid")
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x, gate):
        return gated_rms_norm(x, gate, self.weight, self.eps)
