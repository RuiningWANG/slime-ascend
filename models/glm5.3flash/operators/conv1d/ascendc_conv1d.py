"""AscendC causal Conv1d forward/backward for GLM5-next training.

Adapted from MindSpeed-Bridge's GLM-5.3-Flash kda_conv1d_fused.py.
Inputs retain Slime's [B, T, C] and [C, W] contract. SiLU is fused before
the output cast, matching inference. CP gather/scatter is owned
by npu_backend, so packed boundaries here describe complete sequences.
"""

import os
from functools import lru_cache

import torch

from .packed_sequence import packed_boundaries


@lru_cache(maxsize=1)
def _load_ops():
    try:
        from fla_npu.ops.ascendc import npu_causal_conv1d, npu_causal_conv1d_bwd
        from fla_npu.ops.ascendc._runtime import runtime
    except ImportError as exc:
        raise ImportError(
            "Conv1d requires the FLA-NPU forward and backward bindings from bde95747. "
            "Install the wheel for your SoC with scripts/quick_install.sh --glm5.3-flash a3 (or a5)."
        ) from exc

    acl = runtime()
    symbols = (
        "aclnnCausalConv1d",
        "aclnnCausalConv1dGetWorkspaceSize",
        "aclnnCausalConv1dBwd",
        "aclnnCausalConv1dBwdGetWorkspaceSize",
    )
    try:
        for name in symbols:
            acl.symbol(name)
    except (AttributeError, OSError) as exc:
        raise RuntimeError(
            "FLA-NPU is missing the CANN causal Conv1d forward/backward operators. "
            "Install the GLM5.3Flash FLA-NPU wheel and source its vendor set_env.bash. "
            f"ASCEND_CUSTOM_OPP_PATH={os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')}"
        ) from exc
    return npu_causal_conv1d, npu_causal_conv1d_bwd


class _CausalConv1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, boundaries, activation_mode):
        npu_causal_conv1d, _ = _load_ops()

        width, channels = weight.shape
        sequences = len(boundaries) - 1 if boundaries is not None else x.shape[0]
        states = x.new_zeros((sequences, width - 1, channels))
        output = npu_causal_conv1d(
            x=x,
            weight=weight,
            bias=None,
            conv_states=states,
            query_start_loc=boundaries,
            activation_mode=activation_mode,
            run_mode=0,
        )
        ctx.save_for_backward(x, weight)
        ctx.boundaries = boundaries
        ctx.activation_mode = activation_mode
        # FLA 26.10 uses BSND for logical [B, T, C] Conv1d inputs;
        # this enum does not require reshaping them to four dimensions.
        ctx.layout = "TND" if x.ndim == 2 else "BSND"
        return output

    @staticmethod
    def backward(ctx, grad_output):
        npu_causal_conv1d, npu_causal_conv1d_bwd = _load_ops()

        x, weight = ctx.saved_tensors
        # CANN backward consumes the pre-SiLU value in the input dtype, not
        # the fused output. Recompute it only for backward to avoid retaining
        # another activation tensor throughout the forward pass.
        pre_activation = None
        if ctx.activation_mode:
            sequences = len(ctx.boundaries) - 1 if ctx.boundaries is not None else x.shape[0]
            states = x.new_zeros((sequences, weight.shape[0] - 1, weight.shape[1]))
            pre_activation = npu_causal_conv1d(
                x=x,
                weight=weight,
                bias=None,
                conv_states=states,
                query_start_loc=ctx.boundaries,
                activation_mode=0,
                run_mode=0,
            )
        dx, dw, _, _ = npu_causal_conv1d_bwd(
            x=x,
            y=pre_activation,
            weight=weight,
            dy=grad_output.contiguous(),
            initial_state=None,
            dht=None,
            query_start_loc=ctx.boundaries,
            activation=ctx.activation_mode,
            input_layout=ctx.layout,
        )
        return dx, dw, None, None


def causal_conv1d_ascendc(x, weight, activation="silu", cu_seqlens=None):
    """Apply packed or dense causal depthwise convolution without recurrent state."""
    if x.ndim != 3 or weight.ndim != 2 or x.shape[-1] != weight.shape[0]:
        raise ValueError("Expected x=[B,T,C] and weight=[C,W]")
    if activation not in (None, "silu"):
        raise ValueError(f"Unsupported convolution activation: {activation}")
    if x.device.type != "npu" or x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("AscendC Conv1d requires BF16/FP16 NPU inputs")
    boundaries = None
    logical_x = x.contiguous()
    if cu_seqlens is not None:
        if x.shape[0] != 1:
            raise ValueError("Packed AscendC Conv1d requires batch size 1")
        values = packed_boundaries(cu_seqlens, x.shape[1])
        # Empty packed sequences have no output or gradient; the kernel requires
        # strictly increasing boundaries.
        boundaries = [values[0]] + [b for a, b in zip(values, values[1:]) if b > a]
        logical_x = logical_x.squeeze(0)
    logical_weight = weight.transpose(0, 1).contiguous().to(x.dtype)
    output = _CausalConv1d.apply(logical_x, logical_weight, boundaries, int(activation == "silu"))
    if output.ndim == 2:
        output = output.unsqueeze(0)
    return output
