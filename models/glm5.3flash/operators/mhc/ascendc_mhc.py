"""GLM5-next mHC adapters for the CANN operators used by MindSpeed-Bridge."""

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def load_mhc_ops():
    names = ("mhc_pre_sinkhorn_backward", "mhc_post_backward", "mhc_pre_sinkhorn", "mhc_post")
    # Let the official package discover installed custom operator entry points.
    # Reading modules directly from ASCEND_HOME_PATH would bypass wheel updates.
    try:
        import cann_ops_transformer
    except ImportError as exc:
        raise RuntimeError("Install the CANN 9.2 ops-transformer wheel and its GLM5-next vendor operators") from exc

    # Python wrappers retain the official autograd and keyword interfaces.
    namespace = getattr(cann_ops_transformer, "ops", None)
    missing = [name for name in names if namespace is None or not hasattr(namespace, name)]
    if missing:
        raise RuntimeError(
            f"{cann_ops_transformer.__file__} is missing {missing}. "
            "Install the CANN 9.2 SDK bindings and the matching GLM-5.3 custom wheel."
        )
    return namespace


@lru_cache(maxsize=1)
def _mhc_pre_autograd(pre):
    """Adapt CANN 9.2's residual gradient in this caller, leaving the SDK intact.

    Forward exposes [..., N*N], but the native backward expects [..., N, N].
    Reuse the SDK's forward, saved tensors and backward; only normalize that
    gradient before handing it to the SDK. Reshaping the forward output alone
    cannot fix this: autograd reshapes its gradient back to the flat output.
    """
    sdk_function = pre.__globals__.get("MhcPreSinkhornFunction")
    if sdk_function is None:
        raise RuntimeError("Expected the pinned CANN 9.2 mHC autograd interface")

    class MhcPreSinkhornLayoutFunction(sdk_function):
        @staticmethod
        def backward(ctx, grad_hin, grad_h_post, grad_h_res):
            x = ctx.saved_tensors[0]
            n = x.shape[-2]
            if grad_h_res.shape == (*x.shape[:-2], n * n):
                grad_h_res = grad_h_res.reshape(*x.shape[:-2], n, n).contiguous()
            return sdk_function.backward(ctx, grad_hin, grad_h_post, grad_h_res)

    return MhcPreSinkhornLayoutFunction


def mhc_pre_ascendc(x, weight, scale, bias, n, norm_eps, eps, iterations):
    if n != 4 or iterations != 20:
        raise ValueError("CANN fused mHC requires 4 streams and 20 Sinkhorn iterations")
    if x.shape[-1] % n:
        raise ValueError("mHC stream width must divide the hidden dimension")
    hidden = x.shape[-1] // n
    streams = x.reshape(1, -1, n, hidden).contiguous()
    args = (
        streams,
        weight.float().contiguous(),
        scale.float().contiguous(),
        bias.float().contiguous(),
        n,
        iterations,
        eps,
        norm_eps,
    )
    pre = load_mhc_ops().mhc_pre_sinkhorn
    if torch.is_grad_enabled() and any(t.requires_grad for t in args[:4]):
        collapsed, post, residual = _mhc_pre_autograd(pre).apply(*args)
    else:
        collapsed, post, residual = pre(*args)
    return (
        collapsed.reshape(*x.shape[:-1], hidden),
        post.reshape(*x.shape[:-1], n),
        residual.reshape(*x.shape[:-1], n, n),
    )


def mhc_post_ascendc(y, x, post, residual):
    n = post.shape[-1]
    hidden = y.shape[-1]
    if n != 4:
        raise ValueError("CANN fused mHC requires 4 streams")
    result = load_mhc_ops().mhc_post(
        x.reshape(1, -1, n, hidden).contiguous(),
        residual.reshape(1, -1, n, n).float().contiguous(),
        y.reshape(1, -1, hidden).contiguous(),
        post.reshape(1, -1, n).float().contiguous(),
    )
    if result.numel() != x.numel():
        raise RuntimeError("CANN mHC post returned an incompatible output")
    # Pipeline schedules release output storage and require a viewless tensor.
    # clone preserves autograd while removing the reshape view.
    return result.reshape_as(x).clone()
