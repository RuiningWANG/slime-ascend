# Copyright © 2026 Huawei Technologies Co., Ltd.
# Based on flash-linear-attention: https://github.com/fla-org/flash-linear-attention
#
# This file contains code copied and/or modified from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li


import contextlib
import functools
import inspect
import os
from collections.abc import Callable

import torch
import triton
from slime.utils import accelerator

FLA_CACHE_RESULTS = os.getenv("FLA_CACHE_RESULTS", "1") == "1"

SUPPORTS_AUTOTUNE_CACHE = "cache_results" in inspect.signature(triton.autotune).parameters

autotune_cache_kwargs = {"cache_results": FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}


def input_guard(
    fn: Callable[..., torch.Tensor] | None = None,
    *,
    no_guard_contiguous: bool | list[str] = False,
) -> Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]] | Callable[..., torch.Tensor]:
    def decorator(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            skip_params = set()
            if isinstance(no_guard_contiguous, list):
                skip_params = set(no_guard_contiguous)

            processed_args = []
            for i, arg in enumerate(args):
                if i < len(param_names):
                    param_name = param_names[i]
                else:
                    param_name = f"__arg_{i}"

                if isinstance(arg, torch.Tensor):
                    if no_guard_contiguous is True or param_name in skip_params:
                        processed_args.append(arg)
                    else:
                        processed_args.append(arg.contiguous())
                else:
                    processed_args.append(arg)

            processed_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    if no_guard_contiguous is True or k in skip_params:
                        processed_kwargs[k] = v
                    else:
                        processed_kwargs[k] = v.contiguous()
                else:
                    processed_kwargs[k] = v

            tensor = None
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    tensor = arg
                    break
            if tensor is None:
                for value in kwargs.values():
                    if isinstance(value, torch.Tensor):
                        tensor = value
                        break

            if tensor is not None:
                ctx = custom_device_ctx(tensor.device.index)
            else:
                ctx = contextlib.nullcontext()

            with ctx:
                return fn(*processed_args, **processed_kwargs)

        return wrapper

    if fn is not None:
        return decorator(fn)

    return decorator


device = accelerator.device_type()
device_torch_lib = getattr(torch, device)
autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type=device)
autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type=device)


def custom_device_ctx(index: int):
    return device_torch_lib.device(index)
