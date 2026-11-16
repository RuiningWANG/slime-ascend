import json
import os
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

import torch


_capture = ContextVar("glm53_layer_capture", default=None)
_layer = ContextVar("glm53_layer_index", default=None)


def phase():
    return Path(os.environ["GLM53_PHASE_DIR"])


def _first(value):
    return value[0] if isinstance(value, tuple) else value


def _record(name, value, axis=0):
    trace = _capture.get()
    value = _first(value)
    if trace is None or not isinstance(value, torch.Tensor) or value.ndim <= axis:
        return
    start, end = trace["range"]
    rows = value.movedim(axis, 0)
    if rows.shape[0] < end:
        return
    rows = rows[start:end]
    if trace["engine"] == "megatron" and value.ndim > 2 and axis == 0 and rows.shape[1] == 1:
        rows = rows.squeeze(1)
    trace["layers"][name] = rows.detach().cpu().contiguous()


def _install_layers(layers, inference):
    for index, layer in enumerate(layers):
        if getattr(layer, "_glm53_trace", False):
            continue
        layer._glm53_trace = True
        attention = layer.self_attn if inference else layer.self_attention
        original = layer.forward

        def layer_wrapper(method, index):
            @wraps(method)
            def wrapped(*args, **kwargs):
                if _capture.get() is None:
                    return method(*args, **kwargs)
                marker = _layer.set(index)
                try:
                    hidden = kwargs.get("hidden_states", args[int(inference)] if len(args) > int(inference) else None)
                    _record(f"{index}/layer-input", hidden)
                    output = method(*args, **kwargs)
                    _record(f"{index}/layer-output", output)
                    return output
                finally:
                    _layer.reset(marker)
            return wrapped

        layer.forward = layer_wrapper(original, index)

        def record_attention_input(_module, args, kwargs, index=index):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            _record(f"{index}/attention-input", hidden)

        def record_attention_output(_module, args, output, index=index):
            _record(f"{index}/attention-output", output)

        attention.register_forward_pre_hook(record_attention_input, with_kwargs=True)
        attention.register_forward_hook(record_attention_output)
        if inference and hasattr(getattr(attention, "attn", None), "register_forward_hook"):
            def record_recurrence(_module, args, output, index=index):
                _record(f"{index}/kda-core", output, axis=1)
            attention.attn.register_forward_hook(record_recurrence)


def _run_trace(original, args, kwargs, *, engine, trace, path):
    marker = _capture.set(trace)
    try:
        result = original(*args, **kwargs)
        torch.save(trace, path)
        print(f"GLM53_LAYER_TRACE {engine} file={path} captures={len(trace['layers'])}", flush=True)
        return result
    finally:
        _capture.reset(marker)


def install_inference(cls):
    original = cls.forward

    @wraps(original)
    def forward(self, input_ids, positions, forward_batch, *args, **kwargs):
        flag = phase() / "trace-active"
        if not flag.is_file() or forward_batch.batch_size != 1:
            return original(self, input_ids, positions, forward_batch, *args, **kwargs)
        target_file = Path(flag.read_text().strip())
        target = json.loads(target_file.read_text())
        n = len(target["tokens"])
        input_ids_actual = input_ids if input_ids is not None else forward_batch.input_ids
        if input_ids_actual is None or input_ids_actual[:n].detach().cpu().tolist() != target["tokens"]:
            return original(self, input_ids, positions, forward_batch, *args, **kwargs)
        _install_layers(self.layers, inference=True)
        trace = {"engine": "sglang", "tokens": target["tokens"], "range": (0, n), "layers": {}}
        return _run_trace(original, (self, input_ids, positions, forward_batch, *args), kwargs,
                          engine="sglang", trace=trace, path=target_file.with_name("sglang.pt"))

    cls.forward = forward


def install_training(args, modules, store_prefix):
    if store_prefix:
        return
    from slime_plugins.models.glm5_next import attention as training_attention
    if not getattr(training_attention.chunk_kda, "_glm53_trace", False):
        original_kda = training_attention.chunk_kda

        @wraps(original_kda)
        def kda(*args, **kwargs):
            result = original_kda(*args, **kwargs)
            layer = _layer.get()
            if layer is not None:
                _record(f"{layer}/kda-core", result, axis=1)
            return result

        kda._glm53_trace = True
        training_attention.chunk_kda = kda

    for root in modules:
        for model in root.modules():
            if type(model).__name__ != "Glm5NextModel" or getattr(model, "_glm53_trace", False):
                continue
            model._glm53_trace = True
            _install_layers(model.decoder.layers, inference=False)
            original = model.forward

            def wrap(method):
                @wraps(method)
                def forward(input_ids, *pos, **kwargs):
                    target_file = phase() / "trace-target.json"
                    if not target_file.is_file() or torch.is_grad_enabled():
                        return method(input_ids, *pos, **kwargs)
                    target_file = Path(target_file.read_text().strip())
                    target = json.loads(target_file.read_text())
                    path = target_file.with_name(f"megatron-rank-{torch.distributed.get_rank()}.pt")
                    if path.is_file():
                        return method(input_ids, *pos, **kwargs)
                    packed = kwargs["packed_seq_params"]
                    offsets = packed.cu_seqlens_q.detach().cpu().tolist()
                    token_ids = input_ids.detach().cpu().reshape(-1).tolist()
                    for begin, end in zip(offsets[:-1], offsets[1:], strict=True):
                        if token_ids[begin:end] == target["tokens"]:
                            trace = {"engine": "megatron", "tokens": target["tokens"],
                                     "range": (begin, end), "layers": {}}
                            return _run_trace(method, (input_ids, *pos), kwargs,
                                              engine="megatron", trace=trace, path=path)
                    return method(input_ids, *pos, **kwargs)
                return forward

            model.forward = wrap(original)

