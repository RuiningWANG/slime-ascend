"""Explicit startup configuration; no Python import interception."""
import os

def enabled():
    return os.environ.get("GLM53_ZERO_MISMATCH") == "1"


def configure_npu():
    if os.environ.get("GLM53_ZERO_MISMATCH", "0") != "1":
        return
    import torch_npu
    torch_npu.npu.set_deterministic_level(3)
    if torch_npu.npu._get_deterministic_level() != 3:
        raise RuntimeError("GLM5.3 zero mismatch requires NPU deterministic level 3")
