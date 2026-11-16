import os
import sys
from importlib.machinery import PathFinder
from importlib.util import spec_from_file_location
from pathlib import Path


class DeterministicNPU:
    def find_spec(self, name, path=None, target=None):
        if name != "torch_npu":
            return None
        spec = PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        execute = spec.loader.exec_module

        def configure(module):
            execute(module)
            module.npu.set_deterministic_level(3)
            level = module.npu._get_deterministic_level()
            if level != 3:
                raise RuntimeError(f"Expected NPU deterministic level 3, got {level}")
            print(f"GLM_LEVEL3 pid={os.getpid()} level={level}", flush=True)

        spec.loader.exec_module = configure
        return spec


class BatchInvariantMean:
    def find_spec(self, name, path=None, target=None):
        if name == "sglang.srt.hardware_backend.npu.batch_invariant_ops.npu_batch_invariant_ops":
            return spec_from_file_location(name, Path(__file__).with_name("ascend_bi_ops.py"))
        if (
            name == "sglang.srt.layers.attention.linear.kda_backend"
            and os.getenv("GLM53_BI_KDA_MODE") == "per-sequence"
        ):
            return spec_from_file_location(name, Path(__file__).with_name("sglang_kda_per_sequence.py"))
        if (
            name == "slime_plugins.models.glm5_next.attention"
            and os.getenv("GLM53_SHARED_TRAIN_KDA") == "1"
        ):
            spec = PathFinder.find_spec(name, path)
            filename = spec.origin

            def install_shared_kda(module):
                source = Path(filename).read_text()
                anchor = '"cu_seqlens_cpu": getattr(packed_seq_params, "cu_seqlens_q_cpu", None)'
                assert source.count(anchor) == 1
                source = source.replace(anchor, anchor + ', "raw_beta": beta_logits')
                exec(compile(source, filename, "exec"), module.__dict__)
                from shared_training_kda import install
                install(module)

            spec.loader.exec_module = install_shared_kda
            return spec
        if name == "sglang.srt.models.glm5_next" and os.getenv("GLM53_LAYER_TRACE") == "1":
            spec = PathFinder.find_spec(name, path)
            execute = spec.loader.exec_module

            def install(module):
                execute(module)
                from parity_layer_trace import install_inference
                install_inference(module.Glm5NextModel)

            spec.loader.exec_module = install
            return spec
        return None


sys.meta_path[:0] = [DeterministicNPU(), BatchInvariantMean()]

