import os
import sys
from importlib.machinery import PathFinder


class AscendLevel3:
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "torch_npu":
            return None
        spec = PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        original = spec.loader.exec_module

        def execute(module):
            original(module)
            module.npu.set_deterministic_level(3)
            level = module.npu._get_deterministic_level()
            if level != 3:
                raise RuntimeError(f"Expected NPU deterministic level 3, got {level}")
            print(f"GLM_LEVEL3 pid={os.getpid()} level={level}", flush=True)

        spec.loader.exec_module = execute
        return spec


sys.meta_path.insert(0, AscendLevel3())
