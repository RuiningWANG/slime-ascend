"""Stage the pinned GLM-5.3 ops-transformer bindings for the custom vendor."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def stage(wheel: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        unpacked = Path(temporary)
        subprocess.run([sys.executable, "-m", "wheel", "unpack", str(wheel), "-d", temporary], check=True)
        (root,) = unpacked.iterdir()
        builder = root / "cann_ops_transformer_custom/op_builder/builder.py"
        source = builder.read_bytes()
        original = b"from cann_ops_transformer.utils.arg_check import wrap_op_module"
        if source.count(original) != 1:
            raise ValueError(f"Expected one pinned arg_check import in {builder.name}")
        builder.write_bytes(source.replace(original, b"from ..utils.arg_check import wrap_op_module"))
        # wheel pack regenerates RECORD while preserving entry points and package metadata.
        subprocess.run([sys.executable, "-m", "wheel", "pack", str(root), "-d", str(output)], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    stage(args.wheel, args.output)
