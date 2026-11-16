import json
from pathlib import Path
import sys

import torch


folder = Path(sys.argv[1])
reference = torch.load(folder / "sglang.pt", map_location="cpu", weights_only=True)
ranks = sorted(folder.glob("megatron-rank-*.pt"))
assert ranks, "Megatron did not trace the selected sequence"
report = []
for path in ranks:
    candidate = torch.load(path, map_location="cpu", weights_only=True)
    assert candidate["tokens"] == reference["tokens"]
    layers = {}
    for key, left in reference["layers"].items():
        right = candidate["layers"].get(key)
        if right is None:
            continue
        if left.shape != right.shape or left.dtype != right.dtype:
            layers[key] = {"incompatible": [list(left.shape), str(left.dtype), list(right.shape), str(right.dtype)]}
            continue
        different = left.contiguous().view(torch.uint8) != right.contiguous().view(torch.uint8)
        different = different.reshape(left.shape[0], -1).any(dim=1)
        layers[key] = {
            "tokens": int(left.shape[0]),
            "different_token_rows": int(different.sum()),
            "first_different_position": int(different.nonzero()[0].item()) if different.any() else None,
            "max_abs_diff": float((left.float() - right.float()).abs().max()),
        }
    report.append({"rank_file": path.name, "comparisons": layers})
(folder / "layer-diff.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))

