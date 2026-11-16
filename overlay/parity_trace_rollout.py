import json
from pathlib import Path
import os

from batch_gate import generate
from slime.rollout.sglang_rollout import generate_rollout as native_generate


def capture(args, rollout_id, data_source, evaluation=False):
    result = native_generate(args, rollout_id, data_source, evaluation)
    if evaluation or rollout_id != 0:
        return result
    phase = Path(os.environ["GLM53_PHASE_DIR"])
    trace = phase / f"rollout-{rollout_id}-layers"
    trace.mkdir(exist_ok=True)
    samples = [sample for group in result.samples for sample in group if sample.response_length > 0]
    if not samples:
        raise RuntimeError("No valid rollout sample available for the layer comparison")
    previous = phase.parent / "short-01"
    known = [
        row["all_token_ids"]
        for f in previous.glob("old-logp-rollout-0-rank-*.json")
        for row in json.loads(f.read_text())["records"]
    ]
    sample = next((item for item in samples if list(item.tokens) in known), samples[0])
    target = trace / "target.json"
    target.write_text(json.dumps({"tokens": list(sample.tokens), "sample": int(sample.index),
                                  "response_length": sample.response_length,
                                  "matches_previous_run": list(sample.tokens) in known}) + "\n")
    (phase / "trace-target.json").write_text(str(target) + "\n")
    flag = phase / "trace-active"
    flag.write_text(str(target) + "\n")
    try:
        response = generate(f"http://{os.environ['RAY_NODE_IP']}:15000",
                            list(sample.tokens), 0, "glm53-layer-trace")
    finally:
        flag.unlink(missing_ok=True)
    actual = response["meta_info"]["input_token_logprobs"]
    if [row[1] for row in actual] != list(sample.tokens):
        raise RuntimeError("Layer trace replay returned different token IDs")
    (trace / "sglang-prefill-logp.json").write_text(json.dumps(actual) + "\n")
    return result

