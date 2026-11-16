import json
import math
import os
import struct
import urllib.request
from pathlib import Path


def generate(url, prompts, count, rid):
    batch = bool(prompts and isinstance(prompts[0], list))
    payload = {
        "input_ids": prompts,
        "rid": [f"{rid}-{i}" for i in range(len(prompts))] if batch else rid,
        "sampling_params": {"temperature": 0, "max_new_tokens": count, "ignore_eos": True},
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    request = urllib.request.Request(
        url + "/generate", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=1200) as response:
        return json.load(response)


def compare(left, right):
    values = [(a[0], b[0]) for a, b in zip(left, right) if a[0] is not None and b[0] is not None]
    finite = [(a, b) for a, b in values if math.isfinite(a) and math.isfinite(b)]
    diffs = [abs(a - b) for a, b in finite]
    return {
        "same_token_ids": [x[1] for x in left] == [x[1] for x in right],
        "same_logp_positions": [x[0] is None for x in left] == [x[0] is None for x in right],
        "compared": len(values),
        "bit_mismatches": len(values) - len(finite) + sum(
            struct.pack("!d", a) != struct.pack("!d", b) for a, b in finite
        ),
        "mae": sum(diffs) / len(diffs) if diffs else None,
        "max_abs_diff": max(diffs, default=None),
    }


def passed(item):
    return (
        item["same_token_ids"] and item["same_logp_positions"] and item["compared"] > 0
        and item["bit_mismatches"] == 0
    )


def tokens(length, start):
    return [start + i % 53 for i in range(length)]


def check(url):
    reused = Path(os.environ["GLM53_PHASE_DIR"]) / "reused-inference-evidence.json"
    if reused.is_file():
        print(f"GLM53_BI reusing inference results: {reused}", flush=True)
        return
    evidence = Path(os.environ["GLM53_BATCH_REPORT"]).with_name("reused-short-evidence.json")
    if evidence.exists():
        print(f"GLM53_BI reusing prior short results: {evidence}", flush=True)
    lengths = [int(x) for x in os.getenv("GLM53_BI_LENGTHS", "256,512" if evidence.exists() else "32,97").split(",")]
    output_count = int(os.getenv("GLM53_BI_OUTPUT_TOKENS", "16" if evidence.exists() else "8"))
    batch_report = {}
    pd_report = {}
    for length in lengths:
        target = tokens(length, 100)
        short, medium, long = tokens(15, 122), tokens(63, 301), tokens(max(length + 7, 128), 211)
        name = f"bi-{length}-{output_count}"
        print(f"GLM53_BI begin {name}", flush=True)
        reference = generate(url, target, output_count, f"{name}-single")["meta_info"]
        cases = {
            "repeat": (target, None),
            "batch2_position0": ([target, long], 0),
            "batch2_position1": ([short, target], 1),
            "batch4_position2": ([long, medium, target, short], 2),
        }
        scores = {}
        for case, (prompts, position) in cases.items():
            print(f"GLM53_BI schedule case={case} length={length}", flush=True)
            result = generate(url, prompts, output_count, f"{name}-{case}")
            meta = (result if position is None else result[position])["meta_info"]
            scores[case] = {
                field: compare(reference[field], meta[field])
                for field in ("input_token_logprobs", "output_token_logprobs")
            }
        batch_report[str(length)] = scores
        decode = reference["output_token_logprobs"]
        response_tokens = [item[1] for item in decode]
        replay = generate(url, target + response_tokens, 0, f"{name}-prefill-replay")["meta_info"]
        assert [item[1] for item in replay["input_token_logprobs"]] == target + response_tokens
        prefill = replay["input_token_logprobs"][len(target):]
        pd_report[str(length)] = {"comparison": compare(decode, prefill), "prompt_tokens": target,
                                  "decode": decode, "prefill": prefill}
    bi_pass = all(passed(item) for length in batch_report.values() for case in length.values() for item in case.values())
    pd_pass = all(passed(item["comparison"]) for item in pd_report.values())
    report = Path(os.environ["GLM53_BATCH_REPORT"])
    report.write_text(json.dumps({"passed": bi_pass, "comparisons": batch_report}, indent=2) + "\n")
    report.with_name("prefill-decode.json").write_text(
        json.dumps({"passed": pd_pass, "comparisons": pd_report}, indent=2) + "\n"
    )
    print(f"BATCH_INVARIANCE passed={bi_pass} PREFILL_DECODE passed={pd_pass} report={report}", flush=True)
    if os.getenv("GLM53_BI_STRICT") == "1" and not (bi_pass and pd_pass):
        raise RuntimeError("BI or prefill/decode parity failed")

