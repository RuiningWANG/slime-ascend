"""Validate actual trainer metrics and FP32 rollout/policy audits for one run."""

import argparse
import ast
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import shlex


METRIC = "train/train_rollout_logprob_abs_diff"
ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STEP = re.compile(r"model\.py:\d+ - step (\d+): (\{.*\})")
GRAPH_REPLAY = re.compile(r"decode batch.*(?:cuda|npu) graph:\s*true", re.I)


def complete_lines(path):
    if path.is_file():
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if line.endswith("\n"):
                    yield line


def inspect(run, audit_dir=None):
    command = shlex.split((run / "command.sh").read_text(encoding="utf-8"))

    def value(flag):
        return int(command[command.index(flag) + 1])

    steps = value("--num-rollout")
    batch = value("--global-batch-size")
    world = value("--actor-num-nodes") * value("--actor-num-gpus-per-node")
    model_parallel = math.prod(value(flag) for flag in (
        "--tensor-model-parallel-size", "--pipeline-model-parallel-size", "--context-parallel-size"
    ))
    dp, remainder = divmod(world, model_parallel)
    if remainder or steps <= 0 or batch <= 0 or dp <= 0:
        raise ValueError("Invalid run size or parallel configuration")
    environment = json.loads((run / "runtime-env.json").read_text(encoding="utf-8"))["env_vars"]
    if audit_dir is None:
        audit_dir = Path(environment["GLM53_PHASE_DIR"])

    job = f"lh-{run.name}"
    status_pattern = re.compile(rf"Job (?:'{re.escape(job)}'\s+|{re.escape(job)}:\s+)(?:was\s+)?(succeeded|failed|stopped)\b", re.I)
    outcomes = [match[1].lower() for raw in complete_lines(run / "stdout.log")
                if (match := status_pattern.search(ESCAPE.sub("", raw)))]
    outcome = outcomes[-1] if outcomes else None
    metrics = {}
    issues = [f"Ray job {outcome}"] if outcome in ("failed", "stopped") else []
    evidence = dict(level3_lines=0, graph_captures=0, graph_replays=0, weight_sync_logs=0)

    candidates = [run / "ray-driver.log"]
    if "RAY_TMPDIR" in environment:
        candidates.append(Path(environment["RAY_TMPDIR"]) / "session_latest" / "logs" / f"job-driver-{job}.log")
    driver = next((p for p in candidates if p.is_file()), run / "stdout.log")
    for raw in complete_lines(driver):
        line = ESCAPE.sub("", raw)
        evidence["level3_lines"] += int(
            ("GLM_LEVEL3" in line and "level=3" in line) or "GLM53 NPU deterministic level 3" in line
        )
        evidence["graph_captures"] += int("Capture target decode NPU graph end" in line)
        evidence["graph_replays"] += int(bool(GRAPH_REPLAY.search(line)))
        evidence["weight_sync_logs"] += int("Timer update_weights end" in line)
        match = STEP.search(line)
        if not match or METRIC not in match[2]:
            continue
        step = int(match[1])
        try:
            values = ast.literal_eval(match[2])
            row = {"logpdiff": float(values[METRIC]), "grad_norm": float(values["train/grad_norm"]),
                   "loss": float(values["train/loss"]), "batch": int(values["train/global_batch_size"])}
        except (ValueError, SyntaxError, KeyError, TypeError):
            issues.append(f"Unreadable metric at step {step}")
            continue
        if any(not math.isfinite(num) for num in row.values()) or row["grad_norm"] < 0:
            issues.append(f"Invalid training values at step {step}")
            continue
        if step in metrics and metrics[step] != row:
            issues.append(f"Conflicting trainer metrics at step {step}")
        metrics[step] = row
        if row["logpdiff"] != 0 or row["batch"] != batch:
            issues.append(f"Trainer mismatch or wrong global batch at step {step}")

    audits = defaultdict(dict)
    for path in sorted(audit_dir.glob("strict-actor-audit-rank-*.jsonl")):
        for line in complete_lines(path):
            row = json.loads(line)
            step, rank = row["rollout"], row["rank"]
            if not (0 <= step < steps and 0 <= rank < world):
                issues.append(f"Unexpected audit step {step} or rank {rank}")
                continue
            if rank in audits[step]:
                issues.append(f"Duplicate audit at step {step}, rank {rank}")
            audits[step][rank] = row
            if (row["microbatches"] <= 0 or row["samples"] <= 0 or row["valid_tokens"] <= 0
                    or row["mismatch_tokens"] != 0 or row["max_abs_diff"] != 0.0 or row["absolute_sum"] != 0.0):
                issues.append(f"Failed FP32 token audit at step {step}, rank {rank}")

    rows = {}
    for step in sorted(metrics.keys() | audits.keys()):
        ranks = audits.get(step, {})
        counts = dict(samples=sum(x["samples"] for x in ranks.values()),
                      valid_tokens=sum(x["valid_tokens"] for x in ranks.values()),
                      mismatch_tokens=sum(x["mismatch_tokens"] for x in ranks.values()))
        complete = len(ranks) == dp and counts["samples"] == batch and counts["valid_tokens"] > 0
        rows[step] = {**metrics.get(step, {}), **counts, "audit_ranks": sorted(ranks),
                      "complete_audit": complete}

    expected = set(range(steps))
    if outcome == "succeeded":
        if set(metrics) != expected or set(audits) != expected:
            issues.append("Missing or unexpected step numbers in the completed job")
        if any(not row["complete_audit"] for row in rows.values()):
            issues.append("Missing ranks or samples in the completed job")
        if not all(evidence[key] > 0 for key in ("level3_lines", "graph_captures", "weight_sync_logs")):
            issues.append("Deterministic level 3, graph capture or weight sync evidence is missing")

    evidence["nonzero_gradient_steps"] = [step for step, row in sorted(metrics.items()) if row["grad_norm"] > 0]
    evidence["zero_gradient_steps"] = [step for step, row in sorted(metrics.items()) if row["grad_norm"] == 0]
    status = "failed" if issues else "passed" if outcome == "succeeded" else "incomplete"
    return {"run": str(run), "audit_directory": str(audit_dir), "status": status,
            "expected_steps": steps, "global_batch": batch, "expected_audit_ranks": dp,
            "ray_outcome": outcome, "driver_log": str(driver),
            "evidence": evidence, "issues": issues, "steps": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Single-node launcher job directory")
    parser.add_argument("--audit-dir", type=Path, help="Override the path when checking an archived copy")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect(args.run, args.audit_dir)
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(args.output)
    else:
        print(serialized, end="")
    return {"passed": 0, "failed": 1, "incomplete": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
