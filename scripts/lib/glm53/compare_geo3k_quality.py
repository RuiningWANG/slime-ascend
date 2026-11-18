"""Compare one aligned Geo3K round with the lowest-scoring native round."""
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

native, aligned, output = map(Path, sys.argv[1:4])
left_protocol = json.loads((native / "evaluation-protocol.json").read_text())
right_protocol = json.loads((aligned / "evaluation-protocol.json").read_text())
assert right_protocol["rounds"] == 1
assert {k: v for k, v in left_protocol.items() if k != "rounds"} == {
    k: v for k, v in right_protocol.items() if k != "rounds"
}, "Model, data or sampling protocol differs"

def load(phase):
    rows = [json.loads(line) for line in (phase / "quality-receipts.jsonl").read_text().splitlines()]
    keys = {(r["dataset"], r["round"], r["sample"]) for r in rows}
    assert len(keys) == len(rows), "Duplicate receipts"
    jobs = list((phase / "jobs").iterdir())
    assert len(jobs) == 1
    assert json.loads((jobs[0] / "ray-job-status.json").read_text())["status"] == "SUCCEEDED"
    return rows

left, right = load(native), load(aligned)
round_rows = defaultdict(list)
for row in left:
    round_rows[row["round"]].append(row)
assert sorted(round_rows) == list(range(left_protocol["rounds"]))
native_rounds = {
    i: {"responses": len(rows), "mean_reward": sum(r["reward"] for r in rows) / len(rows)}
    for i, rows in sorted(round_rows.items())
}
selected = min(native_rounds, key=lambda i: (native_rounds[i]["mean_reward"], i))
reference = {(r["dataset"], r["sample"]): r for r in round_rows[selected]}
candidate = {(r["dataset"], r["sample"]): r for r in right}
assert {r["round"] for r in right} == {0}
assert reference.keys() == candidate.keys()
groups = defaultdict(list)
for key, a in reference.items():
    b = candidate[key]
    assert a["label"] == b["label"], key
    assert a["tokens"][:a["prompt_tokens"]] == b["tokens"][:b["prompt_tokens"]], key
    assert (a["question"], a["replicate"]) == (b["question"], b["replicate"])
    groups[(key[0], a["question"])].append((a["reward"], b["reward"]))
assert {len(v) for v in groups.values()} == {right_protocol["samples_per_prompt"]}
per_question = []
for (dataset, question), values in sorted(groups.items()):
    a = sum(x for x, _ in values) / len(values)
    b = sum(y for _, y in values) / len(values)
    per_question.append({"dataset": dataset, "question": question, "responses": len(values),
                         "native": a, "aligned": b, "delta_pp": 100 * (b - a)})
native_mean = native_rounds[selected]["mean_reward"]
aligned_mean = sum(r["reward"] for r in right) / len(right)
delta = aligned_mean - native_mean
differences = [q["delta_pp"] for q in per_question]
generator = random.Random(522)
bootstrap = sorted(
    sum(generator.choices(differences, k=len(differences))) / len(differences)
    for _ in range(10000)
)
def percentile(q):
    position = (len(bootstrap) - 1) * q
    lower = int(position)
    fraction = position - lower
    return bootstrap[lower] * (1 - fraction) + bootstrap[min(lower + 1, len(bootstrap) - 1)] * fraction

report = {
    "native_phase": str(native), "aligned_phase": str(aligned),
    "acceptance_rule": "One aligned round >= minimum native round minus one percentage point",
    "native_protocol": left_protocol, "aligned_protocol": right_protocol,
    "native_rounds": native_rounds, "selected_native_round": selected,
    "native_total_responses": len(left), "responses_per_compared_round": len(right),
    "questions": len(groups), "native_mean_reward": native_mean,
    "aligned_mean_reward": aligned_mean, "delta_percentage_points": delta * 100,
    "minimum_acceptable_aligned_score": native_mean - 0.01,
    "maximum_allowed_drop_percentage_points": 1.0,
    "question_bootstrap_95ci_pp": [percentile(0.025), percentile(0.975)],
    "bootstrap_scope": "Descriptive conditional on selected native round",
    "bootstrap_replicates": 10000, "bootstrap_seed": 522,
    "per_question": per_question, "gate_pass": delta >= -0.01 - 1e-12,
}
output.write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items()
                  if k not in ("per_question", "native_protocol", "aligned_protocol")}), flush=True)
