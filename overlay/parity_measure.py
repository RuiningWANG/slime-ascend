import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

_rollout_id = None


def summarize(train_scores, rollout_scores, masks, tokens, response_lengths, positions):
    records = []
    for train, rollout, mask, token_ids, length, position in zip(
        train_scores, rollout_scores, masks, tokens, response_lengths, positions, strict=True
    ):
        mask = mask.detach().cpu().bool()
        left = train.detach().cpu()[mask].contiguous()
        right = rollout.detach().cpu()[mask].contiguous()
        if left.dtype != right.dtype or left.dtype != torch.float32:
            raise TypeError(f"Expected original FP32 logp, got {left.dtype} and {right.dtype}")
        difference = (left.double() - right.double()).abs()
        if not torch.isfinite(difference).all():
            raise ValueError("Nonfinite response log probabilities")
        records.append({
            "sample": int(position),
            "all_token_ids": token_ids.detach().cpu().tolist(),
            "response_length": int(length),
            "mask": mask.tolist(),
            "train": left.tolist(),
            "rollout": right.tolist(),
            "valid_tokens": left.numel(),
            "mismatches": int((left.view(torch.int32) != right.view(torch.int32)).sum()),
            "sum_abs_diff": float(difference.sum()),
            "max_abs_diff": float(difference.max()) if difference.numel() else None,
            "requires_grad": train.requires_grad,
        })
    count = sum(item["valid_tokens"] for item in records)
    return {
        "valid_tokens": count,
        "mismatches": sum(item["mismatches"] for item in records),
        "logpdiff": sum(item["sum_abs_diff"] for item in records) / count if count else None,
        "max_abs_diff": max((item["max_abs_diff"] for item in records if item["valid_tokens"]), default=None),
        "records": records,
    }


def compare_old(args, rollout_id, data):
    global _rollout_id
    from slime.backends.megatron_utils import loss

    _rollout_id = rollout_id
    if not getattr(loss._maybe_capture_log_probs, "_glm53_probe", False):
        capture_original = loss._maybe_capture_log_probs

        def capture_both(batch, scores):
            capture_original(batch, scores)
            if not torch.is_grad_enabled() or not any(item.requires_grad for item in scores):
                return
            summary = summarize(scores, batch["rollout_log_probs"], batch["loss_masks"],
                                batch["unconcat_tokens"], batch["response_lengths"], batch["partition"])
            summary["grad_enabled"] = True
            out = Path(os.environ["GLM53_PHASE_DIR"]) / f"actor-logp-rollout-{_rollout_id}-rank-{dist.get_rank()}.jsonl"
            with out.open("a") as stream:
                stream.write(json.dumps(summary) + "\n")
            print(f"GLM53_ACTOR_LOGP count={summary['valid_tokens']} mismatches={summary['mismatches']} "
                  f"logpdiff={summary['logpdiff']} max_abs_diff={summary['max_abs_diff']}", flush=True)

        capture_both._glm53_probe = True
        loss._maybe_capture_log_probs = capture_both

    if "log_probs" not in data:
        print(f"GLM53_OLD_LOGP rollout={rollout_id} native old-logp was skipped", flush=True)
        return
    summary = summarize(data["log_probs"], data["rollout_log_probs"], data["loss_masks"],
                        data["tokens"], data["response_lengths"], data["partition"])
    out = Path(os.environ["GLM53_PHASE_DIR"]) / f"old-logp-rollout-{rollout_id}-rank-{dist.get_rank()}.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"GLM53_OLD_LOGP {out} count={summary['valid_tokens']} mismatches={summary['mismatches']} "
          f"logpdiff={summary['logpdiff']} max_abs_diff={summary['max_abs_diff']}", flush=True)


def measure_only(*, pg_loss, loss_masks, **kwargs):
    return pg_loss, loss_masks, {}

