import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)

baseline_path = os.path.join(script_dir, "predictions_baseline.json")
context_path  = os.path.join(script_dir, "predictions_context.json")
output_dir    = os.path.join(repo_root, "TOSEM_Major_Revision")
output_path   = os.path.join(output_dir, "sesame_clone_predictions_v1.json")

# Load
with open(baseline_path) as f:
    baseline = json.load(f)
with open(context_path) as f:
    context = json.load(f)

# Sanity checks
assert len(baseline) == len(context), (
    f"Length mismatch: baseline={len(baseline)}, context={len(context)}"
)
for i, (b, c) in enumerate(zip(baseline, context)):
    assert b["pair_idx"] == c["pair_idx"], (
        f"pair_idx mismatch at position {i}: baseline={b['pair_idx']}, context={c['pair_idx']}"
    )
    assert b["label"] == c["label"], (
        f"label mismatch at position {i} (pair_idx={b['pair_idx']}): "
        f"baseline={b['label']}, context={c['label']}"
    )

# Merge — use context file's caller/callee as canonical (v1 real text)
unified = []
for b, c in zip(baseline, context):
    unified.append({
        "pair_idx":          b["pair_idx"],
        "uid_x":             b["uid_x"],
        "uid_y":             b["uid_y"],
        "label":             b["label"],
        "baseline_pred":     b["pred"],
        "baseline_logit":    b["logit"],
        "context_pred":      c["pred"],
        "context_logit":     c["logit"],
        "code_x":            b["code_x"],
        "code_y":            b["code_y"],
        "code_versions_x":   b["code_versions_x"],
        "code_versions_y":   b["code_versions_y"],
        "calling_x":         c["calling_x"],
        "calling_y":         c["calling_y"],
        "called_x":          c["called_x"],
        "called_y":          c["called_y"],
        "number_of_days_x":  b["number_of_days_x"],
        "number_of_days_y":  b["number_of_days_y"],
    })

# Save
os.makedirs(output_dir, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(unified, f, indent=2, default=str)

# Summary
total     = len(unified)
pos       = sum(r["label"] == 1 for r in unified)
neg       = total - pos
b_correct = sum(r["baseline_pred"] == r["label"] for r in unified)
c_correct = sum(r["context_pred"]  == r["label"] for r in unified)
wins      = sum(r["baseline_pred"] != r["label"] and r["context_pred"] == r["label"] for r in unified)
losses    = sum(r["baseline_pred"] == r["label"] and r["context_pred"] != r["label"] for r in unified)
non_empty_caller = sum(1 for r in unified if r["calling_x"] and r["calling_x"] != r["code_x"])
non_empty_callee = sum(1 for r in unified if r["called_x"] and r["called_x"] != r["code_x"])

print(f"Total pairs                              : {total}")
print(f"Label distribution                       : positives={pos}, negatives={neg}")
print(f"Baseline accuracy                        : {b_correct}/{total} = {b_correct/total:.4f}")
print(f"Context accuracy                         : {c_correct}/{total} = {c_correct/total:.4f}")
print(f"Wins  (baseline wrong, context right)    : {wins}")
print(f"Losses (baseline right, context wrong)   : {losses}")
print(f"Records with real caller_x (≠ code_x)   : {non_empty_caller}")
print(f"Records with real callee_x (≠ code_x)   : {non_empty_callee}")
print(f"\nOutput written to: {output_path}")
