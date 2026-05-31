#!/bin/bash
cd "$(dirname "$0")/.."

echo "=== R2 Q5 v1-augmented clone-detection per-pair dump — CodeT5 ==="
echo "(Baseline is reused from codet5_clone_qualitative_dump/predictions_baseline.json — no re-training needed.)"
echo "--- Full context with v1 caller/callee (code + VH + CG + method age) ---"
python codet5_clone_callgraph_v1_dump/clone_concat.py --model codet5base
echo "=== Done ==="
