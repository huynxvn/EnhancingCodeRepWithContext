#!/bin/bash
cd "$(dirname "$0")/.."

echo "=== R2 Q5 clone-detection per-pair dump — CodeT5 ==="
echo "--- Baseline (code only) ---"
python codet5_clone_qualitative_dump/clone_pure_code.py --model codet5base
echo "--- Full context (code + VH + CG + method age) ---"
python codet5_clone_qualitative_dump/clone_concat.py --model codet5base
echo "=== Done ==="
