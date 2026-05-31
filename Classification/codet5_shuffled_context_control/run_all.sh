#!/bin/bash
cd "$(dirname "$0")/.."

echo "=== R2 Q4 shuffled-context control — CodeT5 ==="
python codet5_shuffled_context_control/class_concat.py --model codet5base
echo "=== Done ==="
