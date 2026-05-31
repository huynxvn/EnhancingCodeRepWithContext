#!/bin/bash
cd "$(dirname "$0")/.."

echo "=== R2 Q4 shuffled-context control — CodeBERT ==="
python transformer_shuffled_context_control/class_concat.py --model codebert
echo "=== Done ==="
