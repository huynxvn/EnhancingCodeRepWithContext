#!/bin/bash

LOG_FILE="./transformer_vh_ablation/vh_ablation_run.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "VH Ablation Run Started: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Clone detection
for mv in 1 3 all; do
    echo "" | tee -a "$LOG_FILE"
    echo "--- Clone: max_versions=$mv | $(date) ---" | tee -a "$LOG_FILE"
    python transformer_vh_ablation/clone_concat.py --model CodeBERT --max_versions $mv 2>&1 | tee -a "$LOG_FILE"
done

# Code classification
for mv in 1 3 all; do
    echo "" | tee -a "$LOG_FILE"
    echo "--- Class: max_versions=$mv | $(date) ---" | tee -a "$LOG_FILE"
    python transformer_vh_ablation/class_concat.py --model CodeBERT --max_versions $mv 2>&1 | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "VH Ablation Run Finished: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
