# Classification — Code Clone Detection and Code Classification

This folder contains all source code for running deep learning experiments on two tasks: **Code Clone Detection** and **Code Classification**. Experiments are conducted across multiple context combinations (version history, call-graph, method age) and aggregation strategies, using the following models:

- **CodeBERT** and **GraphCodeBERT** (transformer family)
- **CodeT5** (codet5 family)
- **PLBART** (summarisation — see [`Summarization/`](../Summarization/))
- **ASTNN**

For LLM experiments (CodeLlama, DeepSeek-Coder, Qwen2.5-Coder), see [`LLM/`](../LLM/).  
For the full replication package overview, see the [root README](../README.md).

---

## Data

Download `Dataset.zip` from Figshare (https://figshare.com/s/71c3233d55c2ad91f30c) and place the folders as follows:

| Folder in `Dataset.zip` | Place at |
|---|---|
| `clone_detection/` | `Classification/data/clone_detection/` |
| `classification/` | `Classification/data/classification/` |
| `clone_detection_vh_ablation/` | `Classification/data/clone_detection_vh_ablation/` |
| `classification_vh_ablation/` | `Classification/data/classification_vh_ablation/` |

The augmented datasets include the original SeSaMe source code enriched with version history, caller/callee context, and method age. See the [root README](../README.md) for full archive contents and file sizes.

---

## Folder Structure

```
Classification/
├── mining/
│   ├── version_history.py         ← mine version history from GitHub (PyDriller)
│   └── data_merging.py            ← post-process and merge version history
├── data_update.py                 ← merge call-graph into augmented dataset
├── preprocess_clone.py            ← produce clone_detection/ .pkl splits
├── preprocess_class.py            ← produce classification/ .pkl splits
├── prepare_vh_ablation_data.py    ← produce history-depth ablation splits
├── global_config.py               ← shared path configuration
├── tree.py / utils.py             ← shared utilities
│
├── astnn_*/                       ← ASTNN variants
├── transformer_*/                 ← CodeBERT / GraphCodeBERT variants
├── codet5_*/                      ← CodeT5 variants
│
├── transformer_shuffled_context_control/   ← Section 6.1: shuffled-context control
├── codet5_shuffled_context_control/        ← Section 6.1: shuffled-context control (CodeT5)
├── transformer_projection_control/         ← Section 5.2: projection control
├── transformer_vh_ablation/                ← Section 5.3: history depth ablation
├── codet5_clone_callgraph_v1_dump/         ← Section 5.6: qualitative analysis (call-graph)
├── codet5_clone_qualitative_dump/          ← Section 5.6: qualitative analysis (clone)
│
├── experiment_all.sh              ← run all experiments
├── experiment_clone.sh            ← run clone detection experiments only
├── experiment_class.sh            ← run classification experiments only
└── experiment.yml                 ← conda environment specification
```

---

## Environment Setup

```bash
conda env create -f experiment.yml
conda activate experiment
```

Key dependencies: Python 3.9, PyTorch 2.1.0 (CUDA 12.1), Transformers 4.38.2, javalang 0.13.0.

---

## Running Experiments

All experiments follow the same naming convention:

```
<model_family>_<context_combination>/<task>_<aggregation>.py
```

Where:
- **model family**: `transformer` (CodeBERT, GraphCodeBERT), `codet5` (CodeT5), `astnn`
- **context combination**: `versionall`, `callgraph`, `versionall_callgraph`, `versionall_callgraph_numofdays`, etc.
- **task**: `clone` (clone detection), `class` (code classification)
- **aggregation**: `pure_code` (baseline), `concat`, `max_pool`, `diff_concat`

### Example commands

```bash
# CodeBERT + version history + call graph + method age, concatenation, clone detection
python -u transformer_versionall_callgraph_numofdays/clone_concat.py --model codebert

# CodeT5 + version history only, max-pooling, code classification
python -u codet5_versionall/class_max_pool.py --model codet5base

# ASTNN + all context, diff-concat, clone detection
python -u astnn_versionall_callgraph_numofdays/clone_diff_concat.py

# Run all experiments
bash experiment_all.sh > log.txt
```

Supported `--model` values: `codebert`, `graphcodebert` (transformer family); `codet5base` (codet5 family).

Results are written to `result.txt`; training logs to `log.txt`.

---

## Control and Ablation Experiments

### Shuffled-context control (Section 6.1)

```bash
bash transformer_shuffled_context_control/run_all.sh
bash codet5_shuffled_context_control/run_all.sh
```

### Projection control (Section 5.2)

```bash
bash transformer_projection_control/run_all.sh
```

### History depth ablation (Section 5.3)

```bash
bash transformer_vh_ablation/run_vh_ablation.sh
```

Uses splits in `data/clone_detection_vh_ablation/` and `data/classification_vh_ablation/` which vary the number of historical versions from 1 to the full available history.

### Qualitative analysis (Section 5.6)

```bash
bash codet5_clone_callgraph_v1_dump/run_all.sh
python codet5_clone_callgraph_v1_dump/merge_predictions.py

bash codet5_clone_qualitative_dump/run_all.sh
python codet5_clone_qualitative_dump/merge_predictions.py
```

---

## LLM Experiments

LLM experiments (CodeLlama, DeepSeek-Coder, Qwen2.5-Coder) are not in this folder. See the [`LLM/`](https://github.com/huynxvn/EnhancingCodeRepWithContext/tree/main/LLM) folder and its [`README`](../LLM/README.md) for full instructions.

---

## Citation

```bibtex
@article{nguyen2025enhancing,
  title     = {Enhancing Neural Code Representation with Additional Context},
  author    = {Nguyen, Huy and Thongtanunam, Patanamon and Treude, Christoph},
  journal   = {ACM Transactions on Software Engineering and Methodology},
  year      = {2025}
}
```

Preprint: https://arxiv.org/abs/2510.12082
