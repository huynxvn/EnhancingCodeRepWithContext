# Enhancing Neural Code Representation with Additional Context

This repository contains the replication package for the paper:

> **"Enhancing Neural Code Representation with Additional Context"**  
> Huy Nguyen, Patanamon Thongtanunam, Christoph Treude  
> *Transactions on Software Engineering and Methodology (TOSEM), 2025*

---

## Replication Package

The full replication package is hosted across two platforms:

- **Figshare** – augmented datasets, human evaluation data, and source code archives
- **GitHub** (this repository) – mining scripts, model training and evaluation scripts, and LLM experiment code

### What is released

| Archive | Contents | Format |
|---|---|---|
| `Augmented_Dataset.zip` (Figshare) | SeSaMe enriched with version history, call-graph, and method age | `.pkl` (train/dev/test splits) |
| `Augmented_Dataset.zip` (Figshare) | CodeSearchNet-Java enriched with version history, call-graph, and method age | `.jsonl` |
| `Augmented_Dataset.zip` (Figshare) | Vul4J enriched with version history, call-graph, and method age | `.jsonl` |
| `Source_Code.zip` (Figshare) | Mining scripts and model training code | `.py` |
| `Human_Evaluation.zip` (Figshare) | Human evaluation data for code summarisation | `.csv` / `.xlsx` |
| This GitHub repo | Mining scripts, all training/evaluation scripts, LLM experiment code | `.py`, `.sh` |

Partial fine-tuned model weights (`.bin` / `.pth`) are provided on Figshare for selected configurations. Training from scratch using the scripts and augmented datasets will reproduce all reported results.

---

## Repository Structure

```
EncodingAdditionalContext/          ← this repo (DL models: ASTNN, CodeBERT, GraphCodeBERT, CodeT5)
├── mining/
│   ├── version_history.py          ← Step 1: mine version history from GitHub
│   ├── data_merging.py             ← Step 2: post-process and merge version history
│   └── output/                     ← intermediate mining outputs
├── data/
│   ├── SeSaMe_VersionHistory_Callgraph.v7.json   ← final augmented SeSaMe dataset
│   ├── clone_detection/            ← preprocessed .pkl splits for clone detection
│   ├── classification/             ← preprocessed .pkl splits for code classification
│   └── classification_vh_ablation/ ← ablation splits (history depth study)
├── astnn_*/                        ← ASTNN variants (versionall, callgraph, numofdays, combinations)
├── transformer_*/                  ← CodeBERT / GraphCodeBERT / CodeBERTa variants
├── codet5_*/                       ← CodeT5 / CodeT5+ variants
├── experiment_all.sh               ← master script to run all DL experiments
├── experiment.yml                  ← conda environment specification
└── scripts/
    └── shuffle_context.py          ← control: shuffled-context ablation

LLM/                                ← LLM experiment code (CodeLlama, DeepSeek, Qwen)  [to be added]
├── script.py                       ← main entry point for LLM fine-tuning / inference
├── llm_models.py                   ← model definitions
├── inference_clone.py              ← clone detection inference
├── inference_class.py              ← code classification inference
├── inference_vuln.py               ← vulnerability detection inference
├── clone_all_finetune.sh           ← run all clone detection fine-tuning jobs
├── clone_all_inference.sh          ← run all clone detection inference jobs
├── vuln_all_inference.sh           ← run all vulnerability detection inference jobs
├── run_all_codellama13b.sh         ← CodeLlama-13B experiments
├── run_all_deepseek.sh             ← DeepSeek experiments
├── run_all_qwen30b.sh              ← Qwen-30B experiments
└── data/                           ← dataset files for LLM experiments
```

---

## Environment Setup

### Deep Learning Models (ASTNN, CodeBERT, GraphCodeBERT, CodeT5)

```bash
conda env create -f experiment.yml
conda activate experiment
```

Key dependencies: Python 3.9, PyTorch 2.1.0 (CUDA 12.1), Transformers 4.38.2, javalang 0.13.0.

### LLM Experiments (CodeLlama, DeepSeek, Qwen)

```bash
pip install torch transformers peft bitsandbytes accelerate
```

The LLM scripts use 4-bit / 8-bit quantisation via `bitsandbytes` and optional LoRA fine-tuning via `peft`.

---

## Context Mining Pipeline

The augmented datasets are already provided on Figshare. Follow these steps only if you need to re-mine context from scratch or extend to new repositories.

### Prerequisites

- Clone all SeSaMe / CodeSearchNet source repositories locally (see dataset documentation for the repository list)
- Install [Java-CallGraph](https://github.com/gousiosg/java-callgraph) for call-graph extraction
- Install [Lizard](https://github.com/terryyin/lizard) for static analysis filtering: `pip install lizard`
- Install PyDriller for Git traversal: `pip install pydriller`

### Step 1 – Mine version history

Edit the path constants at the bottom of `mining/version_history.py` to point to your local repository clones:

```python
SOURCE_DATA_PATH = 'sesame/src/'        # path to SeSaMe CSV
REPOS_DIR        = 'sesame/src/repos'   # path to cloned repositories
OUTPUT_DIR       = 'mining/output/'
```

Then run:

```bash
python mining/version_history.py
# Output: mining/output/version_history_mining.json
```

This script traverses each repository's Git history using PyDriller, extracts all historical versions of each method, applies Lizard static analysis to filter out non-meaningful commits (whitespace-only or comment-only changes), and records commit metadata (SHA, date, author).

Historical versions are ordered most-recent-first, and truncation is applied to fit within the model's 512-token budget.

### Step 2 – Post-process and merge version history

```bash
python mining/data_merging.py
# Output: mining/output/version_history.json
```

This computes `number_of_versions` and `days_to_exist` (method age in days, normalised before being passed to the model) for each method, and produces the merged JSON ready for the next step.

### Step 3 – Extract call-graph context

Run Java-CallGraph on each project's compiled `.jar` to extract caller/callee relationships:

```bash
java -jar javacg-0.1-SNAPSHOT-static.jar <project>.jar > callgraph_output.txt
```

The extracted call-graph is then merged with the version history JSON using `data_update.v01.py` to produce the final augmented dataset (`data/SeSaMe_VersionHistory_Callgraph.v7.json`).

### Step 4 – Preprocess for experiments

```bash
python others/preprocess_clone.py     # produces data/clone_detection/ splits
python others/preprocess_class.py     # produces data/classification/ splits
```

---

## Running Deep Learning Experiments

All DL experiments follow the same naming convention:

```
<model_family>_<context_combination>/<task>_<aggregation>.py
```

Where:
- **model family**: `transformer` (CodeBERT, GraphCodeBERT, CodeBERTa), `codet5` (CodeT5, CodeT5+), `astnn`
- **context combination**: `versionall`, `callgraph`, `versionall_callgraph`, `versionall_callgraph_numofdays`, etc.
- **task**: `clone` (clone detection), `class` (code classification)
- **aggregation**: `pure_code` (baseline), `concat`, `max_pool`, `diff_concat`

### Example commands

```bash
# CodeBERT + version history + call graph + method age, concatenation aggregation, clone detection
python -u transformer_versionall_callgraph_numofdays/clone_concat.py --model codebert

# CodeT5 + version history only, max-pooling aggregation, code classification
python -u codet5_versionall/class_max_pool.py --model codet5base

# ASTNN + all context, diff-concat aggregation, clone detection
python -u astnn_versionall_callgraph_numofdays/clone_diff_concat.py

# Run all experiments (CodeBERT shown; uncomment other models in the script)
bash experiment_all.sh > log.txt
```

Supported `--model` values for transformer/codet5 families: `codebert`, `graphcodebert`, `codeberta`, `codet5base`, `codet5p110membedding`.

Results are written to `result.txt`; training logs to `log.txt`.

### History depth ablation (Section 5.3)

```bash
bash transformer_vh_ablation/run_vh_ablation.sh
```

Uses the `data/clone_detection_vh_ablation/` and `data/classification_vh_ablation/` splits which vary the number of historical versions from 1 to the full available history.

---

## Running LLM Experiments

> **Note:** LLM source code will be placed in the `LLM/` folder of this repository. The experiments cover CodeLlama (7B, 13B, 34B), DeepSeek Coder (6.7B, 33B), and Qwen2.5-Coder (32B) with 4-bit and 8-bit quantisation and optional LoRA fine-tuning.

### Fine-tuning

```bash
cd LLM/
python script.py --model <model_name> --bits 4 --data_file <data_file> --mode train --use_lora
```

Example:

```bash
python script.py --model codellama7b --bits 4 --data_file sesame-clone --mode train --use_lora --test_every_epoch
```

### Inference

```bash
# Clone detection
bash clone_all_inference.sh

# Code classification
bash class_all_inference.sh

# Vulnerability detection
bash vuln_all_inference.sh
```

### Running specific large models

```bash
bash run_all_codellama13b.sh
bash run_all_codellama34b.sh
bash run_all_deepseek.sh
bash run_all_deepseek33b.sh
bash run_all_qwen30b.sh
```

---

## Datasets

| Dataset | Task | Source | Augmented format |
|---|---|---|---|
| SeSaMe | Clone detection, Code classification | [Kamp et al., 2019](https://github.com/sesame-dataset) | `.pkl` (train/dev/test) |
| CodeSearchNet-Java | Code summarisation | [Husain et al., 2019](https://github.com/github/CodeSearchNet) | `.jsonl` |
| Vul4J | Vulnerability detection | [Bui et al., 2022](https://github.com/buiducnhat/Vul4J) | `.jsonl` |

Each record in the augmented dataset includes the original source code plus:
- `version_history`: list of historical method versions (most-recent-first, filtered by Lizard)
- `calling` / `called`: caller and callee method source code
- `number_of_days` (normalised): method age in days

---

## Citation

If you use this replication package, please cite:

```bibtex
@article{nguyen2025enhancing,
  title     = {Enhancing Neural Code Representation with Additional Context},
  author    = {Nguyen, Huy and Thongtanunam, Patanamon and Treude, Christoph},
  journal   = {ACM Transactions on Software Engineering and Methodology},
  year      = {2025}
}
```
