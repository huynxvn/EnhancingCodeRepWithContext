# LLM Experiments

This folder contains the code for all Large Language Model (LLM) experiments reported in the paper. It covers four SE tasks evaluated under two settings (prompt-based context injection and LoRA fine-tuning), using seven LLM configurations.

---

## Models

| Model key | Base model | Parameters | Architecture |
|---|---|---|---|
| `codellama-7b-instruct` | CodeLlama-7B-Instruct | 7B | Decoder-only |
| `codellama-13b-instruct` | CodeLlama-13B-Instruct | 13B | Decoder-only |
| `codellama-34b-instruct` | CodeLlama-34B-Instruct | 34B | Decoder-only |
| `deepseek-coder-6.7b-instruct` | DeepSeek-Coder-6.7B-Instruct | 6.7B | Decoder-only |
| `qwen25-coder-1.5b` | Qwen2.5-Coder-1.5B | 1.5B | Decoder-only |
| `qwen25-coder-7b` | Qwen2.5-Coder-7B | 7B | Decoder-only |
| `qwen25-coder-14b` | Qwen2.5-Coder-14B | 14B | Decoder-only |

Additional models in the registry (not reported in the main paper): `deepseek-coder-1.3b-instruct`, `deepseek-coder-33b-instruct`, `qwen25-coder-32b`, `codellama-34b-instruct`, `starcoder2-3b`, `starcoder2-7b`, `codegemma-2b`, `codegemma-7b`, `phi-3.5-mini`.

---

## Tasks

| Task | Script | Data folder | Metric |
|---|---|---|---|
| Code Summarisation | `script.py` / `inference.py` | `data/data/` | BLEU-4, ROUGE-L |
| Code Clone Detection | `script_clone.py` / `inference_clone.py` | `data/clone_detection/` | F1 |
| Code Classification | `script_class.py` / `inference_class.py` | `data/classification/` | F1 |
| Vulnerability Detection | `inference_vuln.py` | `data/vuln/` | F1 |

---

## Context Experiment Types

The `--exp` flag controls which context is injected into the prompt:

| `--exp` value | Context included |
|---|---|
| `baseline` | Source code only |
| `code_vh` | Source code + version history |
| `code_cg` | Source code + call graph (caller + callee) |
| `code_vh_nod` | Source code + version history + method age |
| `code_vh_cg` | Source code + version history + call graph |
| `code_cg_vh` | Source code + call graph + version history (different order) |
| `code_vh_cg_nod` | Source code + version history + call graph + method age |
| `code_cg_vh_nod` | Source code + call graph + version history + method age |

---

## Hardware Requirements

- **Recommended**: NVIDIA A100 80 GB
- **Minimum**: NVIDIA A100 40 GB for models ≤ 7B with 4-bit quantisation
- Models ≥ 13B require multi-GPU or 4-bit quantisation on a single A100 80 GB
- All scripts use Flash Attention 2 when available

---

## Environment Setup

```bash
pip install torch transformers peft bitsandbytes accelerate wandb tqdm scikit-learn nltk
```

Models are loaded from HuggingFace Hub. Ensure you have internet access or pre-cache the model weights.

---

## Setting 1: Prompt-Based Context Injection (Zero-shot / Few-shot)

Inference scripts load a pre-trained model and run it directly on augmented data — no fine-tuning required.

### Code Summarisation

```bash
python inference.py \
    --model deepseek-coder-6.7b-instruct \
    --bits 4 \
    --data_file data/data/test.jsonl \
    --exp code_vh_cg_nod
```

### Clone Detection

```bash
python inference_clone.py \
    --model codellama-7b-instruct \
    --bits 4 \
    --data_file data/clone_detection/test.pkl \
    --exp code_vh_cg_nod
```

### Code Classification

```bash
python inference_class.py \
    --model qwen25-coder-7b \
    --bits 4 \
    --data_file data/classification/test.pkl \
    --exp code_vh_cg_nod
```

### Vulnerability Detection

```bash
python inference_vuln.py \
    --model deepseek-coder-6.7b-instruct \
    --bits 4 \
    --data_file data/vuln/test.jsonl \
    --exp code_vh_cg_nod
```

---

## Setting 2: LoRA Fine-Tuning

Fine-tuning uses QLoRA (4-bit or 8-bit quantisation + LoRA adapters via `peft`). Training runs for up to 10 epochs with early stopping (patience = 3). WandB logging is enabled by default.

### Code Summarisation

```bash
# Train
python script.py \
    --model deepseek-coder-6.7b-instruct \
    --bits 4 \
    --data_file data \
    --mode train \
    --use_lora \
    --exp code_vh_cg_nod \
    --batch_size 2 \
    --test_every_epoch

# Test (final model)
python script.py \
    --model deepseek-coder-6.7b-instruct \
    --bits 4 \
    --data_file data \
    --mode test \
    --use_lora \
    --exp code_vh_cg_nod
```

### Clone Detection

```bash
python script_clone.py \
    --model codellama-7b-instruct \
    --bits 4 \
    --data_file clone_detection \
    --mode train \
    --use_lora \
    --exp code_vh_cg_nod \
    --batch_size 2
```

### Code Classification

```bash
python script_class.py \
    --model qwen25-coder-7b \
    --bits 4 \
    --data_file classification \
    --mode train \
    --use_lora \
    --exp code_vh_cg_nod \
    --batch_size 2
```

### Output directory structure

Fine-tuning outputs are written to:

```
model/<model>-<exp>-<bits>bit-lora-<data_file>/   ← saved model weights + adapter
output/<model>-<exp>-<bits>bit-lora-<data_file>/  ← test.out + test.gold + metrics_log.csv
```

---

## Evaluation

After inference or fine-tuning, compute metrics from the output files:

```bash
# BLEU / ROUGE for summarisation
python compute_metrics.py --output_dir output/<run_name>/

# F1 / Precision / Recall for classification tasks
python eval_metrics.py --pred output/<run_name>/test.out --gold output/<run_name>/test.gold

# Post-process LLM outputs before evaluation (removes chat artifacts)
python eval_metrics_postproc.py --input output/<run_name>/test.out \
                                 --output output/<run_name>/test.clean.out
```

---

## Data Folder Structure

```
data/
├── data/                     ← CodeSearchNet-Java (code summarisation)
│   ├── train.jsonl
│   ├── valid.jsonl
│   └── test.jsonl
├── clone_detection/          ← SeSaMe (clone detection)
│   ├── train.pkl
│   ├── valid.pkl
│   └── test.pkl
├── classification/           ← SeSaMe (code classification)
│   ├── train.pkl
│   ├── valid.pkl
│   └── test.pkl
└── vuln/                     ← Vul4J (vulnerability detection)
    ├── train.jsonl
    ├── valid.jsonl
    └── test.jsonl
```

The augmented `.jsonl` and `.pkl` files are available on Figshare (DOI: https://doi.org/10.6084/m9.figshare.30296542). Place them in the corresponding subfolders before running scripts.

**Note on Vul4J data:** The Vul4J augmented dataset is prepared during the LLM data pipeline stage. Version history and call-graph context for Vul4J methods are extracted and merged into the `.jsonl` files distributed on Figshare. The mining scripts in `Classification/mining/` target SeSaMe and CodeSearchNet-Java; Vul4J context is handled separately at the dataset preparation stage.
