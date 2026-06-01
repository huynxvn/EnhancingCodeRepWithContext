# Summarization — Code Summarisation

This folder contains all source code for **Code Summarisation** experiments using four task-specific small language models: **CodeBERT**, **GraphCodeBERT**, **CodeT5**, and **PLBART**. Each model is evaluated across multiple context combinations (version history, call-graph, method age) on the CodeSearchNet-Java dataset.

> **Code Summarisation with LLMs** (CodeLlama, DeepSeek-Coder, Qwen2.5-Coder) is handled separately. See [`LLM/`](../LLM/) and its [`README`](../LLM/README.md).

For clone detection and code classification experiments, see [`Classification/`](../Classification/).  
For the full replication package overview, see the [root README](../README.md).

---

## Models

| Model | Folder | Architecture |
|---|---|---|
| CodeBERT | `Task/Code-Summarization/codebert/` | Encoder-Decoder (RoBERTa-based) |
| GraphCodeBERT | `Task/Code-Summarization/graphcodebert/` | Encoder-Decoder (RoBERTa-based) |
| CodeT5 | `Task/Code-Summarization/codet5/` | Encoder-Decoder (T5-based) |
| PLBART | `Task/Code-Summarization/plbart/` | Encoder-Decoder (BART-based) |

---

## Data

Download `Dataset.zip` from Figshare (https://figshare.com/s/71c3233d55c2ad91f30c), unzip it, and copy the `.jsonl` files from the `data/` folder directly into:

```
Summarization/Task/Code-Summarization/dataset/java/
```

The expected files are `train.jsonl`, `valid.jsonl`, and `test.jsonl`. Each record contains the original Java method enriched with version history, caller/callee context, and method age.

---

## Folder Structure

```
Summarization/
├── Task/
│   └── Code-Summarization/
│       ├── codebert/
│       │   ├── run_exp.py         ← main training and evaluation script
│       │   ├── model.py           ← Seq2Seq model definition
│       │   ├── dataset.py         ← dataset loader
│       │   └── bleu.py            ← BLEU evaluation helper
│       ├── graphcodebert/
│       │   ├── run_exp.py
│       │   ├── model.py
│       │   └── bleu.py
│       ├── codet5/
│       │   ├── run_gen_exp.py     ← main training and evaluation script
│       │   ├── models.py
│       │   ├── utils_exp.py
│       │   └── evaluator/         ← BLEU / CodeBLEU evaluation
│       ├── plbart/
│       │   ├── run_exp.py
│       │   ├── model.py
│       │   └── bleu.py
│       ├── evaluator/             ← shared evaluation utilities
│       └── dataset/
│           └── java/              ← place train.jsonl, valid.jsonl, test.jsonl here
└── PLBART/                        ← fairseq-based PLBART setup (alternative)
    ├── scripts/
    ├── evaluation/
    ├── pretrain/
    └── sentencepiece/
```

---

## Environment Setup

```bash
pip install torch transformers datasets rouge-score bert-score nltk
```

Key dependencies: Python 3.9, PyTorch 2.1.0, Transformers 4.38.2.

---

## Running Experiments

Each model folder contains a `run_exp.sh` script that accepts a language and context experiment type as arguments.

### Context experiment types

| `--exp` value | Context included |
|---|---|
| `baseline` | Source code only |
| `code_vh` | Source code + version history |
| `code_cg` | Source code + call graph (caller + callee) |
| `code_vh_nod` | Source code + version history + method age |
| `code_vh_cg` | Source code + version history + call graph |
| `code_cg_vh` | Source code + call graph + version history |
| `code_vh_cg_nod` | Source code + version history + call graph + method age |
| `code_cg_vh_nod` | Source code + call graph + version history + method age |

### CodeBERT

```bash
cd Task/Code-Summarization/codebert
bash ./run_exp.sh java baseline
bash ./run_exp.sh java code_vh
bash ./run_exp.sh java code_cg
bash ./run_exp.sh java code_vh_nod
bash ./run_exp.sh java code_vh_cg
bash ./run_exp.sh java code_cg_vh
bash ./run_exp.sh java code_vh_cg_nod
bash ./run_exp.sh java code_cg_vh_nod
```

### GraphCodeBERT

```bash
cd Task/Code-Summarization/graphcodebert
bash ./run_exp.sh java baseline
# ... same pattern as CodeBERT above
```

### CodeT5

```bash
cd Task/Code-Summarization/codet5
bash ./run_exp.sh java baseline
bash ./run_exp.sh java code_vh
bash ./run_exp.sh java code_cg
bash ./run_exp.sh java code_vh_nod
bash ./run_exp.sh java code_vh_cg
bash ./run_exp.sh java code_cg_vh
bash ./run_exp.sh java code_vh_cg_nod
bash ./run_exp.sh java code_cg_vh_nod
```

### PLBART

```bash
cd Task/Code-Summarization/plbart
bash ./run_exp.sh java baseline
# ... same pattern as CodeBERT above
```

For the fairseq-based PLBART alternative:

```bash
cd PLBART/
bash scripts/run_summarization.sh
```

---

## Evaluation Metrics

Results are evaluated on:

- **BLEU-4** — primary metric, computed via `evaluator/bleu.py`
- **ROUGE-L** — computed via `rouge-score`
- **BERTScore** — computed via `bert-score`
- **METEOR** — computed via `nltk`

Output files are written to the `output_dir` specified in each `run_exp.sh`:
- `dev.output` / `dev.gold` — validation predictions and references
- `test_best-bleu.output` / `test_best-bleu.gold` — test predictions at the best BLEU checkpoint

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
