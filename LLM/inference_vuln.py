import os
import csv
import gc
import json
import torch
import wandb
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import LogitsProcessor, LogitsProcessorList

class TargetBiasProcessor(LogitsProcessor):
    """
    A manual logits processor to apply a bias to specific tokens.
    Ensures compatibility across all transformers versions.
    """
    def __init__(self, bias_dict):
        self.bias_dict = bias_dict

    def __call__(self, input_ids, scores):
        for token_id, bias in self.bias_dict.items():
            scores[:, token_id] += bias
        return scores

# Import model configurations from your existing registry
from models import get_model_config

# A100/H100 Optimizations
torch.backends.cuda.enable_cudnn_sdp(False) 
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

import random
import numpy as np
from transformers.trainer_utils import set_seed

# 1. Standard library and Transformers seeds
def set_reproducibility(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed) # Transformers specific seed helper
    
    # 2. CUDA Determinism (Slightly slower but 100% reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call this at the start of your main() or run_vuln_zeroshot()
set_reproducibility(42)

def append_to_master_csv(file_path, model_name, exp_type, metrics):
    """
    Appends a single run's results to a master CSV file.
    Creates the file and header if it doesn't exist.
    """
    file_exists = os.path.isfile(file_path)
    header = ["Model", "Experiment", "F1", "Precision", "Recall", "Accuracy"]
    
    with open(file_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            "Model": model_name,
            "Experiment": exp_type,
            "F1": f"{metrics['F1']:.5f}",
            "Precision": f"{metrics['Precision']:.5f}",
            "Recall": f"{metrics['Recall']:.5f}",
            "Accuracy": f"{metrics['Accuracy']:.5f}"
        })
        
# ------------------------------------------------------------------------------
# Core-First Truncation & Prompt Builder
# ------------------------------------------------------------------------------
def build_vuln_ablation_input(item, exp_type, tokenizer, max_total_tokens=3800):
    """
    Refined input builder: Extracts raw code from version_history dictionaries.
    Prioritizes Source Code and Call Graph within the token budget.
    """
    # 1. Primary code block (The version currently being audited)
    target_code = f"### Source Code\n{item.get('code', '')}"
    
    # 2. Structural context (Call Graph)
    cg_parts = []
    if 'cg' in exp_type:
        if item.get('caller_code'): 
            cg_parts.append(f"### Caller Context\n{item['caller_code']}")
        if item.get('callee_code'): 
            cg_parts.append(f"### Callee Context\n{item['callee_code']}")
    cg_text = "\n\n".join(cg_parts)

    # 3. Temporal signal (Method Age)
    age_text = ""
    if 'nod' in exp_type and item.get('method_age_days') is not None:
        age_text = f"### Method Age\n{item.get('method_age_days')} days"

    # Tokenize high-priority items first
    target_ids = tokenizer.encode(target_code, add_special_tokens=False)
    cg_ids = tokenizer.encode(cg_text, add_special_tokens=False)
    age_ids = tokenizer.encode(age_text, add_special_tokens=False)
    
    used_tokens = len(target_ids) + len(cg_ids) + len(age_ids)
    remaining_budget = max_total_tokens - used_tokens
    
    # 4. Process Version History (VH) - Extracting 'code' from list of dicts
    vh_text = ""
    if 'vh' in exp_type:
        vh_raw = item.get('version_history', [])
        
        # Handle potential stringified JSON from the JSONL loader
        if isinstance(vh_raw, str):
            try:
                vh_raw = json.loads(vh_raw)
            except:
                vh_raw = []

        vh_entries = []
        if isinstance(vh_raw, list):
            for i, v_dict in enumerate(vh_raw):
                # We only want the code string from the dictionary
                if isinstance(v_dict, dict):
                    past_code = v_dict.get('code', '')
                    if past_code.strip():
                        vh_entries.append(f"### Version {i+1} (Past)\n{past_code}")
        
        if vh_entries:
            full_vh_text = "\n\n".join(vh_entries)
            vh_ids = tokenizer.encode(full_vh_text, add_special_tokens=False)
            
            # Truncate VH if it exceeds the remaining budget
            if len(vh_ids) > remaining_budget:
                vh_text = tokenizer.decode(vh_ids[:max(0, remaining_budget)])
            else:
                vh_text = full_vh_text

    # 5. Final Scenario Assembly
    parts = []
    if exp_type == 'baseline':
        parts = [target_code]
    elif exp_type == 'code_cg':
        parts = [target_code, cg_text]
    elif exp_type == 'code_vh':
        parts = [target_code, vh_text]
    elif exp_type == 'code_vh_nod':
        parts = [target_code, vh_text, age_text]
    elif exp_type == 'code_cg_vh':
        parts = [target_code, cg_text, vh_text]
    elif exp_type == 'code_vh_cg':
        parts = [target_code, vh_text, cg_text]
    elif exp_type == 'code_cg_vh_nod':
        parts = [target_code, cg_text, vh_text, age_text]
    elif exp_type == 'code_vh_cg_nod':
        parts = [target_code, vh_text, cg_text, age_text]

    return "\n\n".join([p for p in parts if p.strip()])

# ------------------------------------------------------------------------------
# Zero-Shot Inference Engine
# ------------------------------------------------------------------------------
# def run_vuln_zeroshot(model_name, dataset_file, exp_type, bits=4):
#     output_dir = f"model_vuln/{model_name}-{exp_type}-zeroshot"
#     os.makedirs(output_dir, exist_ok=True)
    
#     wandb.init(
#         project="vul4j-vulnerability-detection",
#         name=f"{model_name}-{exp_type}-zeroshot",
#         config={"model": model_name, "exp_type": exp_type, "dataset": dataset_file}
#     )

#     print(f"\n--- Starting Zero-Shot Vulnerability Detection ---")
#     gc.collect()
#     torch.cuda.empty_cache()
    
#     model_config = get_model_config(model_name, use_lora=False)
#     tokenizer = model_config.get_tokenizer()
#     tokenizer.padding_side = "left"
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token 

#     quant_config = None
#     if bits == 4:
#         quant_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.bfloat16,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type="nf4"
#         )
    
#     print(f"Loading Base Model: {model_config.base_model}...")
#     model = AutoModelForCausalLM.from_pretrained(
#         model_config.base_model,
#         quantization_config=quant_config,
#         device_map="auto",
#         trust_remote_code=True
#     )
#     model.eval()
    
#     # Loading JSONL directly as requested
#     data_list = []
#     with open(dataset_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             data_list.append(json.loads(line))
            
#     y_true, y_pred, results_list = [], [], []
    
#     print(f"Evaluating {len(data_list)} methods...")
#     for item in tqdm(data_list):
#         input_text = build_vuln_ablation_input(item, exp_type, tokenizer)
#         prompt = model_config.format_prompt_vulnerability(input_text, tokenizer=tokenizer)
        
#         inputs = tokenizer(
#             prompt, 
#             return_tensors='pt',
#             truncation=True,
#             max_length=4096
#         ).to(model.device)
#         prompt_length = inputs['input_ids'].shape[-1]

#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=20, # Limited for strict binary output
#                 num_beams=1, 
#                 do_sample=False,
#                 pad_token_id=tokenizer.pad_token_id,
#             )
            
#         generated_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()
        
#         # Robust binary parsing logic
#         gen_lower = generated_text.lower()
#         if 'yes' in gen_lower:
#             pred_label = 1
#         elif 'no' in gen_lower:
#             pred_label = 0
#         else:
#             pred_label = 0 # Default to Clean if ambiguous
                
#         true_label = int(item['label'])
        
#         y_pred.append(pred_label)
#         y_true.append(true_label)
        
#         results_list.append({
#             "vul_id": item.get('vul_id', 'unknown'),
#             "sample_type": item.get('sample_type'),
#             "true_label": true_label,
#             "predicted_label": pred_label,
#             "raw_output": generated_text,
#             "prompt": prompt
#         })

#         del inputs, outputs
#         torch.cuda.empty_cache()
        
#     # Calculate performance metrics
#     f1 = round(f1_score(y_true, y_pred), 5)
#     prec = round(precision_score(y_true, y_pred, zero_division=0), 5)
#     rec = round(recall_score(y_true, y_pred, zero_division=0), 5)
#     acc = round(accuracy_score(y_true, y_pred), 5)
    
#     print("\n" + "="*50)
#     print(f"RESULTS: {model_name} | EXP: {exp_type}")
#     print("="*50)
#     print(f"F1 Score:  {f1:.5f}")
#     print(f"Precision: {prec:.5f}")
#     print(f"Recall:    {rec:.5f}")
#     print(f"Accuracy:  {acc:.5f}")
#     print("="*50)
    
#     # Save local results
#     metrics = {"F1": f1, "Precision": prec, "Recall": rec, "Accuracy": acc}
#     with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
#         json.dump(metrics, f, indent=4)
        
#     with open(os.path.join(output_dir, "test_predictions.jsonl"), "w") as f:
#         for result in results_list:
#             f.write(json.dumps(result) + '\n')
            
#     wandb.log(metrics)
#     wandb.finish()

def run_vuln_zeroshot(model_name, dataset_file, exp_type, bits=4):
    output_dir = f"model_vuln/{model_name}-{exp_type}-zeroshot"
    os.makedirs(output_dir, exist_ok=True)
    
    wandb.init(
        project="vul4j-vulnerability-detection",
        name=f"{model_name}-{exp_type}-zeroshot",
        config={"model": model_name, "exp_type": exp_type, "dataset": dataset_file}
    )

    print(f"\n--- Starting Zero-Shot Vulnerability Detection ---")
    gc.collect()
    torch.cuda.empty_cache()
    
    model_config = get_model_config(model_name, use_lora=False)
    tokenizer = model_config.get_tokenizer()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    quant_config = None
    if bits == 4:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    print(f"Loading Base Model: {model_config.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_config.base_model,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    # Loading JSONL directly as requested
    data_list = []
    with open(dataset_file, 'r', encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line))
            
    y_true, y_pred, results_list = [], [], []
    
    # --- CALIBRATION ROUND 3: Fine-Tuning the Mid-Range ---
    BIAS_TARGETS = {
        "deepseek-coder-6.7b-instruct": 2.4, 
        "codellama-34b-instruct": 0.6,
        "qwen25-coder-14b": 0.8,
        "codellama-13b-instruct": -1.0,
        
        # Trialing these two now:
        "qwen25-coder-7b": 1.2,         # High positive to force context usage
        "codellama-7b-instruct": -1.8,  # High negative to stop "Yes" saturation
    }

    # Initialize an empty processor list
    logits_processor = LogitsProcessorList()
    
    if model_name in BIAS_TARGETS:
        bias_value = BIAS_TARGETS[model_name]
        yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[-1]
        
        # Add our custom processor to the list
        logits_processor.append(TargetBiasProcessor({yes_token_id: bias_value}))
        print(f"!!! Manual Logit Bias of {bias_value} applied to token {yes_token_id} !!!")

    print(f"Evaluating {len(data_list)} methods...")
    for item in tqdm(data_list):
        input_text = build_vuln_ablation_input(item, exp_type, tokenizer)
        prompt = model_config.format_prompt_vulnerability(input_text, tokenizer=tokenizer)
        
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=4096).to(model.device)
        prompt_length = inputs['input_ids'].shape[-1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                num_beams=1,
                do_sample=False,
                logits_processor=logits_processor, 
                pad_token_id=tokenizer.pad_token_id,
            )
            
        generated_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()
        
        # Robust binary parsing logic
        gen_lower = generated_text.lower()
        if 'yes' in gen_lower:
            pred_label = 1
        elif 'no' in gen_lower:
            pred_label = 0
        else:
            pred_label = 0 # Default to Clean if ambiguous
                
        true_label = int(item['label'])
        
        y_pred.append(pred_label)
        y_true.append(true_label)
        
        results_list.append({
            "vul_id": item.get('vul_id', 'unknown'),
            "sample_type": item.get('sample_type'),
            "true_label": true_label,
            "predicted_label": pred_label,
            "raw_output": generated_text,
            "prompt": prompt
        })

        del inputs, outputs
        torch.cuda.empty_cache()
        
    # Calculate performance metrics
    f1 = round(f1_score(y_true, y_pred), 5)
    prec = round(precision_score(y_true, y_pred, zero_division=0), 5)
    rec = round(recall_score(y_true, y_pred, zero_division=0), 5)
    acc = round(accuracy_score(y_true, y_pred), 5)
    
    print("\n" + "="*50)
    print(f"RESULTS: {model_name} | EXP: {exp_type}")
    print("="*50)
    print(f"F1 Score:  {f1:.5f}")
    print(f"Precision: {prec:.5f}")
    print(f"Recall:    {rec:.5f}")
    print(f"Accuracy:  {acc:.5f}")
    print("="*50)
    
    # Finalize metrics dictionary
    metrics = {
        "F1": f1, 
        "Precision": prec, 
        "Recall": rec, 
        "Accuracy": acc
    }

    # --- NEW: Append to Master CSV ---
    # You can change this path to a centralized location in your project
    master_csv_path = "vulnerability_results_master.csv"
    append_to_master_csv(master_csv_path, model_name, exp_type, metrics)
    print(f"--> Results successfully appended to {master_csv_path}")
    # ---------------------------------

    # Save local results
    metrics = {"F1": f1, "Precision": prec, "Recall": rec, "Accuracy": acc}
    with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    with open(os.path.join(output_dir, "test_predictions.jsonl"), "w") as f:
        for result in results_list:
            f.write(json.dumps(result) + '\n')
            
    wandb.log(metrics)
    wandb.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Registry name (e.g., qwen25-coder-7b)")
    parser.add_argument('--exp', type=str, default='baseline', help="Ablation scenario")
    parser.add_argument('--bits', type=int, default=4)
    parser.add_argument('--dataset', type=str, default='./data/vuln/vul4j_augmented.jsonl')
    
    args = parser.parse_args()
    run_vuln_zeroshot(args.model, args.dataset, args.exp, args.bits)

if __name__ == '__main__':
    main()