import os
import gc
import json
import pandas as pd
import torch
import wandb
import argparse
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Import model configurations from your existing registry
from models import get_model_config

# A100 Optimizations
torch.backends.cuda.enable_cudnn_sdp(False) 
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

VALID_PROJECTS = [
    "openjdk11", "deeplearning4j", "eclipse.jdt.core", "guava", 
    "commons-math", "freemind", "commons-collections", 
    "caffeine", "checkstyle", "commons-lang", "trove"
]

# Comprehensive Alias Mapping to catch natural language model outputs
PROJECT_ALIASES = {
    # Eclipse JDT
    "eclipse.jdt.core": "eclipse.jdt.core",
    "eclipse jdt core": "eclipse.jdt.core",
    "eclipse jdt": "eclipse.jdt.core",
    
    # Apache Commons Math
    "commons-math": "commons-math",
    "apache commons math": "commons-math",
    "commons math": "commons-math",
    
    # Apache Commons Collections
    "commons-collections": "commons-collections",
    "apache commons collections": "commons-collections",
    "commons collections": "commons-collections",
    
    # Apache Commons Lang
    "commons-lang": "commons-lang",
    "apache commons lang": "commons-lang",
    "commons lang": "commons-lang",
    
    # OpenJDK
    "openjdk11": "openjdk11",
    "openjdk 11": "openjdk11",
    "open jdk 11": "openjdk11",
    
    # DeepLearning4J
    "deeplearning4j": "deeplearning4j",
    "dl4j": "deeplearning4j",
    
    # Guava
    "guava": "guava",
    "google guava": "guava",
    
    # Trove
    "trove": "trove",
    "gnu trove": "trove",
    "trove4j": "trove",
    
    # Single-word projects with fewer variations
    "freemind": "freemind",
    "caffeine": "caffeine",
    "checkstyle": "checkstyle"
}

# ------------------------------------------------------------------------------
# Context Builder (Ablation Scenarios)
# ------------------------------------------------------------------------------
def build_class_ablation_input(row, exp_type):
    """
    Builds the prompt input by concatenating context in the exact order specified by the ablation scenario.
    Truncation is handled later by the tokenizer.
    """
    parts = [f"### Source Code\n{row.get('t_code', '')}"]
    
    # Pre-fetch available context blocks
    cg_parts = []
    if pd.notna(row.get('t_calling')) and row['t_calling']: 
        cg_parts.append(f"### Caller Context\n{row['t_calling']}")
    if pd.notna(row.get('t_called')) and row['t_called']: 
        cg_parts.append(f"### Callee Context\n{row['t_called']}")
        
    vh_part = f"### Version History (Past Iterations)\n{row['t_code_versions_all']}" if pd.notna(row.get('t_code_versions_all')) and row['t_code_versions_all'] else ""
    nod_part = f"### Method Age\n{row['t_number_of_days']} days" if pd.notna(row.get('t_number_of_days')) else ""
    
    # Append strictly based on experiment order
    if exp_type == 'baseline':
        pass
    elif exp_type == 'code_cg':
        parts.extend(cg_parts)
    elif exp_type == 'code_vh':
        if vh_part: parts.append(vh_part)
    elif exp_type == 'code_vh_nod':
        if vh_part: parts.append(vh_part)
        if nod_part: parts.append(nod_part)
    elif exp_type == 'code_cg_vh':
        parts.extend(cg_parts)
        if vh_part: parts.append(vh_part)
    elif exp_type == 'code_vh_cg':
        if vh_part: parts.append(vh_part)
        parts.extend(cg_parts)
    elif exp_type == 'code_cg_vh_nod':
        parts.extend(cg_parts)
        if vh_part: parts.append(vh_part)
        if nod_part: parts.append(nod_part)
    elif exp_type == 'code_vh_cg_nod':
        if vh_part: parts.append(vh_part)
        parts.extend(cg_parts)
        if nod_part: parts.append(nod_part)
        
    return "\n\n".join(parts)

def get_quantization_config(bits):
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    return None

# ------------------------------------------------------------------------------
# Zero-Shot Execution
# ------------------------------------------------------------------------------
def run_zeroshot(model_name, dataset_file, exp_type, bits=4):
    output_dir = f"model_class/{model_name}-{exp_type}-zeroshot"
    os.makedirs(output_dir, exist_ok=True)
    
    wandb.init(
        project="sesame-code-classification",
        name=f"{model_name}-{exp_type}-zeroshot",
        config={"model": model_name, "exp_type": exp_type, "dataset": dataset_file, "type": "zeroshot"}
    )

    print(f"\n--- Starting Zero-Shot Classification Phase ---")
    gc.collect()
    torch.cuda.empty_cache()
    
    model_config = get_model_config(model_name, use_lora=False)
    tokenizer = model_config.get_tokenizer()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    quant_config = get_quantization_config(bits)
    
    print(f"Loading Raw Base Model: {model_config.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_config.base_model,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    df = pd.read_pickle(dataset_file)
    y_true = []
    y_pred = []
    results_list = []
    
    print(f"Evaluating {len(df)} methods...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        input_text = build_class_ablation_input(row, exp_type)
        prompt = model_config.format_prompt_classification(input_text, tokenizer=tokenizer)
        
        inputs = tokenizer(
            prompt, 
            return_tensors='pt',
            truncation=True,
            max_length=4096
        ).to(model.device)
        prompt_length = inputs['input_ids'].shape[-1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64, 
                num_beams=1, 
                do_sample=False,  # Forces greedy decoding for deterministic classification
                pad_token_id=tokenizer.pad_token_id,
            )
            
        generated_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()
        
        # Robust string matching parser
        gen_lower = generated_text.lower()
        pred_label = "UNKNOWN"

        # Strategy: Find the first occurrence of a valid project name in the text
        # We sort by length descending to ensure 'commons-math' isn't cut off if 'math' was a keyword
        # Sort ALL alias keys by length descending to ensure the longest, most specific match wins
        for alias in sorted(PROJECT_ALIASES.keys(), key=len, reverse=True):
            if alias in gen_lower:
                pred_label = PROJECT_ALIASES[alias]
                break
                
        true_label = str(row['project'])
        
        y_pred.append(pred_label)
        y_true.append(true_label)
        
        # Save explicit debugging data
        results_list.append({
            "id": row.get('id', index), 
            "true_project": true_label,
            "predicted_project": pred_label,
            "raw_output": generated_text,
            "prompt": prompt
        })

        # emty CUDA cache
        del inputs, outputs
        torch.cuda.empty_cache()
        
    # Calculate metrics, prioritizing Macro-F1
    macro_f1 = round(f1_score(y_true, y_pred, average='macro', labels=VALID_PROJECTS), 5)
    weighted_f1 = round(f1_score(y_true, y_pred, average='weighted', labels=VALID_PROJECTS), 5)
    macro_prec = round(precision_score(y_true, y_pred, average='macro', zero_division=0, labels=VALID_PROJECTS), 5)
    macro_rec = round(recall_score(y_true, y_pred, average='macro', zero_division=0, labels=VALID_PROJECTS), 5)
    acc = round(accuracy_score(y_true, y_pred), 5)
    
    print("\n" + "="*50)
    print(f"ZERO-SHOT RESULTS FOR: {model_name} | EXP: {exp_type}")
    print("="*50)
    print(f"Macro-F1:    {macro_f1:.5f}")
    print(f"Weighted-F1: {weighted_f1:.5f}")
    print(f"Precision:   {macro_prec:.5f}")
    print(f"Recall:      {macro_rec:.5f}")
    print(f"Accuracy:    {acc:.5f}")
    print("="*50)
    
    # Save to disk
    metrics = {
        "Macro_F1": macro_f1,
        "Weighted_F1": weighted_f1,
        "Macro_Precision": macro_prec, 
        "Macro_Recall": macro_rec, 
        "Accuracy": acc
    }
    
    metrics_path = os.path.join(output_dir, f"{model_name}-{exp_type}-zeroshot-metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    predictions_path = os.path.join(output_dir, f"{model_name}-{exp_type}-zeroshot-predictions.jsonl")
    with open(predictions_path, "w") as f:
        for result in results_list:
            f.write(json.dumps(result) + '\n')
            
    wandb.log({
        "test/macro_f1": macro_f1, 
        "test/weighted_f1": weighted_f1,
        "test/macro_precision": macro_prec, 
        "test/macro_recall": macro_rec, 
        "test/accuracy": acc
    })
    wandb.finish()
    
    print(f"Metrics saved to {metrics_path}")
    print(f"Detailed predictions saved to {predictions_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--bits', type=int, default=4)
    parser.add_argument('--dataset_file', type=str, default='./data/classification/test_df.pkl')
    parser.add_argument('--exp', type=str, default='baseline')
    
    args = parser.parse_args()
    
    run_zeroshot(args.model, args.dataset_file, args.exp, args.bits)

if __name__ == '__main__':
    main()