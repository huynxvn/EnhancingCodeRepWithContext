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

# ------------------------------------------------------------------------------
# Prompt Builder (Identical to Fine-Tuning)
# ------------------------------------------------------------------------------
def build_clone_ablation_input(row, exp_type, tokenizer, max_total_tokens=3800):
    x_parts = [f"### Method 1 Source Code\n{row['t_code_x']}"]
    if 'vh' in exp_type and pd.notna(row.get('t_code_versions_all_x')) and row['t_code_versions_all_x']:
        x_parts.append(f"### Method 1 Version History (Past Iterations)\n{row['t_code_versions_all_x']}")
    if 'cg' in exp_type:
        if pd.notna(row.get('t_calling_x')) and row['t_calling_x']: 
            x_parts.append(f"### Method 1 Caller Context\n{row['t_calling_x']}")
        if pd.notna(row.get('t_called_x')) and row['t_called_x']: 
            x_parts.append(f"### Method 1 Callee Context\n{row['t_called_x']}")
    if 'nod' in exp_type and pd.notna(row.get('t_number_of_days_x')):
        x_parts.append(f"### Method 1 Age\n{row['t_number_of_days_x']} days")
    
    y_parts = [f"### Method 2 Source Code\n{row['t_code_y']}"]
    if 'vh' in exp_type and pd.notna(row.get('t_code_versions_all_y')) and row['t_code_versions_all_y']:
        y_parts.append(f"### Method 2 Version History (Past Iterations)\n{row['t_code_versions_all_y']}")
    if 'cg' in exp_type:
        if pd.notna(row.get('t_calling_y')) and row['t_calling_y']: 
            y_parts.append(f"### Method 2 Caller Context\n{row['t_calling_y']}")
        if pd.notna(row.get('t_called_y')) and row['t_called_y']: 
            y_parts.append(f"### Method 2 Callee Context\n{row['t_called_y']}")
    if 'nod' in exp_type and pd.notna(row.get('t_number_of_days_y')):
        y_parts.append(f"### Method 2 Age\n{row['t_number_of_days_y']} days")

    x_text = "\n\n".join(x_parts)
    y_text = "\n\n".join(y_parts)

    x_ids = tokenizer.encode(x_text, add_special_tokens=False)
    y_ids = tokenizer.encode(y_text, add_special_tokens=False)
    
    len_x, len_y = len(x_ids), len(y_ids)
    if len_x + len_y > max_total_tokens:
        half = max_total_tokens // 2
        if len_x <= half:
            y_ids = y_ids[:max_total_tokens - len_x]
        elif len_y <= half:
            x_ids = x_ids[:max_total_tokens - len_y]
        else:
            x_ids = x_ids[:half]
            y_ids = y_ids[:half]
            
    x_final = tokenizer.decode(x_ids)
    y_final = tokenizer.decode(y_ids)
    
    # Just return the raw code blocks separated by a divider
    pair_data = x_final + "\n\n====================\n\n" + y_final
    return pair_data

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
    output_dir = f"model_clone/{model_name}-{exp_type}-zeroshot"
    os.makedirs(output_dir, exist_ok=True)
    
    wandb.init(
        project="sesame-clone-detection",
        name=f"{model_name}-{exp_type}-zeroshot",
        config={"model": model_name, "exp_type": exp_type, "dataset": dataset_file, "type": "zeroshot"}
    )

    print(f"\n--- Starting Zero-Shot Testing Phase ---")
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
        device_map="auto"
    )
    model.eval()
    
    df = pd.read_pickle(dataset_file)
    y_true = []
    y_pred = []
    results_list = []
    
    print(f"Evaluating {len(df)} pairs...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        input_text = build_clone_ablation_input(row, exp_type, tokenizer)
        prompt = model_config.format_prompt_clone_detection(input_text, tokenizer=tokenizer)
        
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        prompt_length = inputs['input_ids'].shape[-1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50, 
                num_beams=1, 
                do_sample=False,  # <--- FIX for CodeLlama-7B: Forces greedy decoding
                pad_token_id=tokenizer.pad_token_id,
            )
            
        # generated_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()
        
        # # Binary logic identical to fine-tuning fix
        # pred_label = 1 if generated_text.lower().startswith('yes') else 0
        # true_label = 1 if int(row['label']) > 0 else 0 

        generated_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()
        
        # Safer binary logic: looks for "yes" inside the string
        gen_lower = generated_text.lower()
        pred_label = 1 if 'yes' in gen_lower else 0 
        true_label = 1 if int(row['label']) > 0 else 0
        
        y_pred.append(pred_label)
        y_true.append(true_label)
        
        # Save explicit debugging data
        results_list.append({
            "id": row.get('id', index), 
            "true_label": true_label,
            "predicted_label": pred_label,
            "raw_output": generated_text,
            "prompt": prompt
        })
        
    # Calculate metrics with 5 decimal rounding
    f1 = round(f1_score(y_true, y_pred), 5)
    prec = round(precision_score(y_true, y_pred, zero_division=0), 5)
    rec = round(recall_score(y_true, y_pred, zero_division=0), 5)
    acc = round(accuracy_score(y_true, y_pred), 5)
    
    print("\n" + "="*50)
    print(f"ZERO-SHOT RESULTS FOR: {model_name} | EXP: {exp_type}")
    print("="*50)
    print(f"F1 Score:  {f1:.5f}")
    print(f"Precision: {prec:.5f}")
    print(f"Recall:    {rec:.5f}")
    print(f"Accuracy:  {acc:.5f}")
    print("="*50)
    
    # Save to disk with 5 decimal places
    metrics = {
        "F1": f1, 
        "Precision": prec, 
        "Recall": rec, 
        "Accuracy": acc
    }
    
    metrics_path = os.path.join(output_dir, f"{model_name}-{exp_type}-zeroshot-metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    predictions_path = os.path.join(output_dir, f"{model_name}-{exp_type}-zeroshot-predictions.jsonl")
    with open(predictions_path, "w") as f:
        for result in results_list:
            f.write(json.dumps(result) + '\n')
            
    wandb.log({"test/f1": f1, "test/precision": prec, "test/recall": rec, "test/accuracy": acc})
    wandb.finish()
    
    print(f"Metrics saved to {metrics_path}")
    print(f"Detailed predictions saved to {predictions_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--bits', type=int, default=4)
    parser.add_argument('--dataset_file', type=str, default='./data/clone_detection/test_blocks.pkl')
    parser.add_argument('--exp', type=str, default='baseline')
    
    args = parser.parse_args()
    
    run_zeroshot(args.model, args.dataset_file, args.exp, args.bits)

if __name__ == '__main__':
    main()