import os
import gc
import json
import pandas as pd
import torch
import wandb
import argparse
import shutil
import glob
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import (
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    AutoModelForCausalLM
)
from peft import get_peft_model, PeftModel, prepare_model_for_kbit_training
from transformers.trainer_utils import set_seed

# Import model configurations from your existing registry
from models import get_model_config, list_available_models

set_seed(42)

# A100 Optimizations
torch.backends.cuda.enable_cudnn_sdp(False) 
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# ------------------------------------------------------------------------------
# Smart "Fill & Spill" Prompt Builder
# ------------------------------------------------------------------------------
def build_clone_ablation_input(row, exp_type, tokenizer, max_total_tokens=3800):
    """
    Builds the clone prompt and smartly balances tokens between Method X and Method Y.
    """
    # 1. Gather Method X Strings
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
    
    # 2. Gather Method Y Strings
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

    # 3. Tokenize to measure lengths
    x_ids = tokenizer.encode(x_text, add_special_tokens=False)
    y_ids = tokenizer.encode(y_text, add_special_tokens=False)
    
    # 4. Smart Balance (Fill & Spill Logic)
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
            
    # 5. Decode back to text
    x_final = tokenizer.decode(x_ids)
    y_final = tokenizer.decode(y_ids)
    
    # 6. Assemble Final Prompt
    # # prompt = "Determine if the following two Java methods are code clones. Answer strictly with 'Yes' or 'No'.\n\n"
    # # prompt += x_final + "\n\n====================\n\n" + y_final
    # return prompt

    pair_data = x_final + "\n\n====================\n\n" + y_final
    return pair_data

# ------------------------------------------------------------------------------
# Dataset Class (For Training Phase)
# ------------------------------------------------------------------------------
class CodeCloneDataset(Dataset):
    def __init__(self, pkl_file, tokenizer, model_config, exp_type="baseline"):
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.exp_type = exp_type
        self.examples = []
        
        print(f"==== Loading Clone Dataset: {pkl_file} (Experiment: {exp_type}) ====")
        df = pd.read_pickle(pkl_file)
             
        for _, row in df.iterrows():
            input_text = build_clone_ablation_input(row, self.exp_type, self.tokenizer)
            
            # Map 1 -> "Yes", 0 -> "No"
            label_int = int(row['label'])
            target_text = "Yes" if label_int > 0 else "No"
            
            self.examples.append({'input_text': input_text, 'target': target_text})
            
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        
        prompt = self.model_config.format_prompt_clone_detection(example['input_text'], tokenizer=self.tokenizer)
        full_text = prompt + example['target']
        
        encoding = self.tokenizer(
            full_text,
            max_length=4096,
            padding=False,
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = input_ids.clone()
        
        prompt_length = len(self.tokenizer(prompt, add_special_tokens=False)['input_ids'])
        labels[:prompt_length] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

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
# Phase 1: Training
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Phase 1: Training
# ------------------------------------------------------------------------------
def train_clone_model(model_name, bits, train_file, valid_file, output_dir, exp_type="baseline", batch_size=1):
    model_config = get_model_config(model_name, use_lora=True)
    
    # Dynamic Math for VRAM Optimization
    eval_batch_size = batch_size * 2
    grad_accum = max(1, 32 // batch_size) # Keeps effective batch size at ~32
    
    wandb.init(
        project="sesame-clone-detection",
        name=f"{model_name}-{exp_type}-{bits}bit",
        config={
            "model": model_name, 
            "exp_type": exp_type, 
            "train_batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "gradient_accumulation_steps": grad_accum,
            "num_epochs": 10
        }
    )
    
    tokenizer = model_config.get_tokenizer()
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    quantization_config = get_quantization_config(bits)
    model = model_config.get_model(quantization_config)
    model = prepare_model_for_kbit_training(model)
    lora_config = model_config.get_lora_config()
    model = get_peft_model(model, lora_config)
    
    train_dataset = CodeCloneDataset(train_file, tokenizer, model_config, exp_type=exp_type)
    eval_dataset = CodeCloneDataset(valid_file, tokenizer, model_config, exp_type=exp_type)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,      # Dynamic
        per_device_eval_batch_size=eval_batch_size,  # Dynamic
        gradient_accumulation_steps=grad_accum,      # Dynamic
        bf16=True,
        gradient_checkpointing=True,
        learning_rate=2e-5,
        warmup_steps=300,
        num_train_epochs=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=5, # forced wandb logging every 5 steps instead of default 500
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    print(f"\n--- Starting Training ({model_name} | Exp: {exp_type}) ---")
    trainer.train()
    
    best_ckpt = trainer.state.best_model_checkpoint
    print(f"\n>>> Best model found at: {best_ckpt}")
    
    print(f"Saving clean best model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    adapter_path = os.path.join(output_dir, 'adapter')
    model.save_pretrained(adapter_path)
    
    print("Cleaning up bulky checkpoint folders...")
    for ckpt_folder in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        try:
            shutil.rmtree(ckpt_folder)
        except Exception as e:
            pass

    # clear GPU memory after training before moving to testing phase    
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return output_dir

# ------------------------------------------------------------------------------
# Phase 2: Testing & Evaluation
# ------------------------------------------------------------------------------
def test_clone_model(base_model_name, checkpoint_dir, test_file, exp_type, bits=4):
    print(f"\n--- Starting Testing Phase ---")
    
    # 1. STRICT MEMORY CLEARING BEFORE TEST
    print("Clearing GPU Memory...")
    gc.collect()
    torch.cuda.empty_cache()
    
    # 2. Setup Base Model & Adapters
    model_config = get_model_config(base_model_name, use_lora=True)
    tokenizer = model_config.get_tokenizer()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    quant_config = get_quantization_config(bits)
    
    print(f"Loading Base Model for Generation...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_config.base_model,
        quantization_config=quant_config,
        device_map="auto"
    )
    
    adapter_path = os.path.join(checkpoint_dir, 'adapter')
    print(f"Injecting Trained LoRA Adapters from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    # 3. Load Data
    df = pd.read_pickle(test_file)
    y_true = []
    y_pred = []
    results_list = [] # For per-item logging
    
    # 4. Generate Predictions
    print(f"Evaluating {len(df)} pairs...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        input_text = build_clone_ablation_input(row, exp_type, tokenizer)
        prompt = model_config.format_prompt_clone_detection(input_text, tokenizer=tokenizer)
        
        # inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
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
                max_new_tokens=50, 
                num_beams=1, 
                pad_token_id=tokenizer.pad_token_id,
            )
            
        generated_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()
        
        # Safer binary logic: looks for "yes" inside the string
        gen_lower = generated_text.lower()
        pred_label = 1 if 'yes' in gen_lower else 0 
        true_label = 1 if int(row['label']) > 0 else 0
        
        y_pred.append(pred_label)
        y_true.append(true_label)

        # Save explicit debugging data for error analysis
        results_list.append({
            "id": row.get('id', index), 
            "true_label": true_label,
            "predicted_label": pred_label,
            "raw_output": generated_text,
            "prompt": prompt
        })
        
        # clear GPU memory after each iteration to prevent OOM
        del inputs, outputs
        torch.cuda.empty_cache()

    # 5. Calculate Metrics (Rounded to 5 decimal places)
    f1 = round(f1_score(y_true, y_pred), 5)
    prec = round(precision_score(y_true, y_pred, zero_division=0), 5)
    rec = round(recall_score(y_true, y_pred, zero_division=0), 5)
    acc = round(accuracy_score(y_true, y_pred), 5)
    
    print("\n" + "="*50)
    print(f"RESULTS FOR: {base_model_name} | EXP: {exp_type}")
    print("="*50)
    print(f"F1 Score:  {f1:.5f}")
    print(f"Precision: {prec:.5f}")
    print(f"Recall:    {rec:.5f}")
    print(f"Accuracy:  {acc:.5f}")
    print("="*50)
    
    # Save to disk
    results = {
        "F1": f1, 
        "Precision": prec, 
        "Recall": rec, 
        "Accuracy": acc
    }
    
    with open(os.path.join(checkpoint_dir, "test_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    # Save detailed predictions (mirroring the zeroshot experiment structure)
    predictions_path = os.path.join(checkpoint_dir, "test_predictions.jsonl")
    with open(predictions_path, "w") as f:
        for result in results_list:
            f.write(json.dumps(result) + '\n')
        
    if wandb.run is not None:
        wandb.log({"test/f1": f1, "test/precision": prec, "test/recall": rec, "test/accuracy": acc})
    
    print(f"Metrics saved to {checkpoint_dir}/test_metrics.json")
    print(f"Detailed predictions saved to {predictions_path}")

# ------------------------------------------------------------------------------
# Main Execution Block
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test', 'both'])
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--bits', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1) 
    parser.add_argument('--train_file', type=str, default='./data/clone_detection/train_blocks.pkl')
    parser.add_argument('--valid_file', type=str, default='./data/clone_detection/dev_blocks.pkl')
    parser.add_argument('--test_file', type=str, default='./data/clone_detection/test_blocks.pkl')
    parser.add_argument('--exp', type=str, default='baseline')
    
    args = parser.parse_args()
    output_dir = f"model_clone/{args.model}-{args.exp}-{args.bits}bit"
    
    if args.mode in ['train', 'both']:
        # Passed args.batch_size here
        train_clone_model(args.model, args.bits, args.train_file, args.valid_file, output_dir, args.exp, args.batch_size)
        
    if args.mode in ['test', 'both']:
        test_clone_model(args.model, output_dir, args.test_file, args.exp, args.bits)
        
    if args.mode == 'both':
        wandb.finish()

if __name__ == '__main__':
    main()