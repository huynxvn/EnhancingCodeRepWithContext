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

from models import get_model_config

set_seed(42)

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

# ------------------------------------------------------------------------------
# Dataset Class
# ------------------------------------------------------------------------------
class CodeClassificationDataset(Dataset):
    def __init__(self, pkl_file, tokenizer, model_config, exp_type="baseline"):
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.exp_type = exp_type
        self.examples = []
        
        print(f"==== Loading Classification Dataset: {pkl_file} (Experiment: {exp_type}) ====")
        df = pd.read_pickle(pkl_file)
             
        for _, row in df.iterrows():
            input_text = build_class_ablation_input(row, self.exp_type)
            
            # Target is strictly the project name string
            target_text = str(row['project'])
            
            self.examples.append({'input_text': input_text, 'target': target_text})
            
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        
        prompt = self.model_config.format_prompt_classification(example['input_text'], tokenizer=self.tokenizer)
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
        # Ignore prompt tokens in the loss calculation
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
def train_class_model(model_name, bits, train_file, valid_file, output_dir, exp_type="baseline", batch_size=1):
    model_config = get_model_config(model_name, use_lora=True)
    
    eval_batch_size = batch_size * 2
    grad_accum = max(1, 32 // batch_size) 
    
    wandb.init(
        project="sesame-code-classification",
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
    
    train_dataset = CodeClassificationDataset(train_file, tokenizer, model_config, exp_type=exp_type)
    eval_dataset = CodeClassificationDataset(valid_file, tokenizer, model_config, exp_type=exp_type)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        gradient_checkpointing=True,
        learning_rate=2e-5,
        warmup_steps=300,
        num_train_epochs=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=5,
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
    
    print(f"\n--- Starting Classification Training ({model_name} | Exp: {exp_type}) ---")
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
    
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return output_dir

# ------------------------------------------------------------------------------
# Phase 2: Testing & Evaluation
# ------------------------------------------------------------------------------
def test_class_model(base_model_name, checkpoint_dir, test_file, exp_type, bits=4):
    print(f"\n--- Starting Testing Phase ---")
    
    gc.collect()
    torch.cuda.empty_cache()
    
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
    
    df = pd.read_pickle(test_file)
    y_true = []
    y_pred = []
    results_list = []
    
    print(f"Evaluating {len(df)} methods...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        input_text = build_class_ablation_input(row, exp_type)
        prompt = model_config.format_prompt_classification(input_text, tokenizer=tokenizer)
        
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
                max_new_tokens=64, 
                num_beams=1, 
                pad_token_id=tokenizer.pad_token_id,
            )
            
        generated_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()
        
        # Robust Regex Parsing for 11 Projects (Add this improved logic)
        gen_lower = generated_text.lower()
        pred_label = "UNKNOWN"

        # Sort ALL alias keys by length descending to ensure the longest, most specific match wins
        for alias in sorted(PROJECT_ALIASES.keys(), key=len, reverse=True):
            if alias in gen_lower:
                pred_label = PROJECT_ALIASES[alias]
                break
                
        true_label = str(row['project'])
        
        y_pred.append(pred_label)
        y_true.append(true_label)

        results_list.append({
            "id": row.get('id', index), 
            "true_project": true_label,
            "predicted_project": pred_label,
            "raw_output": generated_text,
            "prompt": prompt
        })

        # Clear GPU memory after each iteration to prevent OOM
        del inputs, outputs
        torch.cuda.empty_cache()
        
    # Calculate Macro-Averaged Metrics
    macro_f1 = round(f1_score(y_true, y_pred, average='macro', labels=VALID_PROJECTS), 5)
    weighted_f1 = round(f1_score(y_true, y_pred, average='weighted', labels=VALID_PROJECTS), 5)
    macro_prec = round(precision_score(y_true, y_pred, average='macro', zero_division=0, labels=VALID_PROJECTS), 5)
    macro_rec = round(recall_score(y_true, y_pred, average='macro', zero_division=0, labels=VALID_PROJECTS), 5)
    acc = round(accuracy_score(y_true, y_pred), 5)
    
    print("\n" + "="*50)
    print(f"RESULTS FOR: {base_model_name} | EXP: {exp_type}")
    print("="*50)
    print(f"Macro-F1:    {macro_f1:.5f}")
    print(f"Weighted-F1: {weighted_f1:.5f}")
    print(f"Precision:   {macro_prec:.5f}")
    print(f"Recall:      {macro_rec:.5f}")
    print(f"Accuracy:    {acc:.5f}")
    print("="*50)
    
    results = {
        "Macro_F1": macro_f1,
        "Weighted_F1": weighted_f1,
        "Macro_Precision": macro_prec, 
        "Macro_Recall": macro_rec, 
        "Accuracy": acc
    }
    
    with open(os.path.join(checkpoint_dir, "test_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    predictions_path = os.path.join(checkpoint_dir, "test_predictions.jsonl")
    with open(predictions_path, "w") as f:
        for result in results_list:
            f.write(json.dumps(result) + '\n')
        
    if wandb.run is not None:
        wandb.log({
            "test/macro_f1": macro_f1, 
            "test/weighted_f1": weighted_f1,
            "test/macro_precision": macro_prec, 
            "test/macro_recall": macro_rec, 
            "test/accuracy": acc
        })
    
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
    parser.add_argument('--train_file', type=str, default='./data/classification/train_df.pkl')
    parser.add_argument('--valid_file', type=str, default='./data/classification/dev_df.pkl')
    parser.add_argument('--test_file', type=str, default='./data/classification/test_df.pkl')
    parser.add_argument('--exp', type=str, default='baseline')
    
    args = parser.parse_args()
    output_dir = f"model_class/{args.model}-{args.exp}-{args.bits}bit"
    
    if args.mode in ['train', 'both']:
        train_class_model(args.model, args.bits, args.train_file, args.valid_file, output_dir, args.exp, args.batch_size)
        
    if args.mode in ['test', 'both']:
        test_class_model(args.model, output_dir, args.test_file, args.exp, args.bits)
        
    if args.mode == 'both':
        wandb.finish()

if __name__ == '__main__':
    main()