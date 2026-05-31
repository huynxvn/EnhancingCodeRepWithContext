import os
import json
import torch
import wandb
from torch.utils.data import Dataset, DataLoader # Ablation - Fixed: Added DataLoader for batching
from transformers import (
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    EarlyStoppingCallback # Ablation - Fixed: Ensure this is imported
)
from peft import get_peft_model, PeftModel, prepare_model_for_kbit_training
from transformers.trainer_utils import set_seed
import argparse
from tqdm import tqdm
import glob
import shutil
import math
import csv
import itertools

import re
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    
# Import model configurations
from models import get_model_config, list_available_models

# Set seed for reproducibility
set_seed(42)

# A100 Optimizations
torch.backends.cuda.enable_cudnn_sdp(False) 
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# ------------------------------------------------------------------------------
# Helper Function for Ablation Input Construction
# ------------------------------------------------------------------------------
def build_ablation_input(js, exp_type):
    """
    Builds input string respecting the EXACT ORDER of context in exp_type.
    Example: 'code_cg_vh' -> Source Code + Call Graph + Version History
    """
    def to_str(tokens):
        if isinstance(tokens, list):
            return ' '.join(tokens).replace('\n', ' ')
        return str(tokens).replace('\n', ' ')

    # 1. Base Source Code (Always first)
    source_code = to_str(js.get('code_tokens', []))
    parts = [f"### Source Code\n{source_code}"]
    
    # 2. Parse the experiment string to determine order
    # Remove 'code_' prefix and split by underscore
    # e.g., "code_vh_cg_nod" -> ["vh", "cg", "nod"]
    # e.g., "code_cg_vh_nod" -> ["cg", "vh", "nod"]
    context_order = exp_type.replace("code_", "").split("_")
    
    # 3. Append parts in the specific order requested
    for context in context_order:
        
        if context == 'vh': # Version History
            history_items = js.get('version_history', [])
            if len(history_items) > 1:
                # Skip index 0 (current code), use 1..n (past)
                versions = [to_str(item['commit_source_code_tokens']) for item in history_items[1:]]
                # Format: Ver 1: [code] \n Ver 2: [code]
                history_str = "\n".join([f"Ver {i+1}: {code}" for i, code in enumerate(versions)])
                if history_str:
                    parts.append(f"### Version History\n{history_str}")
                
        elif context == 'cg': # Call Graph
            caller_str = to_str(js.get('caller_context_tokens', []))
            callee_str = to_str(js.get('callee_context_tokens', []))
            if caller_str:
                parts.append(f"### Caller Context\n{caller_str}")
            if callee_str:
                parts.append(f"### Callee Context\n{callee_str}")
                
        elif context == 'nod': # Method Age (Number of Days)
            days_str = to_str(js.get('num_of_days_tokens', []))
            if days_str:
                parts.append(f"### Method Age\n{days_str} days")

    # Join with double newlines
    return "\n\n".join(parts)

def clean_prediction(text):
    if not text: return "Summary not generated"
    
    # 1. Line stop: Take only the first line to prevent multi-line rambling
    text = text.split('\n')[0].strip()
    
    # NEW: Strip double and single quotes immediately to fix the "Validates..." issue
    text = text.strip(' "\'')
        
    # 2. Prompt bleeding filter
    text = re.sub(r"in the '### Source Code' section\s*", "", text, flags=re.IGNORECASE)
    
    # 3. Bulletproof Chatty Prefix filter 
    # Qwen style: "The `getPixel` method..."
    qwen_pattern = r"^[^a-zA-Z]*the\s+([\'\"\`‘’“”]?[\w\.\-\$<>]+[\(\)]*[\'\"\`‘’“”]?\s+)?method(?:\s+in\s+the\s+(?:provided\s+java\s+code|[\'\"\`‘’“”]?\w+[\'\"\`‘’“”]?\s+class))?\s+"
    text = re.sub(qwen_pattern, "", text, flags=re.IGNORECASE)
    
    # NEW: DeepSeek style: "The method synchronized Bitmap getNextFrame() " or "The method 'copyOutAttributes' "
    deepseek_pattern = r"^[^a-zA-Z]*the\s+method\s+(?:(?:public|private|protected|static|final|synchronized|abstract|volatile|transient)\s+)*(?:[\w\<\>\[\]]+\s+)?(?:[\'\"\`‘’“”][\w\.\-\$]+[\'\"\`‘’“”]|[\w\.\-\$]+\(\))\s+"
    text = re.sub(deepseek_pattern, "", text, flags=re.IGNORECASE)
    
    # Simple prefix fallback 
    simple_prefixes = [
        r"^[^a-zA-Z]*This Java code ", 
        r"^[^a-zA-Z]*This code ", 
        r"^[^a-zA-Z]*This method ", 
        r"^[^a-zA-Z]*The function ",
        r"^[^a-zA-Z]*Here is ", 
        r"^[^a-zA-Z]*In this code ", 
        r"^[^a-zA-Z]*The provided code ",
        r"^[^a-zA-Z]*The provided ",
        r"^[^a-zA-Z]*The method "
    ]
    for p in simple_prefixes:
        text = re.sub(p, "", text, flags=re.IGNORECASE).strip()
        
    # NEW: Strip quotes AGAIN just in case the prefix removal exposed them
    text = text.strip(' "\'')
            
    # 4. Capitalize the newly exposed verb
    if text:
        text = text[0].upper() + text[1:]
        
    # 5. Extract first sentence
    try:
        sentences = nltk.tokenize.sent_tokenize(text)
        return sentences[0].strip() if sentences else text.strip()
    except:
        return text.strip()
    
class CodeSummarizationDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, model_config, max_source_length=4096, max_target_length=128, exp_type="baseline"):
        self.tokenizer = tokenizer
        self.model_config = model_config
        # Ablation - Fixed: Increased to 4096 to handle history + call graph context
        self.max_source_length = max_source_length 
        self.max_target_length = max_target_length
        self.is_causal_lm = model_config.is_causal_lm
        self.exp_type = exp_type # Ablation - Fixed: Store experiment type
        self.examples = []
        
        print(f"==== Loading Dataset: {jsonl_file} (Experiment: {exp_type}) ====")
        # print total lines in file for progress tracking

        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
            print(f"Total lines in dataset: {total_lines}")
            f.seek(0) # Reset file pointer to beginning

            # # Use islice to only grab the first N records
            # max_examples = 1000
            # subset = itertools.islice(f, max_examples)
            
            # for i, line in enumerate(subset):
            for line in f:
                data = json.loads(line.strip())
                
                # Ablation - Fixed: Build the input text using the helper function
                input_text = build_ablation_input(data, self.exp_type)
                
                docstring = ' '.join(data['docstring_tokens']) if isinstance(data['docstring_tokens'], list) else data['docstring_tokens']
                self.examples.append({'input_text': input_text, 'docstring': docstring})
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Ablation - Fixed: Retrieve the constructed input_text
        input_text = example['input_text']
        docstring = example['docstring']
        
        if self.is_causal_lm:
            # For causal LM: concatenate prompt + code + summary
            # UPDATE: Pass tokenizer here
            # prompt = self.model_config.format_prompt(input_text)
            prompt = self.model_config.format_prompt(input_text, tokenizer=self.tokenizer)
            full_text = prompt + docstring
            
            # OPTIMIZATION: REMOVED padding='max_length'
            # We allow variable lengths here. The DataCollator will pad the batch later.
            encoding = self.tokenizer(
                full_text,
                max_length=self.max_source_length + self.max_target_length,
                padding=False, # CHANGED: Turn off static padding
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            labels = input_ids.clone()
            
            prompt_length = len(self.tokenizer(prompt, add_special_tokens=False)['input_ids'])
            
            if prompt_length >= self.max_source_length + self.max_target_length:
                 prompt_length = self.max_source_length
            
            labels[:prompt_length] = -100
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            }
        else:
            # For encoder-decoder: separate source and target
            source = self.tokenizer(
                input_text, 
                max_length=self.max_source_length,
                padding=False, # CHANGED: Turn off static padding
                truncation=True,
                return_tensors='pt'
            )
            
            target = self.tokenizer(
                docstring,
                max_length=self.max_target_length,
                padding=False, # CHANGED: Turn off static padding
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': source['input_ids'].squeeze(),
                'attention_mask': source['attention_mask'].squeeze(),
                'labels': target['input_ids'].squeeze(),
            }


class EpochTestCallback(TrainerCallback):
    """Callback to test model after each epoch"""
    
    def __init__(self, test_file, output_base_dir, model_config, tokenizer, bits, beam_size=5, exp_type="baseline", batch_size=2):
        self.test_file = test_file
        self.output_base_dir = output_base_dir
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.bits = bits
        self.beam_size = beam_size
        self.exp_type = exp_type # Ablation - Fixed: Store exp_type
        self.batch_size = batch_size # Ablation - Fixed: Store batch_size

    def on_epoch_end(self, args, state, control, model, **kwargs):
        """Run testing at the end of each epoch"""
        epoch = int(state.epoch)
        print(f"\n{'='*60}")
        print(f"Testing after Epoch {epoch} (Exp: {self.exp_type})")
        print(f"{'='*60}\n")
        
        # Create output directory for this epoch
        epoch_output_dir = os.path.join(self.output_base_dir, f'epoch_{epoch}')
        os.makedirs(epoch_output_dir, exist_ok=True)
        
        # Run testing
        test_model_internal(
            model=model,
            tokenizer=self.tokenizer,
            model_config=self.model_config,
            test_file=self.test_file,
            output_dir=epoch_output_dir,
            bits=self.bits,
            beam_size=self.beam_size,
            exp_type=self.exp_type, # Ablation - Fixed: Pass exp_type to test function
            batch_size=self.batch_size # Ablation - Fixed: Pass batch_size to test function
        )
        
        print(f"\n✓ Epoch {epoch} testing completed!")
        print(f"Results saved to {epoch_output_dir}\n")

class PerplexityLoggerCallback(TrainerCallback):
    """FIXED: A callback that computes PPL, saves to CSV, and logs to WandB."""
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.csv_path = os.path.join(output_dir, "metrics_log.csv")
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "eval_loss", "perplexity"])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            eval_loss = metrics["eval_loss"]
            epoch = state.epoch
            try:
                perplexity = math.exp(eval_loss)
            except (OverflowError, ValueError):
                perplexity = float('inf')
            
            # 1. Print to terminal
            print(f"\n>>> Epoch {epoch:.2f}: Eval Loss = {eval_loss:.4f}, PPL = {perplexity:.4f}")
            
            # 2. Append to CSV file
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, eval_loss, perplexity])
            
            # 3. Add to metrics (so it shows up in Trainer's internal logs/state)
            metrics["eval_perplexity"] = perplexity
            
            # 4. FIXED: Sync with WandB if active
            if wandb.run is not None:
                wandb.log({"eval/perplexity": perplexity, "eval/loss": eval_loss}, step=state.global_step)

def get_quantization_config(bits):
    """Get quantization config based on bit precision"""
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif bits == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
    else:
        return None  # 16-bit or full precision

def test_model_internal(model, tokenizer, model_config, test_file, output_dir, bits, beam_size=5, exp_type="baseline", batch_size=2):
    """
    Optimized testing function for Software Engineering Research.
    Ensures 1:1 line mapping and removes LLM chat hallucinations.
    """
    # 1. FIX: Clear Memory before starting inference
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("Cleared GPU memory before testing phase.")

    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    # Determine device (same as before)
    if bits in [4, 8]:
        device = model.device
    else:
        device = next(model.parameters()).device
    
    # Load raw data
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line.strip()))
    
    # Reuse InferenceDataset logic
    class InferenceDataset(Dataset):
        def __init__(self, data, exp_type):
            self.data = data
            self.exp_type = exp_type
        def __len__(self): return len(self.data)
        def __getitem__(self, idx):
            item = self.data[idx]
            input_text = build_ablation_input(item, self.exp_type)
            docstring = ' '.join(item['docstring_tokens']) if isinstance(item['docstring_tokens'], list) else item['docstring_tokens']
            return input_text, docstring

    inf_dataset = InferenceDataset(test_data, exp_type)
    
    # 2. FIX: You successfully set batch_size=8 here. This is good for 4096 tokens.
    # Changed from 32 (Risky) or 8 (Slow) to 16 (Optimal)
    # dataloader = DataLoader(inf_dataset, batch_size=16, shuffle=False) # QwenCoder 2.5 1.5B
    dataloader = DataLoader(inf_dataset, batch_size=batch_size, shuffle=False) # DeepSeek Coder 6.7B Instruct 
    
    predictions = []
    references = []
    
    terminators = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None:
        terminators.append(im_end_id)

    with torch.no_grad():
        for inputs_text, docstrings in tqdm(dataloader, desc="Batch Generating", leave=False):
            if model_config.is_causal_lm:
                # 1. UPDATE: Pass tokenizer to format_prompt for chat template application
                prompts = [model_config.format_prompt(t, tokenizer=tokenizer) for t in inputs_text]
                
                inputs = tokenizer(
                    prompts,
                    max_length=model_config.max_source_length,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                ).to(device)
                
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=model_config.max_target_length,
                    num_beams=beam_size,
                    early_stopping=True,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=terminators,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )
                
                # 3. Process each item in the batch
                prompt_length = inputs['input_ids'].shape[-1]
                for i in range(len(outputs)):
                    # Decode full text first
                    full_text = tokenizer.decode(outputs[i][prompt_length:], skip_special_tokens=True).strip()

                    # 4. Apply our bulletproof regex cleaning
                    final_summary = clean_prediction(full_text)
                    
                    # 5. Clean remaining LLM artifacts (just in case)
                    for artifact in ["Human:", "User:", "Assistant:", "<|im_end|>", "<|im_start|>"]:
                        if artifact in final_summary:
                            final_summary = final_summary.split(artifact)[0].strip()
                    
                    if not final_summary or final_summary.isspace():
                        final_summary = "none"
                        
                    predictions.append(final_summary)
                    references.append(docstrings[i])
            else:
                # Encoder-Decoder Logic (Unchanged)
                inputs = tokenizer(list(inputs_text), max_length=model_config.max_source_length, padding=True, truncation=True, return_tensors='pt').to(device)
                outputs = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=model_config.max_target_length, num_beams=beam_size, early_stopping=True)
                for i in range(len(outputs)):
                    # pred = tokenizer.decode(outputs[i], skip_special_tokens=True).strip().replace('\n', ' ')
                    # predictions.append(pred if pred else "none")
                    # references.append(docstrings[i])

                    # Ensure the consistency of the output by applying the same cleaning as causal LM
                    raw_pred = tokenizer.decode(outputs[i], skip_special_tokens=True).strip()
                    pred = clean_prediction(raw_pred)
                    
                    predictions.append(pred if pred != "none" else "none")
                    references.append(docstrings[i])
    
    # Write outputs (same as before)
    output_file = os.path.join(output_dir, 'test.out')
    gold_file = os.path.join(output_dir, 'test.gold')
    with open(output_file, 'w', encoding='utf-8') as f:
        for pred in predictions: f.write(pred + '\n')
    with open(gold_file, 'w', encoding='utf-8') as f:
        for ref in references: f.write(ref + '\n')

def train_model(model_name, bits, data_file, train_file, valid_file, test_file, 
                output_dir, use_lora=False, test_every_epoch=False, exp_type="baseline", batch_size=2):
    """Train model with specified quantization and optional LoRA"""
    
    # Get model configuration
    model_config = get_model_config(model_name, use_lora=use_lora)
    
    # Initialize wandb
    wandb.init(
        project="code-summarization-ablation", # Ablation - Fixed: Updated project name
        name=f"{model_name}-{exp_type}-{bits}bit{'-lora' if use_lora else ''}-{data_file}", # Ablation - Fixed: Added exp_type to run name
        config={
            "model": model_name,
            "exp_type": exp_type, # Ablation - Fixed: Log exp_type
            "bits": bits,
            "use_lora": use_lora,
            "data_file": data_file,
            "train_batch_size": batch_size, 
            "eval_batch_size": batch_size * 2,
            "num_epochs": 10,
            "max_source_length": 4096, # Ablation - Fixed: Explicitly logging new length
            "max_target_length": model_config.max_target_length,
            "test_every_epoch": test_every_epoch
        }
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    print(f"Loading {model_name} tokenizer...")
    tokenizer = model_config.get_tokenizer()

    ## FIXED: adding missing pad_token for Qwen
    tokenizer.padding_side = "right" # Add this for training
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading {model_name} model for {bits}-bit training{' with LoRA' if use_lora else ''}...")
    
    # Determine training strategy
    if use_lora and bits in [4, 8]:
        # LoRA + Quantization (QLoRA)
        quantization_config = get_quantization_config(bits)
        model = model_config.get_model(quantization_config)
        model = prepare_model_for_kbit_training(model)
        lora_config = model_config.get_lora_config()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        training_mode = "qlora"
    elif use_lora:
        # LoRA without quantization
        model = model_config.get_model()
        lora_config = model_config.get_lora_config()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        training_mode = "lora"
    else:
        # Full model training (no LoRA, no quantization during training)
        model = model_config.get_model()
        training_mode = "full"
    
    # Configure model for generation
    model_config.configure_model_for_generation(model, tokenizer)
    
    # Load datasets
    print(f"Loading datasets for experiment: {exp_type}...")
    
    # Ablation - Fixed: Passing 4096 as max_source_length and exp_type to dataset
    train_dataset = CodeSummarizationDataset(train_file, tokenizer, model_config,
                                             4096, 
                                             model_config.max_target_length,
                                             exp_type=exp_type)
    eval_dataset = CodeSummarizationDataset(valid_file, tokenizer, model_config,
                                           4096,
                                           model_config.max_target_length,
                                           exp_type=exp_type)
    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=output_dir,
        
        # --- A100 OPTIMIZATION: BATCH SIZE & CONTEXT ---
        # # QwenCode2.5-1.5B
        # per_device_train_batch_size=4,  # Ablation - Fixed: Reduced from 16 to 8 for 4096 context length
        # per_device_eval_batch_size=8,  # Ablation - Fixed: Reduced from 32 to 16
        # gradient_accumulation_steps=16,  # Ablation - Fixed: Increased from 4 to 8 to keep effective batch size ~64

        # DeepSeek Coder 6.7B Instruct 
        per_device_train_batch_size= batch_size,  # Lowered for 6.7B model
        per_device_eval_batch_size= batch_size * 2,   # Lowered for 6.7B model
        gradient_accumulation_steps= max(1, 64 // batch_size), # Maintains effective batch size of 64

        # --- A100 OPTIMIZATION: PRECISION ---
        bf16=True,       
        fp16=False,      
        tf32=True,       
        
        # --- A100 OPTIMIZATION: SPEED ---
        gradient_checkpointing=True, # Turned from False to True for memory savings at the cost of some speed (DeepSeek 6.7B)    
        dataloader_num_workers=8,         
        
        # --- STABILITY FIXES (Prevents NaN) ---
        max_grad_norm=1.0,               
        learning_rate=5e-5,              
        warmup_steps=1000,               
        group_by_length=True,            

        # --- Standard & Early Stopping Args ---
        num_train_epochs=10,
        logging_steps=100,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,     # Required for EarlyStoppingCallback
        metric_for_best_model="eval_loss",
        greater_is_better=False,         # Lower loss is better
        report_to="wandb",
        weight_decay=0.01,
        logging_first_step=True,
    )


    # Setup callbacks
    callbacks = []
    if test_every_epoch and test_file:
        print("Setting up epoch-wise testing callback...")
        test_output_dir = output_dir.replace('model/', 'output/')
        callbacks.append(
            EpochTestCallback(
                test_file=test_file,
                output_base_dir=test_output_dir,
                model_config=model_config,
                tokenizer=tokenizer,
                bits=bits,
                beam_size=5,
                exp_type=exp_type, # Ablation - Fixed: Pass exp_type
                batch_size=batch_size # Ablation - Fixed: Pass batch_size
            )
        )
        
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        # Use [*callbacks] to include your EpochTestCallback + new ones
        callbacks=[
            *callbacks, 
            EarlyStoppingCallback(early_stopping_patience=3),
            PerplexityLoggerCallback(output_dir=output_dir)
        ]
    )
    
    # Train
    print(f"Starting training in {training_mode} mode (Exp: {exp_type})...")
    train_result = trainer.train()
    
    # 1. FIX: Report the best checkpoint path for your records
    best_ckpt = trainer.state.best_model_checkpoint
    print(f"\n>>> Best model found at: {best_ckpt}")
    print(f">>> Best metric: {trainer.state.best_metric}")

    # 2. SAVE BEST MODEL (Standard HF Method)
    print(f"Saving clean best model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 3. LoRA SPECIFIC: Save adapter separately if needed
    if use_lora:
        adapter_path = os.path.join(output_dir, 'adapter')
        model.save_pretrained(adapter_path)
        print(f"LoRA adapter exported to {adapter_path}")

    # 4. CLEAN UP: Delete bulky checkpoint folders to save space on cluster
    print("Cleaning up bulky checkpoint folders...")
    for ckpt_folder in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        try:
            shutil.rmtree(ckpt_folder)
            print(f"Deleted: {ckpt_folder}")
        except Exception as e:
            print(f"Failed to delete {ckpt_folder}: {e}")

    # 5. Save metadata
    metadata = {
        'model_name': model_name,
        'exp_type': exp_type, # Ablation - Fixed: Save exp_type
        'bits': bits,
        'use_lora': use_lora,
        'training_mode': training_mode,
        'data_file': data_file,
        'best_checkpoint': best_ckpt,
        'best_metric': trainer.state.best_metric,
        'base_model': model_config.base_model if hasattr(model_config, 'base_model') else None
    }
    with open(os.path.join(output_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save as .pt for legacy compatibility
    torch.save(metadata, os.path.join(output_dir, 'model.pt'))
    
    print(f"Success! Clean model and logs saved to: {output_dir}")
    wandb.finish()
    
    return model, tokenizer

def test_checkpoint(model_name, bits, checkpoint_dir, test_file, output_dir, beam_size=5, exp_type="baseline", batch_size=2):
    """Test a specific checkpoint"""
    
    # Ablation - Fixed: Explicitly log the experiment type being tested
    print(f"Loading checkpoint from {checkpoint_dir} (Exp: {exp_type})...")
    
    # Load metadata
    metadata_path = os.path.join(checkpoint_dir, 'model_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        use_lora = metadata.get('use_lora', False)
        saved_model_name = metadata.get('model_name', model_name)
    else:
        # Try to infer from checkpoint structure
        use_lora = os.path.exists(os.path.join(checkpoint_dir, 'adapter'))
        saved_model_name = model_name
    
    # Get model configuration
    model_config = get_model_config(saved_model_name, use_lora=use_lora)
    
    # Load tokenizer
    if os.path.exists(os.path.join(checkpoint_dir, 'tokenizer_config.json')):
        # FIXED: Use AutoTokenizer explicitly
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    else:
        # Fallback if no tokenizer file found in checkpoint
        tokenizer = model_config.get_tokenizer()
    
    # SAFETY: Ensure pad_token is set after loading (Crucial for Qwen) - FIXED
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    quantization_config = get_quantization_config(bits)
    
    if use_lora:
        print(f"Loading model with LoRA adapters...")
        # Load base model
        if bits in [4, 8]:
            base_model = model_config.get_model(quantization_config)
            adapter_path = os.path.join(checkpoint_dir, 'adapter')
            model = PeftModel.from_pretrained(base_model, adapter_path)
            # model = model.merge_and_unload() # comment out to leave the model as a PeftModel object. Avoid OOM crashes during testing.
        else:
            base_model = model_config.get_model()
            adapter_path = os.path.join(checkpoint_dir, 'adapter')
            model = PeftModel.from_pretrained(base_model, adapter_path)
            # model = model.merge_and_unload() # comment out to leave the model as a PeftModel object. Avoid OOM crashes during testing.
    else:
        print(f"Loading full model...")
        # Check if model was saved with save_pretrained
        if os.path.exists(os.path.join(checkpoint_dir, 'config.json')):
            if bits in [4, 8]:
                # Load with quantization for inference
                from transformers import AutoModelForSeq2SeqLM
                try:
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        checkpoint_dir,
                        quantization_config=quantization_config,
                        device_map="auto"
                    )
                except:
                    # Fallback for encoder-decoder models
                    model = model_config.get_model(quantization_config)
                    try:
                        state_dict = torch.load(os.path.join(checkpoint_dir, 'pytorch_model.bin'), 
                                              map_location='cpu')
                        model.load_state_dict(state_dict, strict=False)
                    except:
                        print("Warning: Using base model weights")
            else:
                # Load without quantization
                from transformers import AutoModelForSeq2SeqLM
                try:
                    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)
                except:
                    model = model_config.get_model()
                    state_dict = torch.load(os.path.join(checkpoint_dir, 'pytorch_model.bin'),
                                          map_location='cpu')
                    model.load_state_dict(state_dict)
        else:
            # Load from .pt file
            model = model_config.get_model(quantization_config if bits in [4, 8] else None)
            model_path = os.path.join(checkpoint_dir, 'model.pt')
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # Configure for generation
    model_config.configure_model_for_generation(model, tokenizer)
    
    # Move to device
    if bits not in [4, 8]:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        print(f"Model loaded on {device}")
    else:
        print(f"Model loaded with automatic device mapping")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run testing
    print(f"Testing on {test_file}...")
    # Ablation - Fixed: Pass exp_type to test_model_internal
    test_model_internal(model, tokenizer, model_config, test_file, output_dir, bits, beam_size, exp_type=exp_type, batch_size=batch_size)
    
    print(f"✓ Results saved to {output_dir}")


def test_all_checkpoints(model_name, bits, model_base_dir, test_file, output_base_dir, beam_size=5, exp_type="baseline", batch_size=2):
    """Test all epoch checkpoints"""
    
    # Find all checkpoint directories
    checkpoint_dirs = sorted(glob.glob(os.path.join(model_base_dir, 'checkpoint-*')))
    
    if not checkpoint_dirs:
        print(f"No checkpoints found in {model_base_dir}")
        return
    
    print(f"\nFound {len(checkpoint_dirs)} checkpoints")
    print(f"{'='*60}\n")
    
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_name = os.path.basename(checkpoint_dir)
        print(f"Testing {checkpoint_name}...")
        
        output_dir = os.path.join(output_base_dir, checkpoint_name)
        
        try:
            # Ablation - Fixed: Pass exp_type
            test_checkpoint(model_name, bits, checkpoint_dir, test_file, output_dir, beam_size, exp_type=exp_type, batch_size=batch_size)
            print(f"✓ {checkpoint_name} completed\n")
        except Exception as e:
            print(f"✗ Error testing {checkpoint_name}: {e}\n")
            continue


def main():
    parser = argparse.ArgumentParser(description='Train and test models with quantization and optional LoRA')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test', 'test_all', 'both'],
                        help='Mode: train, test (final model), test_all (all checkpoints), or both')
    parser.add_argument('--model', type=str, required=True,
                        help=f'Model name. Available: {list_available_models()}')
    parser.add_argument('--bits', type=int, required=True, choices=[4, 8, 16],
                        help='Quantization bits: 4, 8, or 16')
    parser.add_argument('--use_lora', action='store_true',
                        help='Use LoRA for parameter-efficient fine-tuning')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Data file name (e.g., demo-poison, codesearchnet)')
    parser.add_argument('--train_file', type=str, default=None,
                        help='Path to training data (default: data/{data_file}/train.jsonl)')
    parser.add_argument('--valid_file', type=str, default=None,
                        help='Path to validation data (default: data/{data_file}/valid.jsonl)')
    parser.add_argument('--test_file', type=str, default=None,
                        help='Path to test data (default: data/{data_file}/test.jsonl)')
    parser.add_argument('--test_every_epoch', action='store_true',
                        help='Run testing after every epoch during training')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Specific checkpoint directory to test (for test mode)')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size per device')
    
    # Ablation - Fixed: Added argument for ablation study experiments
    parser.add_argument('--exp', type=str, default='baseline', 
                        choices=['baseline', 'code_vh', 'code_cg', 'code_vh_nod', 
                                 'code_vh_cg', 'code_cg_vh', 'code_vh_cg_nod', 'code_cg_vh_nod'],
                        help='Ablation experiment type: baseline (code only), code_vh (code + history), etc.')
    
    args = parser.parse_args()
    
    # Validate model
    if args.model not in list_available_models():
        print(f"Error: Model '{args.model}' not supported.")
        print(f"Available models: {list_available_models()}")
        return
    
    # Set default file paths if not provided
    if args.train_file is None:
        args.train_file = f'data/{args.data_file}/train.jsonl'
    if args.valid_file is None:
        args.valid_file = f'data/{args.data_file}/valid.jsonl'
    if args.test_file is None:
        args.test_file = f'data/{args.data_file}/test.jsonl'
    
    lora_suffix = '-lora' if args.use_lora else ''
    # Ablation - Fixed: Include experiment type in output folder names
    model_dir = f"model/{args.model}-{args.exp}-{args.bits}bit{lora_suffix}-{args.data_file}"
    output_dir = f"output/{args.model}-{args.exp}-{args.bits}bit{lora_suffix}-{args.data_file}"
    
    print(model_dir)
    print(output_dir)
    print("="*60)
    
    print(f"\n{'='*60}")
    print(f"Model: {args.model} | {args.bits}-bit | {'LoRA' if args.use_lora else 'Full'}")
    print(f"Data: {args.data_file} | Mode: {args.mode}")
    print(f"Experiment: {args.exp}") # Ablation - Fixed: Print experiment type
    print(f"{'='*60}\n")
    
    if args.mode in ['train', 'both']:
        print("Starting training phase...")
        train_model(args.model, args.bits, args.data_file, 
                   args.train_file, args.valid_file, args.test_file,
                   model_dir, use_lora=args.use_lora, 
                   test_every_epoch=args.test_every_epoch,
                   exp_type=args.exp,
                   batch_size=args.batch_size) # Ablation - Fixed: Pass exp_type
        print("\n✓ Training completed!\n")
    
    if args.mode == 'test':
        print("Starting testing phase...")
        checkpoint_dir = args.checkpoint if args.checkpoint else model_dir
        test_checkpoint(args.model, args.bits, checkpoint_dir, args.test_file, 
                       output_dir, beam_size=5, exp_type=args.exp, batch_size=args.batch_size) # Ablation - Fixed: Pass exp_type
        print("\n✓ Testing completed!\n")
    
    if args.mode == 'test_all':
        print("Testing all checkpoints...")
        test_all_checkpoints(args.model, args.bits, model_dir, args.test_file, 
                           output_dir, beam_size=5, exp_type=args.exp, batch_size=args.batch_size) # Ablation - Fixed: Pass exp_type and batch_size
        print("\n✓ All checkpoints tested!\n")
    
    if args.mode == 'both' and not args.test_every_epoch:
        # Test final model if not already tested during training
        print("Testing final model...")
        test_checkpoint(args.model, args.bits, model_dir, args.test_file, 
                       output_dir, beam_size=5, exp_type=args.exp, batch_size=args.batch_size) # Ablation - Fixed: Pass exp_type
        print("\n✓ Testing completed!\n")
    
    print(f"{'='*60}")
    print("All operations completed successfully!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()