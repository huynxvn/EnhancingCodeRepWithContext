import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import nltk

from models import get_model_config

# Metrics
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score_calc

from transformers import set_seed
import random

from transformers import BitsAndBytesConfig

def enforce_reproducibility(seed=42):
    """Locks all random seeds and forces deterministic GPU operations."""
    # 1. Lock basic Python/NumPy RNG
    random.seed(seed)
    np.random.seed(seed)
    
    # 2. Lock PyTorch CPU/GPU RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 3. Lock Hugging Face Transformers RNG
    set_seed(seed)
    
    # 4. Force cuDNN to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call this immediately before loading the model
enforce_reproducibility(42)

#A100 SDPA Backend Optimizations <---
torch.backends.cuda.enable_cudnn_sdp(False) # Disables the buggy cuDNN graph compiler
torch.backends.cuda.enable_flash_sdp(True)  # Forces Flash Attention (if compatible)
torch.backends.cuda.enable_math_sdp(True)   # Safe fallback

# Ensure NLTK resources
for resource in ['tokenizers/punkt', 'corpora/wordnet', 'corpora/omw-1.4']:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1])

# -------------------------------------------------------------------
# 1. DATA & CLEANING UTILS
# -------------------------------------------------------------------
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
    
    # # ADD THIS INSTRUCTION
    # instruction = "Write a single-sentence summary for the following Java method. Start with a verb."
    # parts = [instruction]

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


def calculate_and_print_metrics(references, predictions, label):
    print(f"\nCalculating {label} Metrics...")
    
    # BLEU
    ref_bleu = [[ref.split()] for ref in references]
    pred_bleu = [pred.split() for pred in predictions]
    smoothie = SmoothingFunction().method4
    bleu4 = corpus_bleu(ref_bleu, pred_bleu, smoothing_function=smoothie) * 100

    # ROUGE & METEOR
    r_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_list, meteor_list = [], []
    
    for ref, pred in zip(references, predictions):
        rouge_l_list.append(r_scorer.score(ref, pred)['rougeL'].fmeasure)
        meteor_list.append(meteor_score([ref.split()], pred.split()))

    # BERTScore
    try:
        _, _, F1 = bert_score_calc(predictions, references, lang="en", model_type="roberta-base", rescale_with_baseline=True)
        avg_bert_f1 = F1.mean().item()
    except Exception as e:
        print(f"BERTScore Error: {e}")
        avg_bert_f1 = 0.0

    print(f"\n{'='*30}\n{label.upper()} RESULTS\n{'='*30}")
    print(f"BLEU-4:       {bleu4:.4f}  (0-100)")
    print(f"BERTScore F1: {avg_bert_f1:.4f}  (0-1)")
    print(f"ROUGE-L:      {np.mean(rouge_l_list):.4f}  (0-1)")
    print(f"METEOR:       {np.mean(meteor_list):.4f}  (0-1)\n{'='*30}")

# -------------------------------------------------------------------
# 2. MAIN INFERENCE LOOP
# -------------------------------------------------------------------
def map_hf_id_to_internal(model_id):
    """Maps a Hugging Face model path to our internal MODEL_REGISTRY key."""
    model_id = model_id.lower()
    
    # Qwen Models
    if "qwen2.5-coder-1.5b" in model_id: return "qwen25-coder-1.5b"
    if "qwen2.5-coder-7b" in model_id: return "qwen25-coder-7b"
    if "qwen2.5-coder-14b" in model_id: return "qwen25-coder-14b" 
    if "qwen2.5-coder-32b" in model_id: return "qwen25-coder-32b" 
    if "qwen3-coder-30b-a3b" in model_id: return "qwen3-coder-30b-a3b-instruct"
    
    # DeepSeek Models
    if "deepseek-coder-1.3b" in model_id: return "deepseek-coder-1.3b-instruct"
    if "deepseek-coder-6.7b" in model_id: return "deepseek-coder-6.7b-instruct"
    if "deepseek-coder-33b" in model_id: return "deepseek-coder-33b-instruct"
    
    # CodeLlama Models (CRITICAL FOR YOUR NEXT RUN)
    if "codellama-7b" in model_id: return "codellama-7b-instruct"
    if "codellama-13b" in model_id: return "codellama-13b-instruct"
    if "codellama-34b" in model_id: return "codellama-34b-instruct"
    
    raise ValueError(f"Model ID '{model_id}' not mapped in inference script!")

def run_zero_shot_inference(model_id, test_file, output_dir, exp_type, batch_size=4):
    os.makedirs(output_dir, exist_ok=True)
    
    # ---> Dynamic Model Configuration <---
    internal_name = map_hf_id_to_internal(model_id)
    model_config = get_model_config(internal_name)

    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("Cleared GPU memory. Loading BASE model (No LoRA)...")

    # Load Base Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Dynamic Precision: 4-bit for massive models, 16-bit for normal models
    if "33b" in model_id.lower() or "34b" in model_id.lower() or "30b" in model_id.lower():
        print(f"Massive model detected ({model_id}). Loading in 4-bit quantization to prevent OOM...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="auto", 
            quantization_config=quant_config,
            attn_implementation="sdpa" 
        )
    else:
        print(f"Loading {model_id} in 16-bit (bfloat16) precision...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="auto", 
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa" 
        )
    model.eval()
    
    # Load Data
    test_data = [json.loads(line.strip()) for line in open(test_file, 'r', encoding='utf-8')]
    # test_data = test_data[:100]  # For quick testing, remove or adjust as needed

    class InferenceDataset(Dataset):
        def __init__(self, data, exp_type):
            self.data = data
            self.exp_type = exp_type
        def __len__(self): return len(self.data)
        def __getitem__(self, idx):
            item = self.data[idx]
            input_text = build_ablation_input(item, self.exp_type)
            # docstring = ' '.join(item['docstring_tokens']) if isinstance(item['docstring_tokens'], list) else item['docstring_tokens']
            raw_doc = item.get('docstring_tokens', [])
            docstring = ' '.join(raw_doc) if isinstance(raw_doc, list) else str(raw_doc)
            return input_text, docstring

    dataloader = DataLoader(InferenceDataset(test_data, exp_type), batch_size=batch_size, shuffle=False)
    
    raw_preds, clean_preds, references = [], [], []
    terminators = [tokenizer.eos_token_id]
    if tokenizer.convert_tokens_to_ids("<|im_end|>") is not None:
        terminators.append(tokenizer.convert_tokens_to_ids("<|im_end|>"))

    print("Starting Generation...")
    with torch.no_grad():
        for inputs_text, docstrings in tqdm(dataloader, desc="Batch Generating"):
            # Apply custom format prompt (Ensures 100% identical baseline)
            prompts = [model_config.format_prompt(t, tokenizer=tokenizer) for t in inputs_text]
            
            inputs = tokenizer(prompts, max_length=4096, padding=True, truncation=True, return_tensors='pt').to(model.device)
            prompt_length = inputs['input_ids'].shape[-1]
            
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=64, # Reduced from 128: it only needs ~20 tokens for one sentence!
                num_beams=5,
                early_stopping=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=terminators,
                repetition_penalty=1.0 # disables it 
            )
            
            for i in range(len(outputs)):
                raw_text = tokenizer.decode(outputs[i][prompt_length:], skip_special_tokens=True).strip()
                
                # 1. MECHANICAL LINE STOP: Throw away anything after the first newline
                raw_text = raw_text.split('\n')[0].strip()

                cleaned = clean_prediction(raw_text)
                
                # Fallback if it output empty string
                if not raw_text: raw_text = "none"
                if not cleaned: cleaned = "none"

                raw_preds.append(raw_text)
                clean_preds.append(cleaned)
                references.append(docstrings[i])

    # Save outputs
    raw_file = os.path.join(output_dir, 'zeroshot_raw.out')
    clean_file = os.path.join(output_dir, 'zeroshot_clean.out')
    gold_file = os.path.join(output_dir, 'zeroshot.gold')
    
    for file_path, data in [(raw_file, raw_preds), (clean_file, clean_preds), (gold_file, references)]:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(data) + '\n')

    # Calculate and Print Metrics
    calculate_and_print_metrics(references, raw_preds, "RAW (Zero-Shot)")
    calculate_and_print_metrics(references, clean_preds, "POST-PROCESSED (Zero-Shot)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--exp_type", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    run_zero_shot_inference(args.model_id, args.test_file, args.output_dir, args.exp_type, args.batch_size)