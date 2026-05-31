import os
import argparse
import numpy as np
from tqdm import tqdm
import sys

# Metrics Libraries
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score_calc

# --- 1. ROBUST NLTK SETUP ---
print("Checking NLTK resources...")
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("! 'punkt' not found. Downloading...")
    nltk.download('punkt')

# Fix for newer NLTK versions that split punkt into punkt_tab
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab')
    except:
        pass # Ignore if this specific download fails (older NLTK)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("! 'wordnet' not found. Downloading...")
    nltk.download('wordnet')
    
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    print("! 'omw-1.4' not found. Downloading...")
    nltk.download('omw-1.4')


# def clean_prediction(text, debug=False):
#     """
#     Extracts the first 'real' summary sentence, skipping chatty intros.
#     """
#     if not text: return ""
    
#     # 1. Split into sentences using NLTK
#     try:
#         sentences = nltk.tokenize.sent_tokenize(text)
#     except Exception as e:
#         if debug:
#             print(f"[ERROR] NLTK Tokenization failed: {e}")
#         return text # Fallback to original text
        
#     if not sentences: return text
    
#     # 2. Define "Junk" Prefixes common in Qwen/LLMs
#     junk_prefixes = [
#         "The provided Java code",
#         "The provided code",
#         "The provided method",
#         "The provided class",
#         "This Java code",
#         "This code snippet",
#         "The method defines",
#         "The function defines",
#         "This method takes",
#         "This method accepts",
#         "The Java code",
#         "Here is a summary",
#         "This is a method",
#         "In this code",
#         "The code snippet",
#         "This function is",
#         "The code defines a method",
#         "The code defines",
#         "The code is"
#     ]
    
#     selected_sentence = ""
    
#     # 3. Iterate to find the first sentence that DOESN'T start with a junk prefix
#     for sent in sentences:
#         is_junk = False
#         for prefix in junk_prefixes:
#             if sent.lower().startswith(prefix.lower()):
#                 is_junk = True
#                 break
        
#         if not is_junk:
#             # We found a real sentence!
#             selected_sentence = sent
#             break
    
#     # Fallback: If all sentences looked like junk, take the second sentence 
#     # (often the first is "The provided code..." and the second is the real summary)
#     if not selected_sentence and len(sentences) > 1:
#         selected_sentence = sentences[1]
#     elif not selected_sentence and sentences:
#         selected_sentence = sentences[0]

#     return selected_sentence.strip()

def clean_prediction(text, debug=False):
    """
    Extracts the first 'real' summary sentence and slices off chatty intros.
    """
    if not text: return ""
    
    # 1. Split into sentences using NLTK
    try:
        sentences = nltk.tokenize.sent_tokenize(text)
    except Exception as e:
        if debug:
            print(f"[ERROR] NLTK Tokenization failed: {e}")
        return text # Fallback to original text
        
    if not sentences: return text
    
    # 2. Define "Junk" Prefixes
    junk_prefixes = [
        "The provided Java code",
        "The provided code",
        "The provided method",
        "The provided class",
        "This Java code",
        "This code snippet",
        "The method defines",
        "The function defines",
        "This method takes",
        "This method accepts",
        "The Java code",
        "Here is a summary",
        "This is a method",
        "In this code",
        "The code snippet",
        "This function is",
        "The code defines a method named",
        "The code defines a method",
        "The code defines",
        "The code defines a static method",
        "The code is",
        "A static method",
        "A method that",
        "A private method",
        "A method that",
        "A private static method named",
        "Defines a method that",
        "Defines a method named",
        "Defines a method",
        "A Java method that",
        "A protected static method named",
        "A private static method named"
    ]
    
    # CRITICAL FIX 1: Sort prefixes by length (longest first)
    # This prevents "The code defines" from matching before "The code defines a method", 
    # which would leave "a method" dangling in your text.
    junk_prefixes.sort(key=len, reverse=True)
    
    selected_sentence = ""
    
    # 3. Process sentences to physically remove the prefix
    for sent in sentences:
        cleaned_sent = sent.strip()
        
        for prefix in junk_prefixes:
            if cleaned_sent.lower().startswith(prefix.lower()):
                # CRITICAL FIX 2: Slice the prefix off the front of the string
                cleaned_sent = cleaned_sent[len(prefix):].strip()
                # Stop checking prefixes once we successfully removed one
                break 
                
        # If the sentence has actual content left after stripping, we keep it!
        if cleaned_sent:
            selected_sentence = cleaned_sent
            
            # Polish: Capitalize the newly exposed first letter
            # e.g., "to sort an array." -> "To sort an array."
            selected_sentence = selected_sentence[0].upper() + selected_sentence[1:]
            break
            
    # Fallback just in case everything was wiped out
    if not selected_sentence:
        selected_sentence = sentences[0]

    return selected_sentence.strip()

def calculate_metrics(gold_file, pred_file):
    print(f"\nLoading files...\nGold: {gold_file}\nPred: {pred_file}")
    
    if not os.path.exists(pred_file):
        print(f"ERROR: Prediction file not found at {pred_file}")
        return

    # 1. Load and Process Files
    with open(gold_file, 'r', encoding='utf-8') as f:
        references = [line.strip() for line in f]
    
    with open(pred_file, 'r', encoding='utf-8') as f:
        raw_predictions = [line.strip() for line in f]

    if len(references) != len(raw_predictions):
        print(f"Warning: File length mismatch! Gold: {len(references)}, Pred: {len(raw_predictions)}")
        min_len = min(len(references), len(raw_predictions))
        references = references[:min_len]
        raw_predictions = raw_predictions[:min_len]

    # --- FULL PROCESSING ---
    print(f"\nProcessing all {len(raw_predictions)} samples (extracting first sentence)...")
    clean_predictions = []
    for raw in tqdm(raw_predictions, desc="Cleaning"):
        clean = clean_prediction(raw)
        if not clean and raw: 
            clean = raw
        clean_predictions.append(clean)

    # --- SAVE PROCESSED OUTPUT ---
    base, ext = os.path.splitext(pred_file)
    postproc_file = f"{base}_postproc{ext}"
    abs_path = os.path.abspath(postproc_file)
    
    print(f"\nSaving processed predictions to:\n{abs_path}")
    try:
        with open(postproc_file, 'w', encoding='utf-8') as f:
            for p in clean_predictions:
                f.write(p + '\n')
        print("✓ File saved successfully.")
    except Exception as e:
        print(f"✗ Failed to save file: {e}")

    # 2. METRICS CALCULATION
    print("\nCalculating metrics...")
    
    # BLEU-4
    ref_bleu = [[ref.split()] for ref in references]
    pred_bleu = [pred.split() for pred in clean_predictions]
    smoothie = SmoothingFunction().method4
    bleu4 = corpus_bleu(ref_bleu, pred_bleu, smoothing_function=smoothie) * 100

    # ROUGE & METEOR
    r_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_list = []
    meteor_list = []

    for ref, pred in tqdm(zip(references, clean_predictions), total=len(references), desc="Scoring"):
        rs = r_scorer.score(ref, pred)
        rouge_l_list.append(rs['rougeL'].fmeasure)
        m_score = meteor_score([ref.split()], pred.split())
        meteor_list.append(m_score)

    avg_rouge_l = np.mean(rouge_l_list)
    avg_meteor = np.mean(meteor_list)

    # BERTScore
    print("Calculating BERTScore (this may take a moment)...")
    try:
        P, R, F1 = bert_score_calc(clean_predictions, references, lang="en", model_type="roberta-base", rescale_with_baseline=True)
        avg_bert_f1 = F1.mean().item()
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        avg_bert_f1 = 0.0

    # Output Results
    print("\n" + "="*30)
    print("FINAL EVALUATION RESULTS (Post-Processed)")
    print("="*30)
    # UPDATED ORDER: BLEU-4 -> BERTScore -> ROUGE-L -> METEOR
    print(f"BLEU-4:       {bleu4:.4f}  (Percentage 0-100)")
    print(f"BERTScore F1: {avg_bert_f1:.4f}  (Decimal 0-1)")
    print(f"ROUGE-L:      {avg_rouge_l:.4f}  (Decimal 0-1)")
    print(f"METEOR:       {avg_meteor:.4f}  (Decimal 0-1)")
    print("="*30)
    print(f"Processed file saved at: {postproc_file}")

    print(round(bleu4, 2))
    print(round(avg_bert_f1,4))
    print(round(avg_rouge_l,4))
    print(round(avg_meteor,4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=str, required=True, help="Path to test.gold")
    parser.add_argument("--pred", type=str, required=True, help="Path to test.out")
    args = parser.parse_args()

    calculate_metrics(args.gold, args.pred)