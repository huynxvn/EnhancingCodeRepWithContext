import os
import argparse
import numpy as np
from tqdm import tqdm

# Metrics Libraries
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score_calc

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

def calculate_metrics(gold_file, pred_file):
    print(f"Loading files...\nGold: {gold_file}\nPred: {pred_file}")
    
    # 1. Load the files
    with open(gold_file, 'r', encoding='utf-8') as f:
        references = [line.strip() for line in f] # Keep empty lines as empty strings for alignment
    with open(pred_file, 'r', encoding='utf-8') as f:
        predictions = [line.strip() for line in f]

    if len(references) != len(predictions):
        print(f"Warning: File length mismatch! Gold: {len(references)}, Pred: {len(predictions)}")
        min_len = min(len(references), len(predictions))
        references = references[:min_len]
        predictions = predictions[:min_len]

    print(f"Total samples to evaluate: {len(references)}")

    # 2. BLEU-4 Calculation (Corpus-level)
    # KEEPING AS PERCENTAGE (0-100) per your requirement
    ref_bleu = [[ref.split()] for ref in references]
    pred_bleu = [pred.split() for pred in predictions]
    
    smoothie = SmoothingFunction().method4
    bleu4 = corpus_bleu(ref_bleu, pred_bleu, smoothing_function=smoothie) * 100

    # 3. ROUGE-L & METEOR Calculation (Sentence-level average)
    r_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_list = []
    meteor_list = []

    for ref, pred in tqdm(zip(references, predictions), total=len(references), desc="Calculating Metrics"):
        # ROUGE-L
        rs = r_scorer.score(ref, pred)
        rouge_l_list.append(rs['rougeL'].fmeasure)

        # METEOR
        # tokenize for meteor (NLTK expects lists of tokens)
        m_score = meteor_score([ref.split()], pred.split())
        meteor_list.append(m_score)

    # UPDATED: No multiplication by 100 (Decimal 0-1)
    avg_rouge_l = np.mean(rouge_l_list)
    avg_meteor = np.mean(meteor_list)

    # 4. BERTScore Calculation (F1)
    print("Calculating BERTScore (this may take a moment)...")
    try:
        P, R, F1 = bert_score_calc(predictions, references, lang="en", model_type="roberta-base", rescale_with_baseline=True)
        # UPDATED: No multiplication by 100 (Decimal 0-1)
        avg_bert_f1 = F1.mean().item()
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        avg_bert_f1 = 0.0

    # Output Results
    print("\n" + "="*30)
    print("FINAL EVALUATION RESULTS")
    print("="*30)
    # Requested Order: BLEU 4, BertScore F1, Rouge-L, Meteor
    print(f"BLEU-4:       {bleu4:.4f}  (Percentage 0-100)")
    print(f"BERTScore F1: {avg_bert_f1:.4f}  (Decimal 0-1)")
    print(f"ROUGE-L:      {avg_rouge_l:.4f}  (Decimal 0-1)")
    print(f"METEOR:       {avg_meteor:.4f}  (Decimal 0-1)")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=str, required=True, help="Path to test.gold")
    parser.add_argument("--pred", type=str, required=True, help="Path to test.out")
    args = parser.parse_args()

    calculate_metrics(args.gold, args.pred)