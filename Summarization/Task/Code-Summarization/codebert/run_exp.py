# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import os
import sys
import bleu
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from model import Seq2Seq
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

from bert_score import score # BERTscore
from rouge_score import rouge_scorer # ROUGE-L
import nltk
from nltk.translate.meteor_score import meteor_score # METEOR
from nltk.tokenize import word_tokenize
# Download the punkt_tab tokenizer data
nltk.download('punkt_tab')

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target

def read_examples(filename, experiment):
    """Read examples from filename."""
    examples=[]    
    with open(filename,encoding="utf-8") as f:        
        for idx, line in enumerate(f):
            line=line.strip()
            js=json.loads(line)
            if 'idx' not in js:
                js['idx']=idx

            code = ''
            if experiment == "baseline":
                # code only                
                source_code = ' '.join(js['code_tokens']).replace('\n', ' ')
                combined_code = (
                    f"<code> {source_code} "
                )
                
            elif experiment == "code_vh":
                # code + version_history               
                source_code = ' '.join(js['code_tokens']).replace('\n', ' ')
                version_history = '<version> '.join([' '.join(item['commit_source_code_tokens']).replace('\n', ' ') for item in js['version_history'][1:]])                
                # Combine features using the special tokens
                combined_code = (
                    f"<code> {source_code} "
                    f"<history> {version_history} "
                )
                
            elif experiment == "code_cg":
                # code + callgraph
                source_code = ' '.join(js['code_tokens']).replace('\n', ' ')
                caller_code = ' '.join(js['caller_context_tokens']).replace('\n', ' ')
                callee_code = ' '.join(js['callee_context_tokens']).replace('\n', ' ')
                # Combine features using the special tokens
                combined_code = (
                    f"<code> {source_code} "
                    f"<caller> {caller_code} "
                    f"<callee> {callee_code} "
                )

            elif experiment == "code_vh_nod":                
                # code + version_history + num_of_days                
                source_code = ' '.join(js['code_tokens']).replace('\n', ' ')
                version_history = '<version> '.join([' '.join(item['commit_source_code_tokens']).replace('\n', ' ') for item in js['version_history'][1:]])
                num_of_days = ' '.join(js['num_of_days_tokens'])
                # Combine features using the special tokens
                combined_code = (
                    f"<code> {source_code} "
                    f"<history> {version_history} "
                    f"<days> {num_of_days} "
                )
                
            elif experiment == "code_vh_cg":
                # code + version_history + callgraph
                source_code = ' '.join(js['code_tokens']).replace('\n', ' ')
                version_history = '<version> '.join([' '.join(item['commit_source_code_tokens']).replace('\n', ' ') for item in js['version_history'][1:]])                
                caller_code = ' '.join(js['caller_context_tokens']).replace('\n', ' ')
                callee_code = ' '.join(js['callee_context_tokens']).replace('\n', ' ')
                # Combine features using the special tokens
                combined_code = (
                    f"<code> {source_code} "
                    f"<history> {version_history} "
                    f"<caller> {caller_code} "
                    f"<callee> {callee_code} "
                )
            
            elif experiment == "code_cg_vh":
                # code + version_history + callgraph
                source_code = ' '.join(js['code_tokens']).replace('\n', ' ')                
                caller_code = ' '.join(js['caller_context_tokens']).replace('\n', ' ')
                callee_code = ' '.join(js['callee_context_tokens']).replace('\n', ' ')
                version_history = '<version> '.join([' '.join(item['commit_source_code_tokens']).replace('\n', ' ') for item in js['version_history'][1:]])                
                # Combine features using the special tokens
                combined_code = (
                    f"<code> {source_code} "
                    f"<caller> {caller_code} "
                    f"<callee> {callee_code} "
                    f"<history> {version_history} "                    
                )

            elif experiment == "code_vh_cg_nod":
                # code + version_history + callgraph + num_of_days
                source_code = ' '.join(js['code_tokens']).replace('\n', ' ')
                version_history = '<version> '.join([' '.join(item['commit_source_code_tokens']).replace('\n', ' ') for item in js['version_history'][1:]])
                caller_code = ' '.join(js['caller_context_tokens']).replace('\n', ' ')
                callee_code = ' '.join(js['callee_context_tokens']).replace('\n', ' ')
                num_of_days = ' '.join(js['num_of_days_tokens'])
                # Combine features using the special tokens
                combined_code = (
                    f"<code> {source_code} "
                    f"<history> {version_history} "
                    f"<caller> {caller_code} "
                    f"<callee> {callee_code} "
                    f"<days> {num_of_days} "
                )
            
            elif experiment == "code_cg_vh_nod":
                # code + version_history + callgraph + num_of_days
                source_code = ' '.join(js['code_tokens']).replace('\n', ' ')                
                caller_code = ' '.join(js['caller_context_tokens']).replace('\n', ' ')
                callee_code = ' '.join(js['callee_context_tokens']).replace('\n', ' ')
                version_history = '<version> '.join([' '.join(item['commit_source_code_tokens']).replace('\n', ' ') for item in js['version_history'][1:]])
                num_of_days = ' '.join(js['num_of_days_tokens'])
                # Combine features using the special tokens
                combined_code = (
                    f"<code> {source_code} "                    
                    f"<caller> {caller_code} "
                    f"<callee> {callee_code} "
                    f"<history> {version_history} "
                    f"<days> {num_of_days} "
                )

            else:
                logger.error("!!!Unknown experiment params!!!")
                exit()
            
            code = ' '.join(combined_code.split())
            nl=' '.join(js['docstring_tokens']).replace('\n','')
            nl=' '.join(nl.strip().split())            
            examples.append(
                Example(
                        idx = idx,
                        source=code,
                        target = nl,
                        ) 
            )
            # if len(examples) > 1000: break
    return examples

class InputFeatures(object):
    """A single training/test features for an example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
                 exceeds_token_limit=False):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask
        self.exceeds_token_limit = exceeds_token_limit

def convert_examples_to_features(examples, tokenizer, args, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        exceeds_token_limit = False

        # Source tokens
        source_tokens = tokenizer.tokenize(example.source)
        if len(source_tokens) > args.max_source_length - 2:
            exceeds_token_limit = True
        source_tokens = source_tokens[:args.max_source_length - 2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * len(source_tokens)
        padding_length = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length

        # **Added Type Casting for Source and Target IDs and Masks**
        source_ids = torch.tensor(source_ids, dtype=torch.long).tolist()
        source_mask = torch.tensor(source_mask, dtype=torch.long).tolist()

        # Target tokens
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)
            if len(target_tokens) > args.max_target_length - 2:
                exceeds_token_limit = True
            target_tokens = target_tokens[:args.max_target_length - 2]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_tokens)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        # **Added Type Casting for Target IDs and Masks**
        target_ids = torch.tensor(target_ids, dtype=torch.long).tolist()
        target_mask = torch.tensor(target_mask, dtype=torch.long).tolist()

        # Log details for first 5 examples
        if example_index < 5 and stage == 'train':
            logger.info("*** Example ***")
            logger.info("idx: {}".format(example.idx))
            logger.info("source_tokens: {}".format([x.replace('\u0120', '_') for x in source_tokens]))
            logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
            logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))
            logger.info("target_tokens: {}".format([x.replace('\u0120', '_') for x in target_tokens]))
            logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
            logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))

        # Append features with flag for exceeding token limit
        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
                source_mask,
                target_mask,
                exceeds_token_limit=exceeds_token_limit
            )
        )
    return features

class TextDataset(Dataset):
    def __init__(self, examples, args):
        self.examples = examples
        self.args = args  
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item):        
        return (
            torch.tensor(self.examples[item].source_ids),
            torch.tensor(self.examples[item].target_ids),
            torch.tensor(self.examples[item].source_mask),
            torch.tensor(self.examples[item].target_mask),
            torch.tensor(self.examples[item].exceeds_token_limit) 
        )

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def eval_bleu(args, dev_dataset, model, device, tokenizer):
    #Calculate bleu  
    if 'dev_bleu' in dev_dataset:
        eval_examples,eval_data=dev_dataset['dev_bleu']
    else:
        eval_examples = read_examples(args.dev_filename, args.exp)
        eval_examples = random.sample(eval_examples,min(1000,len(eval_examples)))
        eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
        all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)    
        eval_data = TensorDataset(all_source_ids,all_source_mask)   
        dev_dataset['dev_bleu']=eval_examples,eval_data


    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval() 
    p=[]
    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)
        source_ids,source_mask= batch                  
        with torch.no_grad():
            preds = model(source_ids=source_ids,source_mask=source_mask)  
            for pred in preds:
                t=pred[0].cpu().numpy()
                t=list(t)
                if 0 in t:
                    t=t[:t.index(0)]
                text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                p.append(text)
    model.train()
    predictions=[]
    with open(os.path.join(args.output_dir,"dev.output"),'w') as f, open(os.path.join(args.output_dir,"dev.gold"),'w') as f1:
        for ref,gold in zip(p,eval_examples):
            predictions.append(str(gold.idx)+'\t'+ref)
            f.write(str(gold.idx)+'\t'+ref+'\n')
            f1.write(str(gold.idx)+'\t'+gold.target+'\n')     

    (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "dev.gold")) 
    dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
    return dev_bleu

def eval_metrics(args, dev_dataset, model, device, tokenizer):
    # Calculate BLEU as previously done
    if 'dev_bleu' in dev_dataset:
        eval_examples, eval_data = dev_dataset['dev_bleu']
    else:
        eval_examples = read_examples(args.dev_filename, args.exp)
        eval_examples = random.sample(eval_examples, min(1000, len(eval_examples)))
        eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
        all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
        exceeds_token_limit = torch.tensor([f.exceeds_token_limit for f in eval_features], dtype=torch.bool)  # Highlighted line: Added exceeds_token_limit
        eval_data = TensorDataset(all_source_ids, all_source_mask, exceeds_token_limit)  # Highlighted line: Added exceeds_token_limit to dataset
        dev_dataset['dev_bleu'] = eval_examples, eval_data

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    predictions = []
    references = []
    flagged_records = []  # Highlighted line: Added a list to track flagged records

    # Collect predictions and references
    p = []
    for batch in eval_dataloader:
        # batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  # Highlighted line: Handle dictionary batch
        batch = tuple(t.to(device) for t in batch)
        # source_ids, source_mask, exceeds_token_limit = batch["source_ids"], batch["source_mask"], batch["exceeds_token_limit"]
        # source_ids, source_mask, exceeds_token_limit = tuple(t.to(device) for t in batch)
        source_ids, source_mask, exceeds_token_limit = batch
        with torch.no_grad():
            preds = model(source_ids=source_ids, source_mask=source_mask)
            for idx, (pred, exceed_limit) in enumerate(zip(preds, exceeds_token_limit)):
                t = pred[0].cpu().numpy()
                t = list(t)
                if 0 in t:
                    t = t[:t.index(0)]
                text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                p.append(text)
                if exceed_limit:
                    flagged_records.append(idx)  # Highlighted line: Track indices of records exceeding the token limit

    model.train()

    # Write predictions and references to files
    with open(os.path.join(args.output_dir, "dev.output"), 'w') as f, open(os.path.join(args.output_dir, "dev.gold"), 'w') as f1:
        for ref, gold in zip(p, eval_examples):
            predictions.append(str(gold.idx) + '\t' + ref)
            references.append(gold.target)
            f.write(str(gold.idx) + '\t' + ref + '\n')
            f1.write(str(gold.idx) + '\t' + gold.target + '\n')

    # Log flagged records
    if flagged_records:
        logger.info("Records exceeding token limit: %s", flagged_records)  # Highlighted line: Log flagged records

    # Calculate BLEU
    (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "dev.gold"))
    dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)

    # Calculate BERTScore
    P, R, F1 = score(predictions, references, model_type="roberta-base", lang='en', rescale_with_baseline=True)
    avg_f1 = round(F1.mean().item(), 5)

    # Calculate ROUGE-L
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_scores = [rouge_scorer_instance.score(ref, pred)['rougeL'].fmeasure for ref, pred in zip(references, predictions)]
    avg_rouge_l = round(sum(rouge_l_scores) / len(rouge_l_scores), 5)

    # Calculate METEOR
    meteor_scores = []
    for ref, pred in zip(references, predictions):
        tokenized_ref = word_tokenize(ref)
        tokenized_pred = word_tokenize(pred)
        meteor_scores.append(meteor_score([tokenized_ref], tokenized_pred))
    avg_meteor = round(sum(meteor_scores) / len(meteor_scores), 5)

    # Print metrics
    logger.info("  %s = %s " % ("BLEU-4", str(dev_bleu)))
    logger.info("  %s = %s " % ("BERTScore F1", str(avg_f1)))
    logger.info("  %s = %s " % ("ROUGE-L", str(avg_rouge_l)))
    logger.info("  %s = %s " % ("METEOR", str(avg_meteor)))
    logger.info("  " + "*" * 20)

    return dev_bleu, avg_f1, avg_rouge_l, avg_meteor

def test(args, model, tokenizer, device, epoch=0, criteria=""):
    file = args.test_filename
    logger.info("Test file: {}".format(file))
    eval_examples = read_examples(file, args.exp)
    eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
    eval_data = TextDataset(eval_features, args)

    # Calculate BLEU
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    predictions = []
    references = []
    flagged_records = []  # Track records that exceed token limit

    p = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        source_ids, target_ids, source_mask, target_mask, exceeds_token_limit = batch
        with torch.no_grad():
            preds = model(source_ids=source_ids, source_mask=source_mask)
            for pred, flag in zip(preds, exceeds_token_limit):
                t = pred[0].cpu().numpy()
                t = list(t)
                if 0 in t:
                    t = t[:t.index(0)]
                text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                p.append(text)
                flagged_records.append(flag.item())  # Track flag value (0 or 1)
    model.train()

    # Write predictions and references to files
    with open(os.path.join(args.output_dir, f"test_{criteria}.output"), 'w') as f, \
         open(os.path.join(args.output_dir, f"test_{criteria}.gold"), 'w') as f1:
        for idx, (ref, gold, flag) in enumerate(zip(p, eval_examples, flagged_records)):
            predictions.append(str(gold.idx) + '\t' + ref + '\t' + str(flag))
            references.append(gold.target)
            f.write(str(gold.idx) + '\t' + ref + '\t' + str(flag) + '\n')
            f1.write(str(gold.idx) + '\t' + gold.target + '\n')

    # Calculate BLEU
    (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, f"test_{criteria}.gold"))
    dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)

    # Calculate BERTScore
    P, R, F1 = score(predictions, references, model_type="roberta-base", lang='en', rescale_with_baseline=True)
    avg_f1 = round(F1.mean().item(), 5)

    # Calculate ROUGE-L
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_scores = [rouge_scorer_instance.score(ref, pred)['rougeL'].fmeasure for ref, pred in zip(references, predictions)]
    avg_rouge_l = round(sum(rouge_l_scores) / len(rouge_l_scores), 5)

    # Calculate METEOR
    meteor_scores = []
    for ref, pred in zip(references, predictions):
        tokenized_ref = word_tokenize(ref)
        tokenized_pred = word_tokenize(pred)
        meteor_scores.append(meteor_score([tokenized_ref], tokenized_pred))
    avg_meteor = round(sum(meteor_scores) / len(meteor_scores), 5)

    # Print metrics
    logger.info("  %s = %s " % ("BLEU-4", str(dev_bleu)))
    logger.info("  %s = %s " % ("BERTScore F1", str(avg_f1)))
    logger.info("  %s = %s " % ("ROUGE-L", str(avg_rouge_l)))
    logger.info("  %s = %s " % ("METEOR", str(avg_meteor)))

    # Log flagged records
    logger.info("  %s = %s " % ("Number of records exceeding token limit in test set", flagged_records.count(1)))
    logger.info("  " + "*" * 20)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str, 
                        help="Path to trained model: Should contain the .bin files" )    
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str, 
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str, 
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str, 
                        help="The test filename. Should contain the .jsonl files for this task.")  
    
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name") 
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--stop_no_improve_epochs", default=5, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")   
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--exp', type=str, default="baseline",
                        help="combination of code with contexts for experiment")
    # print arguments
    args = parser.parse_args()
    logger.info(args)    

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device

    # Set seed
    set_seed(args.seed)

    # Make dir if output_dir does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)

    # Step 1: Load the tokenizer and add new tokens
    new_tokens = ["<code>", "<history>", "<version>", "<caller>", "<callee>", "<days>"]
    num_added_tokens = tokenizer.add_tokens(new_tokens)
    print(f"Added {num_added_tokens} extended tokens to the tokenizer.")

    # Step 2: Load the model and resize its embeddings
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    model.resize_token_embeddings(len(tokenizer))

    # Build model
    encoder = model  # Use resized model after adding new tokens
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                    beam_size=args.beam_size, max_length=args.max_target_length,
                    sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
    if args.load_model_path is not None:
        logger.info("Reloading model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))
        
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    epoch = -1
    if args.do_train:
        # Prepare training data loader        
        train_examples = read_examples(args.train_filename, args.exp)
        train_features = convert_examples_to_features(train_examples, tokenizer, args, stage='train')
        train_data = TextDataset(train_features, args)

        if args.local_rank == -1:            
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        # train_dataloader = DataLoader(train_data, 
        #                               sampler=train_sampler, 
        #                               batch_size=args.train_batch_size // args.gradient_accumulation_steps)
        train_dataloader = DataLoader(
                                train_data, 
                                sampler=train_sampler, 
                                batch_size=args.train_batch_size // args.gradient_accumulation_steps,
                                worker_init_fn=lambda worker_id: set_seed(args.seed + worker_id),  
                                generator=torch.Generator().manual_seed(args.seed)  # Set generator seed
                            )

        num_train_optimization_steps = args.train_steps

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total * 0.1),
                                                    num_training_steps=t_total)
    
        # Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epochs = %d", args.num_train_epochs)
        
        model.train()
        dev_dataset = {}
        nb_tr_examples, nb_tr_steps, tr_loss, global_step = 0, 0, 0, 0
        best_bleu, best_loss, best_bertscore, best_rouge_l, best_meteor = 0, 1e6, 0, 0, 0
        epochs_no_improve = 0  # Counter for early stopping        

        for epoch in range(args.num_train_epochs):
            bar = tqdm(train_dataloader, total=len(train_dataloader))
            train_num_exceed_token_limit = 0  # Initialize counter for the epoch

            for batch in bar:
                batch = tuple(t.to(device) for t in batch)
                source_ids, target_ids, source_mask, target_mask, exceeds_token_limit = batch

                # Count how many records in this batch exceed the token limit
                train_num_exceed_token_limit += exceeds_token_limit.sum().item()

                loss, _, _ = model(source_ids=source_ids, source_mask=source_mask, target_ids=target_ids, target_mask=target_mask)

                if args.n_gpu > 1:
                    loss = loss.mean()  # Mean to average on multi-GPU
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                bar.set_description("epoch {} loss {}".format(epoch, train_loss))
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

            # Log the count of records that exceed the token length limit after each epoch
            logger.info(f"Epoch {epoch}: Number of training records exceeding token limit: {train_num_exceed_token_limit}")

            if args.do_eval:
                # Eval model with dev dataset
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                eval_flag = False
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples = read_examples(args.dev_filename, args.exp)
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='dev')
                    eval_data = TextDataset(eval_features, args)
                    dev_dataset['dev_loss'] = eval_examples, eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                # Start evaluating model
                model.eval()
                eval_loss, tokens_num = 0, 0
                valid_num_exceed_token_limit = 0  # Initialize counter for validation

                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids, target_ids, source_mask, target_mask, exceeds_token_limit = batch

                    # Count how many records in this batch exceed the token limit
                    valid_num_exceed_token_limit += exceeds_token_limit.sum().item()

                    with torch.no_grad():
                        _, loss, num = model(source_ids=source_ids, source_mask=source_mask,
                                             target_ids=target_ids, target_mask=target_mask)
                    eval_loss += loss.sum().item()
                    tokens_num += num.sum().item()

                # Print loss of dev dataset    
                model.train()
                eval_loss = eval_loss / tokens_num
                result = {'eval_ppl': round(np.exp(eval_loss), 5),
                          'global_step': global_step + 1,
                          'train_loss': round(train_loss, 5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)

                # Log the count of records that exceed the token length limit after evaluation
                logger.info(f"Validation: Number of records exceeding token limit: {valid_num_exceed_token_limit}")

                # Save last checkpoint
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model
                output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)

                if eval_loss < best_loss:
                    logger.info("  Best ppl:%s", round(np.exp(eval_loss), 5))
                    best_loss = eval_loss
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                else:
                    epochs_no_improve += 1

                dev_bleu, dev_bertscore, dev_rouge_l, dev_meteor = eval_metrics(args, dev_dataset, model, device, tokenizer)

                # Save best checkpoints based on evaluation metrics
                if dev_bleu > best_bleu:
                    logger.info("  Best BLEU:%s", dev_bleu)
                    best_bleu = dev_bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model 
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)

                # if dev_bertscore > best_bertscore:
                #     logger.info("  Best BERTScore: %s", dev_bertscore)
                #     best_bertscore = dev_bertscore
                #     output_dir = os.path.join(args.output_dir, 'checkpoint-best-bertscore')
                #     if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                #     model_to_save = model.module if hasattr(model, 'module') else model
                #     output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                #     torch.save(model_to_save.state_dict(), output_model_file)
                
                # if dev_rouge_l > best_rouge_l:
                #     logger.info("  Best ROUGE-L: %s", dev_rouge_l)
                #     best_rouge_l = dev_rouge_l
                #     output_dir = os.path.join(args.output_dir, 'checkpoint-best-rougel')
                #     if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                #     model_to_save = model.module if hasattr(model, 'module') else model
                #     output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                #     torch.save(model_to_save.state_dict(), output_model_file)

                if dev_meteor > best_meteor:
                    logger.info("  Best METEOR: %s", dev_meteor)
                    best_meteor = dev_meteor
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-meteor')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                
                logger.info("  " + "*" * 20)

            # Store model training metrics
            model_log_dir = os.path.join(args.output_dir, 'training_result.csv')
            with open(model_log_dir, 'a') as f:
                if int(epoch) == 0:
                    f.write("model,experiment,epoch,train_loss,validation_loss,bleu-4,best-bleu-4,bertscore,best-bertscore,rouge-l,best-rouge-l,meteor,best-meteor,train_num_exceed_token_limit,valid_num_exceed_token_limit\n")
                f.write(f"codebert,{args.exp},{epoch},{train_loss},{eval_loss},{dev_bleu},{best_bleu},{dev_bertscore},{best_bertscore},{dev_rouge_l},{best_rouge_l},{str(dev_meteor)},{best_meteor},{train_num_exceed_token_limit},{valid_num_exceed_token_limit}\n")
                f.close()

            # Stop training if no improvement after a specified number of epochs
            if epochs_no_improve >= args.stop_no_improve_epochs:
                logger.info("Early stopping triggered after %d epochs without improvement", epochs_no_improve)
                break

    if args.do_test:
        logger.info("  " + "***** Testing *****")

        # Define the criteria for loading different best models
        criteria_list = ['last', 'best-bleu', 'best-ppl']

        for criteria in criteria_list:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint-{criteria}', 'pytorch_model.bin')

            if os.path.exists(checkpoint_path):
                logger.info(f"Loading model from checkpoint-{criteria} for testing.")
                model.load_state_dict(torch.load(checkpoint_path))
                model.to(device)

                logger.info(f"***** Running test with {criteria} checkpoint *****")
                test(args, model, tokenizer, device, epoch, criteria)
            else:
                logger.warning(f"Checkpoint for {criteria} not found at {checkpoint_path}. Skipping this test.")

if __name__ == "__main__":
    main()