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

import os
import logging
import argparse
import math
import numpy as np
from tqdm import tqdm
import multiprocessing
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from models import build_or_load_gen_model
from evaluator import smooth_bleu
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu
# from utils import get_filenames, get_elapse_time, load_and_cache_gen_data
from utils_exp import get_filenames, get_elapse_time, load_and_cache_gen_data
from configs import add_args, set_seed, set_dist

from bert_score import score # BERTscore
from rouge_score import rouge_scorer # ROUGE-L
import nltk
from nltk.translate.meteor_score import meteor_score # METEOR
from nltk.tokenize import word_tokenize
import string
nltk.download('wordnet')

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=4, pin_memory=True)
    
    # Start evaluating model
    logger.info("  " + "***** Running Perplexity Evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    eval_loss, batch_num = 0, 0
    num_exceed_limit = 0  # Counter for inputs exceeding token limit

    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc=f"Evaluating Perplexity on {len(eval_examples)} examples"):
        # Unpack the batch, including the exceeds_token_limits flag
        source_ids, target_ids, exceeds_token_limits = batch
        source_ids = source_ids.to(args.device)
        target_ids = target_ids.to(args.device)
        exceeds_token_limits = exceeds_token_limits.to(args.device)

        source_mask = source_ids.ne(tokenizer.pad_token_id)
        target_mask = target_ids.ne(tokenizer.pad_token_id)

        # Use the precomputed exceeds_token_limits flag to count records exceeding the token length limit
        exceed_count = exceeds_token_limits.sum().item()
        num_exceed_limit += exceed_count

        with torch.no_grad():
            if args.model_type == 'roberta':
                loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                   target_ids=target_ids, target_mask=target_mask)
            else:
                outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask)
                loss = outputs.loss

            if args.n_gpu > 1:
                loss = loss.mean()  # Average loss over all GPUs if using DataParallel

        eval_loss += loss.item()  # Accumulate the loss
        batch_num += 1

    # Calculate the average loss across batches
    if batch_num > 0:
        eval_loss /= batch_num
    else:
        eval_loss = float('inf')

    # Calculate perplexity
    eval_ppl = round(np.exp(eval_loss), 5) if eval_loss != float('inf') else float('inf')

    # Log the number of records exceeding the token limit
    percent_exceed_limit = (num_exceed_limit / len(eval_examples)) * 100 if len(eval_examples) > 0 else 0
    logger.info(f"[eval_ppl_epoch] Number of records exceeding token length limit in eval: {num_exceed_limit} out of {len(eval_examples)} ({percent_exceed_limit:.2f}% of records)")

    return eval_ppl, num_exceed_limit

def eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria):
    logger.info("  ***** Running bleu evaluation on {} data*****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_data)
    if args.data_num == -1:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     num_workers=4, pin_memory=True)
    else:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    change = False
    if isinstance(model,torch.nn.DataParallel):
        model = model.module
        change = True
    model.eval()
    pred_ids = []
    bleu, codebleu = 0.0, 0.0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for {} set".format(split_tag)):
        source_ids = batch[0].to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            if args.model_type == 'roberta':
                preds = model(source_ids=source_ids, source_mask=source_mask)

                top_preds = [pred[0].cpu().numpy() for pred in preds]
            else:
                preds = model.generate(
                                       input_ids=source_ids,
                                       attention_mask=source_mask,
                                       use_cache=True,
                                       num_beams=args.beam_size,
                                       early_stopping=args.task == 'summarize',
                                       max_length=args.max_target_length)
                top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)
    if change:
        model = torch.nn.DataParallel(model)
    pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]

    output_fn = os.path.join(args.res_dir, "test_{}.output".format(criteria))
    gold_fn = os.path.join(args.res_dir, "test_{}.gold".format(criteria))
    src_fn = os.path.join(args.res_dir, "test_{}.src".format(criteria))

    if args.task in ['defect']:
        target_dict = {0: 'false', 1: 'true'}
        golds = [target_dict[ex.target] for ex in eval_examples]
        eval_acc = np.mean([int(p == g) for p, g in zip(pred_nls, golds)])
        result = {'em': eval_acc * 100, 'bleu': 0, 'codebleu': 0}

        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                f.write(pred_nl.strip() + '\n')
                f1.write(target_dict[gold.target] + '\n')
                f2.write(gold.source.strip() + '\n')
    else:
        dev_accs, predictions = [], []
        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                dev_accs.append(pred_nl.strip() == gold.target.strip())
                if args.task in ['summarize']:
                    predictions.append(str(gold.idx) + '\t' + pred_nl)
                    f.write(str(gold.idx) + '\t' + pred_nl.strip() + '\n')
                    f1.write(str(gold.idx) + '\t' + gold.target.strip() + '\n')
                    f2.write(str(gold.idx) + '\t' + gold.source.strip() + '\n')
                else:
                    f.write(pred_nl.strip() + '\n')
                    f1.write(gold.target.strip() + '\n')
                    f2.write(gold.source.strip() + '\n')

        if args.task == 'summarize':
            (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_fn)
            bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        else:
            bleu = round(_bleu(gold_fn, output_fn), 2)
            if args.task in ['concode', 'translate', 'refine']:
                codebleu = calc_code_bleu.get_codebleu(gold_fn, output_fn, args.lang)

        result = {'em': np.mean(dev_accs) * 100, 'bleu': bleu}
        if args.task in ['concode', 'translate', 'refine']:
            result['codebleu'] = codebleu * 100

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result

def eval_metrics_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria):
    logger.info("  ***** Running BLEU evaluation on {} data *****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=4, pin_memory=True)

    # Handle DataParallel model case
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    model.eval()
    pred_ids = []
    num_exceed_limit = 0
    exceed_flags = []

    # Generate predictions
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc=f"Evaluating BLEU on {split_tag} set"):
        # Unpack the batch, including the exceeds_token_limits flag
        source_ids, target_ids, exceeds_token_limits = batch
        source_ids = source_ids.to(args.device)
        target_ids = target_ids.to(args.device)

        # Keep track of exceed limits
        num_exceed_limit += exceeds_token_limits.sum().item()
        exceed_flags.extend(exceeds_token_limits.cpu().tolist())

        source_mask = source_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            if args.model_type == 'roberta':
                preds = model(source_ids=source_ids, source_mask=source_mask)
                top_preds = [pred[0].cpu().numpy() for pred in preds]
            else:
                preds = model.generate(
                    input_ids=source_ids,
                    attention_mask=source_mask,
                    use_cache=True,
                    num_beams=args.beam_size,
                    early_stopping=args.task == 'summarize',
                    max_length=args.max_target_length
                )
                top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)

    # Log number of records exceeding token length limit
    percent_exceed_limit = (num_exceed_limit / len(eval_examples)) * 100 if len(eval_examples) > 0 else 0
    logger.info(f"[eval_metrics_epoch] Number of records in {split_tag} set exceeding token length limit: {num_exceed_limit} out of {len(eval_examples)} ({percent_exceed_limit:.2f}% of records)")

    # Decode predicted summaries
    pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]

    # Prepare filenames for output
    output_fn = os.path.join(args.res_dir, f"test_{criteria}.output")
    gold_fn = os.path.join(args.res_dir, f"test_{criteria}.gold")
    src_fn = os.path.join(args.res_dir, f"test_{criteria}.src")

    # Write results to files and collect gold summaries
    dev_accs, predictions = [], []
    gold_summaries = []
    with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
        for idx, (pred_nl, gold, exceeds_token_limit) in enumerate(zip(pred_nls, eval_examples, exceed_flags)):
            exceeds_flag = "True" if exceeds_token_limit else "False"
            dev_accs.append(pred_nl.strip() == gold.target.strip())

            # Collect predictions and gold summaries for BLEU and other metrics
            gold_summaries.append(gold.target)
            predictions.append(f"{str(gold.idx)}\t{pred_nl.strip()}\t{exceeds_flag}")

            # Write outputs
            if args.task in ['summarize']:
                f.write(f"{str(gold.idx)}\t{pred_nl.strip()}\t{exceeds_flag}\n")
                f1.write(f"{str(gold.idx)}\t{gold.target.strip()}\n")
                f2.write(f"{str(gold.idx)}\t{gold.source.strip()}\n")
            else:
                f.write(f"{pred_nl.strip()}\t{exceeds_flag}\n")
                f1.write(gold.target.strip() + '\n')
                f2.write(gold.source.strip() + '\n')

    # Extract predictions for BLEU calculation
    predictions_for_bleu = ["{}\t{}".format(prediction.split('\t')[0], prediction.split('\t')[1]) for prediction in predictions]

    # BLEU calculation
    if args.task == 'summarize':
        goldMap, predictionMap = smooth_bleu.computeMaps(predictions_for_bleu, gold_fn)
        bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
    else:
        bleu = round(_bleu(gold_fn, output_fn), 2)
        if args.task in ['concode', 'translate', 'refine']:
            codebleu = calc_code_bleu.get_codebleu(gold_fn, output_fn, args.lang)

    # Initialize additional metrics
    rouge_l_scores, meteor_scores = [], []

    # Calculate ROUGE-L and METEOR
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    for pred_nl, gold_summary in zip(pred_nls, gold_summaries):
        # ROUGE-L
        rouge_score = scorer.score(gold_summary, pred_nl)
        rouge_l_scores.append(rouge_score['rougeL'].fmeasure)

        # METEOR - Tokenize the hypothesis and reference sentences
        gold_tokens = gold_summary.split()  # Tokenize reference
        pred_tokens = pred_nl.split()       # Tokenize hypothesis
        meteor = meteor_score([gold_tokens], pred_tokens)  # Provide tokenized input to METEOR
        meteor_scores.append(meteor)

    avg_rouge_l = np.mean(rouge_l_scores)
    avg_meteor = np.mean(meteor_scores)

    # Calculate BERTScore
    P, R, F1 = score(pred_nls, gold_summaries, lang="en", model_type="roberta-base", rescale_with_baseline=True)
    avg_bert_f1 = F1.mean().item()

    # Create result dictionary with keys in the desired order
    result = {
        'bleu': bleu,
        'bertscore_f1': avg_bert_f1,
        'rouge_l': avg_rouge_l,
        'meteor': avg_meteor,
        'em': np.mean(dev_accs) * 100,
        'num_exceed_token_limit': num_exceed_limit  # Add count of records exceeding token length
    }

    if args.task in ['concode', 'translate', 'refine']:
        result['codebleu'] = codebleu * 100

    # Log evaluation results
    logger.info("***** Eval results *****")
    for key in result.keys():
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result

def main():
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logger.info(args)
    t0 = time.time()

    set_dist(args)
    set_seed(args)

    # Build or load the model, tokenizer, and configuration
    config, model, tokenizer = build_or_load_gen_model(args)
    special_tokens_dict = {
        'additional_special_tokens': ['<code>', '<caller>', '<callee>', '<history>', '<version>', '<days>']
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    model.to(args.device)
    if args.n_gpu > 1:
        # for DataParallel
        model = torch.nn.DataParallel(model)
    pool = multiprocessing.Pool(args.cpu_cont)
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task, args=args)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train:
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)

        # Prepare training data loader
        train_InputFeatures, train_data = load_and_cache_gen_data(args, args.train_filename, pool, tokenizer, 'train')
        train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      num_workers=4, pin_memory=True)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)

        dev_dataset = {}
        global_step, best_bleu_em, best_ppl, best_bertscore_f1, best_rouge_l, best_meteor = 0, -1, 1e6, -1, -1, -1
        not_loss_dec_cnt, not_bleu_em_inc_cnt = 0, 0 if args.do_eval_bleu else 1e6

        # Initialize counters for tracking records exceeding the token limit
        train_num_exceed_token_limit = 0
        valid_num_exceed_token_limit = 0

        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            model.train()
            for step, batch in enumerate(bar):
                # Unpack the batch, including the exceeds_token_limits flag
                source_ids, target_ids, exceeds_token_limits = batch
                source_ids = source_ids.to(args.device)
                target_ids = target_ids.to(args.device)
                exceeds_token_limits = exceeds_token_limits.to(args.device)

                source_mask = source_ids.ne(tokenizer.pad_token_id)
                target_mask = target_ids.ne(tokenizer.pad_token_id)

                # Count the number of records exceeding the token length limit during training
                exceed_count = exceeds_token_limits.sum().item()
                train_num_exceed_token_limit += exceed_count

                # Log the number of exceeding records in this batch
                # logger.info(f"[Training] Number of records exceeding token length limit in batch: {exceed_count}")

                if args.model_type == 'roberta':
                    loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                       target_ids=target_ids, target_mask=target_mask)
                else:
                    outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                    labels=target_ids, decoder_attention_mask=target_mask)
                    loss = outputs.loss

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))
            
            # Log the total number of records exceeding the token length limit after each epoch
            logger.info(f"[Epoch {cur_epoch}] Total number of records exceeding token length limit in training set: {train_num_exceed_token_limit}")

            if args.do_eval:
                # Eval model with dev dataset
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev')
                    dev_dataset['dev_loss'] = eval_examples, eval_data

                eval_ppl, valid_num_exceed_token_limit_epoch = eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer)
                valid_num_exceed_token_limit += valid_num_exceed_token_limit_epoch  # Accumulate the count of exceeding records

                result = {'epoch': cur_epoch, 'global_step': global_step, 'eval_ppl': eval_ppl}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)
                if args.data_num == -1:
                    tb_writer.add_scalar('dev_ppl', eval_ppl, cur_epoch)

                # Save last checkpoint
                if args.save_last_checkpoints:
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the last model into %s", output_model_file)

                if eval_ppl < best_ppl:
                    not_loss_dec_cnt = 0
                    logger.info("  Best ppl:%s", eval_ppl)
                    logger.info("  " + "*" * 20)
                    fa.write("[%d] Best ppl changed into %.4f\n" % (cur_epoch, eval_ppl))
                    best_ppl = eval_ppl

                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if args.always_save_model:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the best ppl model into %s", output_model_file)
                else:
                    not_loss_dec_cnt += 1
                    logger.info("Ppl does not decrease for %d epochs", not_loss_dec_cnt)
                    if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                        early_stop_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                            cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                        logger.info(early_stop_str)
                        fa.write(early_stop_str)
                        break
                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
                if args.do_eval_bleu:
                    eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev',
                                                                    only_src=False, is_sample=True)

                    # Perform evaluation and get metrics
                    result = eval_metrics_epoch(args, eval_data, eval_examples, model, tokenizer, 'dev', 'e%d' % cur_epoch)
                    dev_bleu, dev_em = result['bleu'], result['em']
                    dev_bertscore_f1 = result['bertscore_f1']  # BERTScore F1
                    dev_rouge_l = result['rouge_l']  # ROUGE-L
                    dev_meteor = result['meteor']  # METEOR

                    if args.task in ['summarize']:
                        dev_bleu_em = dev_bleu
                    elif args.task in ['defect']:
                        dev_bleu_em = dev_em
                    else:
                        dev_bleu_em = dev_bleu + dev_em

                    # Log metrics to TensorBoard
                    if args.data_num == -1:
                        tb_writer.add_scalar('dev_bleu_em', dev_bleu_em, cur_epoch)
                        tb_writer.add_scalar('dev_bertscore_f1', dev_bertscore_f1, cur_epoch)
                        tb_writer.add_scalar('dev_rouge_l', dev_rouge_l, cur_epoch)
                        tb_writer.add_scalar('dev_meteor', dev_meteor, cur_epoch)

                    # Track best BLEU + EM
                    if dev_bleu_em > best_bleu_em:
                        not_bleu_em_inc_cnt = 0
                        logger.info("  [%d] Best bleu+em: %.2f (bleu: %.2f, em: %.2f, bertscore_f1: %.2f, rouge_l: %.4f, meteor: %.4f)",
                                    cur_epoch, dev_bleu_em, dev_bleu, dev_em, dev_bertscore_f1, dev_rouge_l, dev_meteor)
                        logger.info("  " + "*" * 20)
                        best_bleu_em = dev_bleu_em
                        fa.write("[%d] Best bleu+em changed into %.2f (bleu: %.2f, em: %.2f, bertscore_f1: %.2f, rouge_l: %.4f, meteor: %.4f)\n" % (
                            cur_epoch, best_bleu_em, dev_bleu, dev_em, dev_bertscore_f1, dev_rouge_l, dev_meteor))
                        # Save best checkpoint for best BLEU
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        if args.data_num == -1 or args.always_save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Save the best bleu model into %s", output_model_file)
                    else:
                        not_bleu_em_inc_cnt += 1
                        logger.info("Bleu does not increase for %d epochs", not_bleu_em_inc_cnt)
                        fa.write(
                            "[%d] Best bleu+em (%.2f) does not drop changed for %d epochs, cur bleu+em: %.2f (bleu: %.2f, em: %.2f, bertscore_f1: %.2f, rouge_l: %.4f, meteor: %.4f)\n" % (
                                cur_epoch, best_bleu_em, not_bleu_em_inc_cnt, dev_bleu_em, dev_bleu, dev_em, dev_bertscore_f1, dev_rouge_l, dev_meteor))
                        if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                            stop_early_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                                cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                            logger.info(stop_early_str)
                            fa.write(stop_early_str)
                            break

            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

            # Store model training metrics
            model_log_dir = os.path.join(args.output_dir, 'training_result.csv')
            with open(model_log_dir, 'a') as f:
                if int(cur_epoch) == 0:
                    f.write(f"model,experiment,epoch,train_loss,validation_loss,bleu-4,best-bleu-4,bertscore_f1,best-bertscore_f1,rouge_l,best-rouge_l,meteor,best-meteor,train_num_exceed_token_limit,valid_num_exceed_token_limit\n")
                f.write(f"codet5,{args.exp},{cur_epoch},{round(train_loss, 4)},{round(eval_ppl, 4)},{round(dev_bleu_em, 4)},{round(best_bleu_em, 4)},{round(dev_bertscore_f1, 4)},{round(best_bertscore_f1, 4)},{round(dev_rouge_l, 4)},{round(best_rouge_l, 4)},{round(dev_meteor, 4)},{round(best_meteor, 4)},{train_num_exceed_token_limit},{valid_num_exceed_token_limit}\n")

        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()
        logger.info("Finish training and take %s", get_elapse_time(t0))

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        # Ensure the result file is open for appending
        with open(args.res_fn, 'a+') as fa:
            for criteria in ['last', 'best-bleu', 'best-ppl']:
                file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
                logger.info("Reload model from {}".format(file))

                # Reload model checkpoint
                try:
                    if hasattr(model, 'module'):
                        model.module.load_state_dict(torch.load(file))
                    else:
                        model.load_state_dict(torch.load(file))
                except FileNotFoundError:
                    logger.warning(f"Checkpoint file {file} not found, skipping evaluation for this checkpoint.")
                    continue
                except Exception as e:
                    logger.error(f"Error loading model checkpoint {file}: {str(e)}")
                    continue

                # Load test data
                eval_examples, eval_data = load_and_cache_gen_data(
                    args, args.test_filename, pool, tokenizer, 'test', only_src=False, is_sample=False
                )

                # Prepare DataLoader for evaluation
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                            num_workers=4, pin_memory=True)

                model.eval()
                pred_ids = []
                num_exceed_limit = 0

                # Evaluate model on test data
                for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc=f"Testing {criteria} set"):
                    # Unpack the batch, including the exceeds_token_limits flag
                    source_ids, target_ids, exceeds_token_limits = batch
                    source_ids = source_ids.to(args.device)
                    target_ids = target_ids.to(args.device)
                    exceeds_token_limits = exceeds_token_limits.to(args.device)

                    source_mask = source_ids.ne(tokenizer.pad_token_id)

                    # Count the number of records exceeding the token length limit for the batch
                    exceed_count = exceeds_token_limits.sum().item()
                    num_exceed_limit += exceed_count

                    with torch.no_grad():
                        if args.model_type == 'roberta':
                            preds = model(source_ids=source_ids, source_mask=source_mask)
                            top_preds = [pred[0].cpu().numpy() for pred in preds]
                        else:
                            preds = model.generate(
                                input_ids=source_ids,
                                attention_mask=source_mask,
                                use_cache=True,
                                num_beams=args.beam_size,
                                early_stopping=args.task == 'summarize',
                                max_length=args.max_target_length
                            )
                            top_preds = list(preds.cpu().numpy())
                        pred_ids.extend(top_preds)

                # Decode predicted summaries
                pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]

                # Prepare filenames for output
                output_fn = os.path.join(args.res_dir, "test_{}.output".format(criteria))
                gold_fn = os.path.join(args.res_dir, "test_{}.gold".format(criteria))
                src_fn = os.path.join(args.res_dir, "test_{}.src".format(criteria))

                # Write results to files
                with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
                    # Iterate through predictions and examples
                    for idx, (pred_nl, example, batch_tuple) in enumerate(zip(pred_nls, eval_examples, eval_data)):
                        # Unpack the batch_tuple properly
                        source_ids, target_ids, exceeds_token_limits = batch_tuple

                        # Convert exceeds_token_limits to a scalar value
                        exceeds_flag = "True" if exceeds_token_limits.item() == 1 else "False"

                        # Write outputs with exceeds flag
                        if args.task in ['summarize']:
                            f.write(f"{str(example.idx)}\t{pred_nl.strip()}\t{exceeds_flag}\n")
                            f1.write(f"{str(example.idx)}\t{example.target.strip()}\n")
                            f2.write(f"{str(example.idx)}\t{example.source.strip()}\n")
                        else:
                            f.write(f"{pred_nl.strip()}\t{exceeds_flag}\n")
                            f1.write(example.target.strip() + '\n')
                            f2.write(example.source.strip() + '\n')

                # Extract the count of records exceeding the token length limit
                test_num_exceed_token_limit = num_exceed_limit

                # Extract other results
                result = eval_metrics_epoch(args, eval_data, eval_examples, model, tokenizer, 'test', criteria)
                test_bleu = result.get('bleu', 0)
                test_em = result.get('em', 0)
                test_codebleu = result.get('codebleu', 0)
                test_bertscore_f1 = result.get('bertscore_f1', 0)
                test_rouge_l = result.get('rouge_l', 0)
                test_meteor = result.get('meteor', 0)

                # Log and write the results to file
                result_str = (
                    "[{}] bleu-4: {:.2f}, em: {:.4f}, codebleu: {:.4f}, bertscore_f1: {:.2f}, "
                    "rouge_l: {:.4f}, meteor: {:.4f}, num_exceed_token_limit: {}\n"
                ).format(
                    criteria, test_bleu, test_em, test_codebleu, test_bertscore_f1,
                    test_rouge_l, test_meteor, test_num_exceed_token_limit
                )

                logger.info(result_str)
                fa.write(result_str)

                if args.res_fn:
                    fa.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))

            # Finalize logging
            logger.info("Finish and take {}".format(get_elapse_time(t0)))
            fa.write("Finish and take {}\n".format(get_elapse_time(t0)))
            fa.close()

if __name__ == "__main__":
    main()



    # if args.do_test:
    #     logger.info("  " + "***** Testing *****")
    #     logger.info("  Batch size = %d", args.eval_batch_size)

    #     # Ensure the result file is open for appending
    #     with open(args.res_fn, 'a+') as fa:
    #         for criteria in ['last', 'best-bleu', 'best-ppl']:
    #             file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
    #             logger.info("Reload model from {}".format(file))
                
    #             # Reload model checkpoint
    #             try:
    #                 if hasattr(model, 'module'):
    #                     model.module.load_state_dict(torch.load(file))
    #                 else:
    #                     model.load_state_dict(torch.load(file))
    #             except FileNotFoundError:
    #                 logger.warning(f"Checkpoint file {file} not found, skipping evaluation for this checkpoint.")
    #                 continue
    #             except Exception as e:
    #                 logger.error(f"Error loading model checkpoint {file}: {str(e)}")
    #                 continue

    #             # Load test data
    #             eval_examples, eval_data = load_and_cache_gen_data(
    #                 args, args.test_filename, pool, tokenizer, 'test', only_src=True, is_sample=False
    #             )

    #             # Perform evaluation and get metrics
    #             result = eval_metrics_epoch(args, eval_data, eval_examples, model, tokenizer, 'test', criteria)

    #             # Extract the count of records exceeding the token length limit
    #             test_num_exceed_token_limit = result.get('num_exceed_token_limit', 0)

    #             # Extract other results
    #             test_bleu = result.get('bleu', 0)
    #             test_em = result.get('em', 0)
    #             test_codebleu = result.get('codebleu', 0)
    #             test_bertscore_f1 = result.get('bertscore_f1', 0)
    #             test_rouge_l = result.get('rouge_l', 0)
    #             test_meteor = result.get('meteor', 0)

    #             # Log and write the results to file
    #             result_str = (
    #                 "[{}] bleu-4: {:.2f}, em: {:.4f}, codebleu: {:.4f}, bertscore_f1: {:.2f}, "
    #                 "rouge_l: {:.4f}, meteor: {:.4f}, num_exceed_token_limit: {}\n"
    #             ).format(
    #                 criteria, test_bleu, test_em, test_codebleu, test_bertscore_f1,
    #                 test_rouge_l, test_meteor, test_num_exceed_token_limit
    #             )

    #             logger.info(result_str)
    #             fa.write(result_str)
                
    #             if args.res_fn:
    #                 fa.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))

    #         # Finalize logging
    #         logger.info("Finish and take {}".format(get_elapse_time(t0)))
    #         fa.write("Finish and take {}\n".format(get_elapse_time(t0)))
    #         fa.close()

# ==================== Old code ====================
                    # # Track best BERTScore F1
                    # if dev_bertscore_f1 > best_bertscore_f1:
                    #     best_bertscore_f1 = dev_bertscore_f1
                    #     logger.info("  [%d] Best bertscore_f1: %.2f", cur_epoch, dev_bertscore_f1)
                    #     logger.info("  " + "*" * 20)
                    #     fa.write("[%d] Best bertscore_f1 changed into %.2f\n" % (cur_epoch, best_bertscore_f1))
                    #     # Save best checkpoint for best BERTScore F1
                    #     output_dir = os.path.join(args.output_dir, 'checkpoint-best-bertscore')
                    #     if not os.path.exists(output_dir):
                    #         os.makedirs(output_dir)
                    #     if args.data_num == -1 or args.always_save_model:
                    #         model_to_save = model.module if hasattr(model, 'module') else model
                    #         output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    #         torch.save(model_to_save.state_dict(), output_model_file)
                    #         logger.info("Save the best bertscore model into %s", output_model_file)

                    # # Track best ROUGE-L
                    # if dev_rouge_l > best_rouge_l:
                    #     best_rouge_l = dev_rouge_l
                    #     logger.info("  [%d] Best rouge_l: %.4f", cur_epoch, dev_rouge_l)
                    #     logger.info("  " + "*" * 20)
                    #     fa.write("[%d] Best rouge_l changed into %.4f\n" % (cur_epoch, best_rouge_l))
                    #     # Save best checkpoint for best ROUGE-L
                    #     output_dir = os.path.join(args.output_dir, 'checkpoint-best-rouge-l')
                    #     if not os.path.exists(output_dir):
                    #         os.makedirs(output_dir)
                    #     if args.data_num == -1 or args.always_save_model:
                    #         model_to_save = model.module if hasattr(model, 'module') else model
                    #         output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    #         torch.save(model_to_save.state_dict(), output_model_file)
                    #         logger.info("Save the best rouge-l model into %s", output_model_file)

                    # # Track best METEOR
                    # if dev_meteor > best_meteor:
                    #     best_meteor = dev_meteor
                    #     logger.info("  [%d] Best meteor: %.4f", cur_epoch, dev_meteor)
                    #     logger.info("  " + "*" * 20)
                    #     fa.write("[%d] Best meteor changed into %.4f\n" % (cur_epoch, best_meteor))
                    #     # Save best checkpoint for best METEOR
                    #     output_dir = os.path.join(args.output_dir, 'checkpoint-best-meteor')
                    #     if not os.path.exists(output_dir):
                    #         os.makedirs(output_dir)
                    #     if args.data_num == -1 or args.always_save_model:
                    #         model_to_save = model.module if hasattr(model, 'module') else model
                    #         output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    #         torch.save(model_to_save.state_dict(), output_model_file)
                    #         logger.info("Save the best meteor model into %s", output_model_file)