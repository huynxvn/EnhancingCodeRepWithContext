import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import random
import pandas as pd
import time
import numpy as np
import warnings
import json
import os
from gensim.models.word2vec import Word2Vec
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support
from utilities import SharedFunction
# warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.utils.data import DataLoader
from dataset import CodeClassificationDataset

p_task = "classification" # "clone" or "classification"
# ===============================================================================
# PARSING ARGUMENTS
import sys
sys.path.append('./')
import global_config as cfg
import datetime
from timeit import default_timer as timer
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model', nargs='?', default="CodeBERT", help="type of model checkpoints: CodeBERT, CodeBERTa, GraphCodeBERT, CodeT5", type=str)

args = parser.parse_args()
pars = {
    'model': args.model
}

# log date and time
run_start = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
time_start = timer()


# params to variables
configs = cfg.experiment_settings

# find items in list of dictionaries where model_name == pars['model']
exp_setting = next(item for item in configs if item["model_name"].lower() == pars['model'].lower())
p_model = exp_setting["model_checkpoint"]
p_model_save_dir = exp_setting["model_save_dir"] + "/versionall_projection"
if p_task == "clone":
    p_learning_rate = exp_setting["learning_rate_clone"]
    p_batch_size = exp_setting["batch_size_clone"]
else: # classification
    p_learning_rate = exp_setting["learning_rate_class"]
    p_batch_size = exp_setting["batch_size_class"]
SharedFunction.init_folder(p_model_save_dir)
# ===============================================================================

class BatchProgramClassifier(nn.Module):
    def __init__(self,
                 encoder_name,
                 hidden_dim,
                 label_size
                ):
        super(BatchProgramClassifier, self).__init__()

        self.encoder_name = encoder_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
        self.config = AutoConfig.from_pretrained(self.encoder_name)

        self.encoder = AutoModel.from_pretrained(self.encoder_name, config=self.config, use_safetensors=True)

        self.hidden_dim = hidden_dim
        self.label_size = label_size

        self.projection = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.hidden2label = nn.Linear(self.hidden_dim, self.label_size)

    def encode(self, input_ids):
        return self.encoder(input_ids,attention_mask=input_ids.ne(1))[0][:, 0,:]

    def encode_number(self, x):
        return x

    def forward(self, x1=None, x2=None, x3=None, x4=None, x8=None, x9=None, x10=None):
        code = self.encode(x1)
        code_versions_all = self.encode(x10)

        inputs = torch.cat([code, code_versions_all], dim=1)

        proj = self.dropout(F.relu(self.projection(inputs)))
        y = self.hidden2label(proj)
        return y

def get_batch_transformer(batch):
    x1 = batch['code_ids']
    x2 = batch['code_versions_ids']
    x3 = batch['calling_ids']
    x4 = batch['called_ids']
    x8 = batch['number_of_days_ids']
    x9 = batch['number_of_versions_ids']
    x10 = batch['code_versions_all_ids']

    train_labels = batch['label']
    return x1, x2, x3, x4, x8, x9, x10, train_labels

if __name__ == '__main__':
    RANDOM_SEED = 42
    MODEL_DIR = p_model_save_dir

    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 11
    EPOCHS = 20
    BATCH_SIZE = p_batch_size
    USE_GPU = True

    torch.manual_seed(RANDOM_SEED)

    print(f"Train for classification - {p_model} - PROJECTION - With VERSION ALL (fixed-dim control)")

    model = BatchProgramClassifier(p_model, hidden_dim=768, label_size=LABELS)

    train_data = CodeClassificationDataset("train", model.tokenizer)
    val_data = CodeClassificationDataset("dev", model.tokenizer)
    test_data = CodeClassificationDataset("test", model.tokenizer)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
    dev_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters, lr=p_learning_rate)

    loss_function = torch.nn.CrossEntropyLoss()

    precision, recall, f1 = 0, 0, 0
    print('Start training...')
    run_start = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    time_start = timer()

    best_loss = 10
    best_model = None
    for epoch in range(EPOCHS):
        start_time = time.time()
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0

        for batch in train_loader:
            train_code, train_code_versions, train_calling, train_called, train_number_of_days, train_number_of_versions, train_code_versions_all, \
                train_labels = get_batch_transformer(batch)

            if USE_GPU:
                train_labels = train_labels.cuda()

            model.zero_grad()
            model.batch_size = len(train_labels)

            output = model(train_code, train_code_versions, train_calling, train_called, train_number_of_days, train_number_of_versions, train_code_versions_all)

            train_labels = train_labels.squeeze(1)
            loss = loss_function(output, train_labels.long())

            loss.backward()
            optimizer.step()

            log_prediction = torch.softmax(output, dim=1)
            predicted = torch.argmax(log_prediction, dim=1)
            for idx in range(len(predicted)):
                if predicted[idx] == train_labels[idx]:
                    total_acc += 1
            total += len(train_labels)
            total_loss += loss.item() * len(train_labels)
        train_loss = total_loss / total
        train_acc = total_acc / total

        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        for batch in dev_loader:
            dev_code, dev_code_versions, dev_calling, dev_called, dev_number_of_days, dev_number_of_versions, dev_code_versions_all, \
                dev_labels = get_batch_transformer(batch)

            if USE_GPU:
                dev_labels = dev_labels.cuda()

            model.batch_size = len(dev_labels)
            output = model(dev_code, dev_code_versions, dev_calling, dev_called, dev_number_of_days, dev_number_of_versions, dev_code_versions_all)

            dev_labels = dev_labels.squeeze(1)
            loss = loss_function(output, dev_labels.long())

            log_prediction = torch.softmax(output, dim=1)
            predicted = torch.argmax(log_prediction, dim=1)
            for idx in range(len(predicted)):
                if predicted[idx] == dev_labels[idx]:
                    total_acc += 1
            total += len(dev_labels)
            total_loss += loss.item() * len(dev_labels)
        epoch_loss = total_loss / total
        epoch_acc = total_acc / total
        end_time = time.time()
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = model
        print('[Epoch: %3d/%3d] Train Loss: %.4f, Validation Loss: %.4f, '
              'Train Acc: %.3f, Validation Acc: %.3f, Time Cost: %.3f s'
              % (epoch + 1, EPOCHS, train_loss, epoch_loss, train_acc,
                 epoch_acc, end_time - start_time))

    model = best_model
    torch.save(model.state_dict(), MODEL_DIR + '/class_vh_projection.pth')

    """
    test
    """
    if USE_GPU:
        model.cuda()

    predicts = []
    trues = []
    total_loss = 0.0
    total_acc = 0
    total = 0.0
    i = 0
    for batch in test_loader:
        test_code, test_code_versions, test_calling, test_called, test_number_of_days, test_number_of_versions, test_code_versions_all, \
                test_labels = get_batch_transformer(batch)

        if USE_GPU:
            test_labels = test_labels.cuda()

        model.batch_size = len(test_labels)
        output = model(test_code, test_code_versions, test_calling, test_called, test_number_of_days, test_number_of_versions, test_code_versions_all)

        log_prediction = torch.softmax(output, dim=1)
        predicted = torch.argmax(log_prediction, dim=1)
        for idx in range(len(predicted)):
            if predicted[idx] == test_labels[idx]:
                total_acc += 1
        total += len(test_labels)
        predicts.extend(predicted.cpu().detach().numpy())
        trues.extend(test_labels.cpu().numpy())

    acc = total_acc / total
    print(f"Total accuracy: {total_acc}/{int(total)}")
    p, r, f, _ = precision_recall_fscore_support(trues, predicts, average='macro')

    time_to_run = timer() - time_start
    str_time2run = str(datetime.timedelta(seconds=time_to_run))
    print("run start:", run_start)
    print("time to run: ", str_time2run)

    print("Total testing results(acc,P,R,F1):%.5f, %.5f, %.5f, %.5f" % (acc, p, r, f))

    model_log = "Transformer-%s-VersionAll-Class Projection, %.5f, %.5f, %.5f, %.5f, %s" % (p_model, acc, p, r, f, str_time2run)
    obj = SharedFunction(model_log)
    obj.AppendFile()

    print(f"[RESULT] class_vh_projection | {pars['model']} | Accuracy: {acc:.4f} | Macro-F1: {f:.4f}")

    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_projection.json')
    if os.path.exists(result_path):
        with open(result_path) as fp:
            results = json.load(fp)
    else:
        results = {"clone_detection": {}, "classification": {}}
    model_name = pars['model'].lower()
    if model_name not in results["classification"]:
        results["classification"][model_name] = {}
    results["classification"][model_name]["code+vh_projection"] = {"accuracy": float(acc), "macro_f1": float(f)}
    with open(result_path, 'w') as fp:
        json.dump(results, fp, indent=2)
