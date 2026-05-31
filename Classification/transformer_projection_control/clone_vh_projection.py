import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import torch
import time
import numpy as np
import warnings
import json
import os
from gensim.models.word2vec import Word2Vec
from sklearn.metrics import precision_recall_fscore_support
# warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.utils.data import DataLoader
from dataset import CodeCloneDataset
from utilities import SharedFunction

p_task = "clone" # "clone" or "classification"
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
                 label_size= 2
                ):
        super(BatchProgramClassifier, self).__init__()
        self.encoder_name = encoder_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
        self.config = AutoConfig.from_pretrained(self.encoder_name)
        self.config.num_labels = 1

        self.encoder = AutoModel.from_pretrained(self.encoder_name, config=self.config, use_safetensors=True)

        self.hidden_dim = hidden_dim
        self.label_size = label_size

        self.projection = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.hidden2label = nn.Linear(self.hidden_dim, self.label_size)

    def encode(self, input_ids):
        return self.encoder(input_ids,attention_mask=input_ids.ne(1))[0][:, 0,:]

    def forward(self, x1=None, x2=None, x3=None, x4=None, x8=None, x9=None, x10=None, y1=None, y2=None, y3=None, y4=None, y8=None, y9=None, y10=None):
        l_code, r_code = self.encode(x1), self.encode(y1)
        l_code_versions_all, r_code_versions_all = self.encode(x10), self.encode(y10)

        lvec = torch.cat([l_code, l_code_versions_all], 1)
        rvec = torch.cat([r_code, r_code_versions_all], 1)

        abs_dist = torch.abs(torch.add(lvec, -rvec))

        proj = self.dropout(F.relu(self.projection(abs_dist)))
        y = self.hidden2label(proj)

        return y

def get_batch_transformer(batch):
    x1 = batch['code_ids_x']
    x2 = batch['code_versions_ids_x']
    x3 = batch['calling_ids_x']
    x4 = batch['called_ids_x']
    x8 = batch['number_of_days_ids_x']
    x9 = batch['number_of_versions_ids_x']
    x10 = batch['code_versions_all_ids_x']

    y1 = batch['code_ids_y']
    y2 = batch['code_versions_ids_y']
    y3 = batch['calling_ids_y']
    y4 = batch['called_ids_y']
    y8 = batch['number_of_days_ids_y']
    y9 = batch['number_of_versions_ids_y']
    y10 = batch['code_versions_all_ids_y']

    train_labels = batch['label']
    return x1, x2, x3, x4, x8, x9, x10, y1, y2, y3, y4, y8, y9, y10, train_labels

if __name__ == '__main__':
    RANDOM_SEED = 42
    DATA_DIR = './data/clone_detection'
    MODEL_DIR = p_model_save_dir

    word2vec = Word2Vec.load(DATA_DIR + '/node_w2v_128').wv
    MAX_TOKENS = word2vec.vectors.shape[0]
    EMBEDDING_DIM = word2vec.vectors.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec.vectors.shape[0]] = word2vec.vectors

    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 1
    EPOCHS = 20
    BATCH_SIZE = p_batch_size
    USE_GPU = True

    torch.manual_seed(RANDOM_SEED)

    print(f"Train for clone detection - {p_model} with VERSION ALL - Clone Projection (fixed-dim control)")

    model = BatchProgramClassifier(p_model, hidden_dim=768, label_size=LABELS)

    train_data = CodeCloneDataset("train", model.tokenizer)
    val_data = CodeCloneDataset("dev", model.tokenizer)
    test_data = CodeCloneDataset("test", model.tokenizer)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
    dev_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters, lr=p_learning_rate)

    loss_function = torch.nn.BCEWithLogitsLoss()

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
            train_code_x, train_code_versions_x, train_calling_x, train_called_x, train_number_of_days_x, train_number_of_versions_x, train_code_versions_all_x, \
                train_code_y, train_code_versions_y, train_calling_y, train_called_y, train_number_of_days_y, train_number_of_versions_y, train_code_versions_all_y, \
                    train_labels = get_batch_transformer(batch)

            if USE_GPU:
                train_labels = train_labels.cuda()

            model.zero_grad()
            model.batch_size = len(train_labels)
            output = model(
                train_code_x, train_code_versions_x, train_calling_x, train_called_x, train_number_of_days_x, train_number_of_versions_x, train_code_versions_all_x, \
                train_code_y, train_code_versions_y, train_calling_y, train_called_y, train_number_of_days_y, train_number_of_versions_y, train_code_versions_all_y, \
                )

            loss = loss_function(output, train_labels.float())
            loss.backward()
            optimizer.step()

            output = torch.sigmoid(output)
            predicted = torch.round(output)
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
            dev_code_x, dev_code_versions_x, dev_calling_x, dev_called_x, dev_number_of_days_x, dev_number_of_versions_x, dev_code_versions_all_x, \
                dev_code_y, dev_code_versions_y, dev_calling_y, dev_called_y, dev_number_of_days_y, dev_number_of_versions_y, dev_code_versions_all_y, \
                dev_labels = get_batch_transformer(batch)

            if USE_GPU:
                dev_labels = dev_labels.cuda()

            model.batch_size = len(dev_labels)
            output = model(
                dev_code_x, dev_code_versions_x, dev_calling_x, dev_called_x, dev_number_of_days_x, dev_number_of_versions_x, dev_code_versions_all_x, \
                dev_code_y, dev_code_versions_y, dev_calling_y, dev_called_y, dev_number_of_days_y, dev_number_of_versions_y, dev_code_versions_all_y, \
                )

            loss = loss_function(output, dev_labels.float())

            output = torch.sigmoid(output)
            predicted = torch.round(output)
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
    torch.save(model.state_dict(), MODEL_DIR + '/clone_vh_projection.pth')

    """
    test
    """
    model = BatchProgramClassifier(p_model, hidden_dim=768, label_size=LABELS)
    model.load_state_dict(torch.load(MODEL_DIR + '/clone_vh_projection.pth'))

    if USE_GPU:
        model.cuda()

    predicts = []
    trues = []
    total_loss = 0.0
    total_acc = 0
    total = 0.0
    i = 0
    for batch in test_loader:
        test_code_x, test_code_versions_x, test_calling_x, test_called_x, test_number_of_days_x, test_number_of_versions_x, test_code_versions_all_x, \
            test_code_y, test_code_versions_y, test_calling_y, test_called_y, test_number_of_days_y, test_number_of_versions_y, test_code_versions_all_y, \
            test_labels = get_batch_transformer(batch)

        if USE_GPU:
            test_labels = test_labels.cuda()

        model.batch_size = len(test_labels)
        output = model(
                test_code_x, test_code_versions_x, test_calling_x, test_called_x, test_number_of_days_x, test_number_of_versions_x, test_code_versions_all_x, \
                test_code_y, test_code_versions_y, test_calling_y, test_called_y, test_number_of_days_y, test_number_of_versions_y, test_code_versions_all_y, \
                )

        output = torch.sigmoid(output)
        predicted = torch.round(output)
        for idx in range(len(predicted)):
            if predicted[idx] == test_labels[idx]:
                total_acc += 1
        total += len(test_labels)
        predicts.extend(predicted.cpu().detach().numpy())
        trues.extend(test_labels.cpu().numpy())

    acc = total_acc / total
    print("total accuracy: ", total_acc)
    p, r, f, _ = precision_recall_fscore_support(trues, predicts, average='binary')

    time_to_run = timer() - time_start
    str_time2run = str(datetime.timedelta(seconds=time_to_run))
    print("run start:", run_start)
    print("time to run: ", str_time2run)

    print("Total testing results(acc,P,R,F1):%.5f, %.5f, %.5f, %.5f" % (acc, p, r, f))

    model_log = "Transformer-%s-VersionAll-Clone Projection, %.5f, %.5f, %.5f, %.5f, %s" % (p_model, acc, p, r, f, str_time2run)
    obj = SharedFunction(model_log)
    obj.AppendFile()

    print(f"[RESULT] clone_vh_projection | {pars['model']} | F1: {f:.4f} | Precision: {p:.4f} | Recall: {r:.4f}")

    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_projection.json')
    if os.path.exists(result_path):
        with open(result_path) as fp:
            results = json.load(fp)
    else:
        results = {"clone_detection": {}, "classification": {}}
    model_name = pars['model'].lower()
    if model_name not in results["clone_detection"]:
        results["clone_detection"][model_name] = {}
    results["clone_detection"][model_name]["code+vh_projection"] = {"precision": float(p), "recall": float(r), "f1": float(f)}
    with open(result_path, 'w') as fp:
        json.dump(results, fp, indent=2)
