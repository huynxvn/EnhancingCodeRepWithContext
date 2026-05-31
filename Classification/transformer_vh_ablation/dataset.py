from torch.utils.data import Dataset
import json
import os
import torch
import pickle as pkl

class InputFeatures(object):
    """
    A single training/test features for a data instance.
    """

    def __init__(self,
                 code_tokens_x, code_tokens_y,
                 code_versions_tokens_x, code_versions_tokens_y,
                 calling_token_x, calling_token_y,
                 called_token_x, called_token_y,
                 number_of_days_token_x, number_of_days_token_y,
                 number_of_versions_token_x, number_of_versions_token_y,
                 versions_all_token_x, versions_all_token_y,

                 code_ids_x, code_ids_y,
                 code_versions_ids_x, code_versions_ids_y,
                 calling_ids_x, calling_ids_y,
                 called_ids_x, called_ids_y,
                 number_of_days_ids_x, number_of_days_ids_y,
                 number_of_versions_ids_x, number_of_versions_ids_y,
                 versions_all_ids_x, versions_all_ids_y,

                 idx_x, idx_y,

                 label,

    ):
        # token
        self.code_tokens_x = code_tokens_x
        self.code_versions_tokens_x = code_versions_tokens_x
        self.calling_token_x = calling_token_x
        self.called_token_x = called_token_x
        self.number_of_days_token_x = number_of_days_token_x
        self.number_of_versions_token_x = number_of_versions_token_x
        self.versions_all_token_x = versions_all_token_x

        self.code_tokens_y = code_tokens_y
        self.code_versions_tokens_y = code_versions_tokens_y
        self.calling_token_y = calling_token_y
        self.called_token_y = called_token_y
        self.number_of_days_token_y = number_of_days_token_y
        self.number_of_versions_token_y = number_of_versions_token_y
        self.versions_all_token_y = versions_all_token_y

        #ids_x
        self.code_ids_x = code_ids_x
        self.code_versions_ids_x = code_versions_ids_x
        self.calling_ids_x = calling_ids_x
        self.called_ids_x = called_ids_x
        self.number_of_days_ids_x = number_of_days_ids_x
        self.number_of_versions_ids_x = number_of_versions_ids_x
        self.code_versions_all_ids_x = versions_all_ids_x

        self.code_ids_y = code_ids_y
        self.code_versions_ids_y = code_versions_ids_y
        self.calling_ids_y = calling_ids_y
        self.called_ids_y = called_ids_y
        self.number_of_days_ids_y = number_of_days_ids_y
        self.number_of_versions_ids_y = number_of_versions_ids_y
        self.code_versions_all_ids_y = versions_all_ids_y

        # id num
        self.idx_x=str(idx_x)
        self.idx_y=str(idx_y)

        # label
        self.label=label

class InputFeaturesClassification(object):
    """
    A single training/test features for a data instance.
    """

    def __init__(self,
                 code_tokens,
                 code_versions_tokens,
                 calling_token,
                 called_token,
                 number_of_days_token,
                 number_of_versions_token,
                 versions_all_token,

                 code_ids,
                 code_versions_ids,
                 calling_ids,
                 called_ids,
                 number_of_days_ids,
                 number_of_versions_ids,
                 versions_all_ids,

                 idx,

                 label,

    ):
        # token
        self.code_tokens = code_tokens
        self.code_versions_tokens = code_versions_tokens
        self.calling_token = calling_token
        self.called_token = called_token
        self.number_of_days_token = number_of_days_token
        self.number_of_versions_token = number_of_versions_token
        self.versions_all_token = versions_all_token

        #ids
        self.code_ids = code_ids
        self.code_versions_ids = code_versions_ids
        self.calling_ids = calling_ids
        self.called_ids = called_ids
        self.number_of_days_ids = number_of_days_ids
        self.number_of_versions_ids = number_of_versions_ids
        self.code_versions_all_ids = versions_all_ids

        # id num
        self.idx=str(idx)

        # label
        self.label=label

class CodeCloneDataset(Dataset):
    def __init__(self,
                 data_type,
                 tokenizer,
                 max_versions='all',
                 is_force = True
                 ):

        self.max_versions = max_versions
        self.raw_path = f"./data/clone_detection_vh_ablation/{data_type}_blocks.pkl"
        self.save_dir = f"./data/clone_detection_vh_ablation/processed_{max_versions}/{data_type}/"

        self.save_path = os.path.join(self.save_dir, data_type + "_blocks.pkl")
        self.tokenizer = tokenizer
        self.data = []

        #Process
        if self.has_cache() and not is_force:
            self.load()
        else:
            self.process()
            self.save()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        label = 1 if self.data[i].label > 0 else 0
        return {
            "code_ids_x": torch.tensor(self.data[i].code_ids_x, dtype=torch.long).cuda(),
            "code_versions_ids_x": torch.tensor(self.data[i].code_versions_ids_x, dtype=torch.long).cuda(),
            "calling_ids_x": torch.tensor(self.data[i].calling_ids_x, dtype=torch.long).cuda(),
            "called_ids_x": torch.tensor(self.data[i].called_ids_x, dtype=torch.long).cuda(),
            "number_of_days_ids_x": torch.tensor(self.data[i].number_of_days_ids_x, dtype=torch.float).cuda(),
            "number_of_versions_ids_x": torch.tensor(self.data[i].number_of_versions_ids_x, dtype=torch.float).cuda(),
            "code_versions_all_ids_x": torch.tensor(self.data[i].code_versions_all_ids_x, dtype=torch.long).cuda(),

            "code_ids_y": torch.tensor(self.data[i].code_ids_y, dtype=torch.long).cuda(),
            "code_versions_ids_y": torch.tensor(self.data[i].code_versions_ids_y, dtype=torch.long).cuda(),
            "calling_ids_y": torch.tensor(self.data[i].calling_ids_y, dtype=torch.long).cuda(),
            "called_ids_y": torch.tensor(self.data[i].called_ids_y, dtype=torch.long).cuda(),
            "number_of_days_ids_y": torch.tensor(self.data[i].number_of_days_ids_y, dtype=torch.float).cuda(),
            "number_of_versions_ids_y": torch.tensor(self.data[i].number_of_versions_ids_y, dtype=torch.float).cuda(),
            "code_versions_all_ids_y": torch.tensor(self.data[i].code_versions_all_ids_y, dtype=torch.long).cuda(),

            "label": torch.tensor(label, dtype=torch.long).unsqueeze(0).cuda(),
        }

    def load_raw_data(self):
        with open(self.raw_path, "rb") as pf:
            data_list = pkl.load(pf)
            pf.close()
        return data_list

    def extract_features(self, js, tokenizer):
        code_x                  = js['t_code_x']
        code_versions_x         = js['t_code_versions_x']
        calling_x               = js['t_calling_x']
        called_x                = js['t_called_x']
        number_of_days_x        = js['number_of_days_x']
        number_of_versions_x    = js['number_of_versions_x']

        # Select version history column based on max_versions
        if self.max_versions == 1:
            code_versions_all_x = js['t_code_versions_all_1_x']
            code_versions_all_y = js['t_code_versions_all_1_y']
        elif self.max_versions == 3:
            code_versions_all_x = js['t_code_versions_all_3_x']
            code_versions_all_y = js['t_code_versions_all_3_y']
        else:  # 'all' or any other value
            code_versions_all_x = js['t_code_versions_all_x']
            code_versions_all_y = js['t_code_versions_all_y']

        code_y                  = js['t_code_y']
        code_versions_y         = js['t_code_versions_y']
        calling_y               = js['t_calling_y']
        called_y                = js['t_called_y']
        number_of_days_y        = js['number_of_days_y']
        number_of_versions_y    = js['number_of_versions_x']

        #Truncation
        code_tokens_x                   = tokenizer.tokenize(code_x)[:tokenizer.max_len_single_sentence-2]
        code_versions_token_x           = tokenizer.tokenize(code_versions_x)[:tokenizer.max_len_single_sentence-2]
        calling_token_x                 = tokenizer.tokenize(calling_x)[:tokenizer.max_len_single_sentence-2]
        called_token_x                  = tokenizer.tokenize(called_x)[:tokenizer.max_len_single_sentence-2]
        code_versions_all_token_x       = tokenizer.tokenize(code_versions_all_x)[:tokenizer.max_len_single_sentence-2]

        code_tokens_y                   = tokenizer.tokenize(code_y)[:tokenizer.max_len_single_sentence-2]
        code_versions_token_y           = tokenizer.tokenize(code_versions_y)[:tokenizer.max_len_single_sentence-2]
        calling_token_y                 = tokenizer.tokenize(calling_y)[:tokenizer.max_len_single_sentence-2]
        called_token_y                  = tokenizer.tokenize(called_y)[:tokenizer.max_len_single_sentence-2]
        code_versions_all_token_y       = tokenizer.tokenize(code_versions_all_y)[:tokenizer.max_len_single_sentence-2]

        #Add CLS + SEP
        cs_code_tokens_x =[tokenizer.cls_token]+code_tokens_x+[tokenizer.sep_token]
        cs_code_versions_tokens_x =[tokenizer.cls_token]+code_versions_token_x+[tokenizer.sep_token]
        cs_calling_token_x =[tokenizer.cls_token]+calling_token_x+[tokenizer.sep_token]
        cs_called_token_x =[tokenizer.cls_token]+called_token_x+[tokenizer.sep_token]
        cs_code_versions_all_token_x =[tokenizer.cls_token]+code_versions_all_token_x+[tokenizer.sep_token]

        cs_code_tokens_y =[tokenizer.cls_token]+code_tokens_y+[tokenizer.sep_token]
        cs_code_versions_tokens_y =[tokenizer.cls_token]+code_versions_token_y+[tokenizer.sep_token]
        cs_calling_token_y =[tokenizer.cls_token]+calling_token_y+[tokenizer.sep_token]
        cs_called_token_y =[tokenizer.cls_token]+called_token_y+[tokenizer.sep_token]
        cs_code_versions_all_token_y =[tokenizer.cls_token]+code_versions_all_token_y+[tokenizer.sep_token]

        #Convert tokens to ids
        code_ids_x =  tokenizer.convert_tokens_to_ids(cs_code_tokens_x)
        code_versions_ids_x =  tokenizer.convert_tokens_to_ids(cs_code_versions_tokens_x)
        calling_ids_x =  tokenizer.convert_tokens_to_ids(cs_calling_token_x)
        called_ids_x =  tokenizer.convert_tokens_to_ids(cs_called_token_x)
        code_versions_all_ids_x =  tokenizer.convert_tokens_to_ids(cs_code_versions_all_token_x)

        code_ids_y =  tokenizer.convert_tokens_to_ids(cs_code_tokens_y)
        code_versions_ids_y =  tokenizer.convert_tokens_to_ids(cs_code_versions_tokens_y)
        calling_ids_y =  tokenizer.convert_tokens_to_ids(cs_calling_token_y)
        called_ids_y =  tokenizer.convert_tokens_to_ids(cs_called_token_y)
        code_versions_all_ids_y =  tokenizer.convert_tokens_to_ids(cs_code_versions_all_token_y)

        #Padding
        code_ids_x+=[tokenizer.pad_token_id]*(tokenizer.max_len_single_sentence - len(code_ids_x))
        code_versions_ids_x+=[tokenizer.pad_token_id]*(tokenizer.max_len_single_sentence - len(code_versions_ids_x))
        calling_ids_x+=[tokenizer.pad_token_id]*(tokenizer.max_len_single_sentence - len(calling_ids_x))
        called_ids_x+=[tokenizer.pad_token_id]*(tokenizer.max_len_single_sentence - len(called_ids_x))
        code_versions_all_ids_x+=[tokenizer.pad_token_id]*(tokenizer.max_len_single_sentence - len(code_versions_all_ids_x))

        code_ids_y+=[tokenizer.pad_token_id]*(tokenizer.max_len_single_sentence - len(code_ids_y))
        code_versions_ids_y+=[tokenizer.pad_token_id]*(tokenizer.max_len_single_sentence - len(code_versions_ids_y))
        calling_ids_y+=[tokenizer.pad_token_id]*(tokenizer.max_len_single_sentence - len(calling_ids_y))
        called_ids_y+=[tokenizer.pad_token_id]*(tokenizer.max_len_single_sentence - len(called_ids_y))
        code_versions_all_ids_y+=[tokenizer.pad_token_id]*(tokenizer.max_len_single_sentence - len(code_versions_all_ids_y))

        return InputFeatures(
            cs_code_tokens_x, cs_code_tokens_y,
            cs_code_versions_tokens_x, cs_code_versions_tokens_y,
            cs_calling_token_x, cs_calling_token_y,
            cs_called_token_x, cs_called_token_y,
            number_of_days_x, number_of_days_y,
            number_of_versions_x, number_of_versions_y,
            cs_code_versions_all_token_x, cs_code_versions_all_token_y,

            code_ids_x, code_ids_y,
            code_versions_ids_x, code_versions_ids_y,
            calling_ids_x, calling_ids_y,
            called_ids_x, called_ids_y,
            number_of_days_x, number_of_days_y,
            number_of_versions_x, number_of_versions_y,
            code_versions_all_ids_x, code_versions_all_ids_y,

            js['id1'], js['id2'],
            int(js['label']))

    def process(self):
        json_data = self.load_raw_data()
        for data in json_data.to_dict('records'):
            features = self.extract_features(data, self.tokenizer)
            self.data.append(features)

    def save(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        with open(self.save_path, "wb") as pf:
            pkl.dump({'data': self.data}, pf)

    def load(self):
        with open(self.save_path, "rb") as pf:
            self.data = pkl.load(pf)

    def has_cache(self):
        if os.path.exists(self.save_path):
            print("Data exists")
            return True
        return False

class CodeClassificationDataset(Dataset):
    def __init__(self,
                 data_type,
                 tokenizer,
                 max_versions='all',
                 is_force = True
                 ):

        self.max_versions = max_versions
        self.raw_path = f"./data/classification_vh_ablation/{data_type}_df.pkl"
        self.save_dir = f"./data/classification_vh_ablation/processed_{max_versions}/{data_type}/"

        self.save_path = os.path.join(self.save_dir, data_type + "_blocks.pkl")
        self.tokenizer = tokenizer
        self.data = []

        #Process
        if self.has_cache() and not is_force:
            self.load()
        else:
            self.process()
            self.save()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        label = self.data[i].label

        return {
            "code_ids": torch.tensor(self.data[i].code_ids, dtype=torch.long).cuda(),
            "code_versions_ids": torch.tensor(self.data[i].code_versions_ids, dtype=torch.long).cuda(),
            "calling_ids": torch.tensor(self.data[i].calling_ids, dtype=torch.long).cuda(),
            "called_ids": torch.tensor(self.data[i].called_ids, dtype=torch.long).cuda(),
            "number_of_days_ids": torch.tensor(self.data[i].number_of_days_ids, dtype=torch.float).cuda(),
            "number_of_versions_ids": torch.tensor(self.data[i].number_of_versions_ids, dtype=torch.float).cuda(),
            "code_versions_all_ids": torch.tensor(self.data[i].code_versions_all_ids, dtype=torch.long).cuda(),

            "label": torch.tensor(label, dtype=torch.long).unsqueeze(0).cuda(),
        }

    def load_raw_data(self):
        with open(self.raw_path, "rb") as pf:
            data_list = pkl.load(pf)
            pf.close()
        return data_list

    def extract_features(self, js, tokenizer):
        code                  = js['t_code']
        code_versions         = js['t_code_versions']
        calling               = js['t_calling']
        called                = js['t_called']
        number_of_days        = js['number_of_days']
        number_of_versions    = js['number_of_versions']

        # Select version history column based on max_versions
        if self.max_versions == 1:
            code_versions_all = js['t_code_versions_all_1']
        elif self.max_versions == 3:
            code_versions_all = js['t_code_versions_all_3']
        else:
            code_versions_all = js['t_code_versions_all']

        #Truncation
        code_tokens                   = tokenizer.tokenize(code)[:tokenizer.max_len_single_sentence-2]
        code_versions_token           = tokenizer.tokenize(code_versions)[:tokenizer.max_len_single_sentence-2]
        calling_token                 = tokenizer.tokenize(calling)[:tokenizer.max_len_single_sentence-2]
        called_token                  = tokenizer.tokenize(called)[:tokenizer.max_len_single_sentence-2]
        code_versions_all_token       = tokenizer.tokenize(code_versions_all)[:tokenizer.max_len_single_sentence-2]

        #Add CLS + SEP
        cs_code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        cs_code_versions_tokens =[tokenizer.cls_token]+code_versions_token+[tokenizer.sep_token]
        cs_calling_token =[tokenizer.cls_token]+calling_token+[tokenizer.sep_token]
        cs_called_token =[tokenizer.cls_token]+called_token+[tokenizer.sep_token]
        cs_code_versions_all_token =[tokenizer.cls_token]+code_versions_all_token+[tokenizer.sep_token]

        #Convert tokens to ids
        code_ids =  tokenizer.convert_tokens_to_ids(cs_code_tokens)
        code_versions_ids =  tokenizer.convert_tokens_to_ids(cs_code_versions_tokens)
        calling_ids =  tokenizer.convert_tokens_to_ids(cs_calling_token)
        called_ids =  tokenizer.convert_tokens_to_ids(cs_called_token)
        code_versions_all_ids =  tokenizer.convert_tokens_to_ids(cs_code_versions_all_token)

        #Padding
        code_ids+=[tokenizer.pad_token_id]*(tokenizer.max_len_single_sentence - len(code_ids))
        code_versions_ids+=[tokenizer.pad_token_id]*(tokenizer.max_len_single_sentence - len(code_versions_ids))
        calling_ids+=[tokenizer.pad_token_id]*(tokenizer.max_len_single_sentence - len(calling_ids))
        called_ids+=[tokenizer.pad_token_id]*(tokenizer.max_len_single_sentence - len(called_ids))
        code_versions_all_ids+=[tokenizer.pad_token_id]*(tokenizer.max_len_single_sentence - len(code_versions_all_ids))

        return InputFeaturesClassification(
            cs_code_tokens,
            cs_code_versions_tokens,
            cs_calling_token,
            cs_called_token,
            number_of_days,
            number_of_versions,
            cs_code_versions_all_token,

            code_ids,
            code_versions_ids,
            calling_ids,
            called_ids,
            number_of_days,
            number_of_versions,
            code_versions_all_ids,

            js['id'],
            int(js['label']))

    def process(self):
        json_data = self.load_raw_data()
        for data in json_data.to_dict('records'):
            features = self.extract_features(data, self.tokenizer)
            self.data.append(features)

    def save(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        with open(self.save_path, "wb") as pf:
            pkl.dump({'data': self.data}, pf)

    def load(self):
        with open(self.save_path, "rb") as pf:
            self.data = pkl.load(pf)

    def has_cache(self):
        if os.path.exists(self.save_path):
            print("Data exists")
            return True
        return False
