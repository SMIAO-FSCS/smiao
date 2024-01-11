import random

import torch
import os
from transformers import BertModel, BertTokenizer, RobertaConfig, RobertaModelWithHeads, PrefixTuningConfig, \
    AutoModelForSequenceClassification, TextClassificationPipeline, AutoAdapterModel
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
import pickle
from transformers import AutoTokenizer, AutoModel
#半精度
from torch.cuda.amp import autocast as autocast


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class LineByLineTextDataset(Dataset):
    def __init__(self, file_path: str):
        assert os.path.isfile(file_path)
        print("read data file at:", file_path)

        with open(file_path, encoding="utf-8") as f:
            self.lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        # 截断测试
        # self.lines = self.lines[:10000]

        self.text_lines = []
        self.code_lines = []
        self.labels = []

        for line in self.lines:
            temp_line = line.split("<CODESPLIT>")
            if (len(temp_line)) == 5:  # 确保<CODESPLIT>分开的每个部分都有值，不是Null
                if(str(temp_line[0]) == "1"): #只要正例
                    self.text_lines.append(temp_line[-2].lower()) #注释
                    self.code_lines.append(temp_line[-1].lower()) #代码
                    self.labels.append(int(temp_line[0]))

        print("注释和代码总行数:", len(self.text_lines), len(self.code_lines))

    def __len__(self):
        return len(self.text_lines)  # 注意这个len本质是数据的数量

    def __getitem__(self, i):
        a = self.text_lines[i]
        b = self.code_lines[i]
        c = self.labels[i]
        return a, b, c


def save_pkl(output_file, mean):
    with open(output_file, "wb") as fout:
        print(f"***** Saving features means.pkl *****")
        # mean = np.array([m.to('cpu').numpy() for m in mean])
        mean = np.array([m for m in mean])
        pickle.dump({'mean': mean}, fout)
    fout.close()

def extract_features_without_adapter(file_path, tokenizer_name, model_name, out_query_features_path, out_code_features_path):
    set_seed(1)
    print("run")
    # 配置

    valid_batch_size = 32
    valid_file_path = file_path

    fine_turn_model_name = model_name
    tokenizer_name = tokenizer_name

    ########################## 数据 #########################

    valid_dataset = LineByLineTextDataset(file_path=valid_file_path)
    valid_dataLoader = DataLoader(valid_dataset, valid_batch_size, shuffle=False)

    # device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("train_device: ", device)


    ########## MODEL ##############################
    model = AutoModelForSequenceClassification.from_pretrained(fine_turn_model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # adapter_name = model.load_adapter(adaptor_save_dir)
    # model.set_active_adapters(adapter_name)

    model.to(device)


    ########valid########
    all_text_hidden_states = []
    all_code_hidden_states = []
    progress_bar_in = tqdm(range(len(valid_dataLoader)))
    for text, code, labels in valid_dataLoader:
        model.eval()
        # targets = labels.to(device)

        with autocast():
            text_batch_tokenized = tokenizer(list(text), add_special_tokens=True,
                                             padding=True, max_length=128,
                                             truncation=True, return_tensors="pt").to(device)  # tokenize、add special token、pad
            # code_batch_tokenized = tokenizer(list(code), add_special_tokens=True,
            #                                  padding=True, max_length=128,
            #                                  truncation=True, return_tensors="pt").to(device)  # tokenize、add special token、pad

            # model.eval() #为了保证相同的sentence，拿出来的句子向量是一样的
            text_fea = model(**text_batch_tokenized, output_hidden_states=True).hidden_states
            # code_fea = model(**code_batch_tokenized, output_hidden_states=True).hidden_states
            # print(len(outputs.hidden_states)) #13，说明取的是hidden_states，这个hidden_states会不会受到adaptor的影响呢？

            with torch.no_grad():
                for h in text_fea[-1][:, 0, :]:
                    all_text_hidden_states.append(h.to('cpu').numpy())
                # for h in code_fea[-1][:, 0, :]:
                #     all_code_hidden_states.append(h.to('cpu').numpy())

            progress_bar_in.update(1)

    with torch.no_grad():
        save_pkl(out_query_features_path, all_text_hidden_states)
        # save_pkl(out_code_features_path, all_code_hidden_states)
        all_text_hidden_states = []
        all_code_hidden_states = []


#这个是普通的方式抽取特征
if __name__ == '__main__':
    set_seed(1)
    lang = 'solidity'

    in_file_path = "../../../data/train_valid/"+ lang +"/train_confident.txt"
    out_query_features_path = "text.pkl"
    out_code_features_path = "code.pkl"

    fine_turn_model_name = "../../../graph_code_bert"
    tokenizer_name = "../../../graph_code_bert"

    extract_features_without_adapter(in_file_path, fine_turn_model_name,  tokenizer_name, out_query_features_path, out_code_features_path)

#感觉可能检索不到困难负样本，但是可以生成，质量不高的生成当作困难副样本
#或者全网检索困难负样本
#或者基于原来的query，通过替换token，改造成困难负样本
