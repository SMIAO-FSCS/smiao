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

class LineByLineTextDataset(Dataset):
    def __init__(self, file_path: str):
        assert os.path.isfile(file_path)
        print("read data file at:", file_path)

        with open(file_path, encoding="utf-8") as f:
            self.lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        # 截断测试
        self.lines = self.lines[:100000]

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

def main(file_path, model_name, tokenizer_name, code_pkl_out, query_pkl_out):
    print("run")
    # 配置

    valid_batch_size = 32


    ########################## 数据 #########################
    valid_dataset = LineByLineTextDataset(file_path=file_path)
    valid_dataLoader = DataLoader(valid_dataset, valid_batch_size, shuffle=False)

    # device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("train_device: ", device)

    ########## MODEL ##############################
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    model.to(device)

    ########valid########
    all_text_hidden_states = []
    all_code_hidden_states = []
    progress_bar_in = tqdm(range(len(valid_dataLoader)))
    for text, code, labels in valid_dataLoader:
        model.eval() #为了保证相同的sentence，拿出来的句子向量是一样的
        targets = labels.to(device)

        with torch.no_grad():
            with autocast():
                text_batch_tokenized = tokenizer(list(text), add_special_tokens=True,
                                                 padding=True, max_length=180,
                                                 truncation=True, return_tensors="pt").to(device)  # tokenize、add special token、pad
                code_batch_tokenized = tokenizer(list(code), add_special_tokens=True,
                                                 padding=True, max_length=180,
                                                 truncation=True, return_tensors="pt").to(device)  # tokenize、add special token、pad

                text_fea = model(**text_batch_tokenized, output_hidden_states=True).hidden_states
                code_fea = model(**code_batch_tokenized, output_hidden_states=True).hidden_states


                for h in text_fea[-1][:, 0, :]:
                    all_text_hidden_states.append(h.to('cpu').numpy())
                for h in code_fea[-1][:, 0, :]:
                    all_code_hidden_states.append(h.to('cpu').numpy())

            progress_bar_in.update(1)

    with torch.no_grad():
        save_pkl(query_pkl_out, all_text_hidden_states)
        save_pkl(code_pkl_out, all_code_hidden_states)


#这个是普通的方式抽取特征
if __name__ == '__main__':
    lang = "sql"
    # for lang in ["sql","solidity","ruby","python","php","go","java", "javascript"]:

    file_path = "../../../data/train_valid/" + lang +"/train.txt"

    model_name = "../../../graph_code_bert"
    tokenizer_name = model_name

    code_pkl_out = "./pkl/" + lang + "_code.pkl"
    query_pkl_out = "./pkl/" + lang + "_query.pkl"

    main(file_path, model_name, tokenizer_name, code_pkl_out, query_pkl_out)

