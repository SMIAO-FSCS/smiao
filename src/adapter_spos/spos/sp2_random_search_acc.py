import random
import torch
import os
from transformers import AutoModelForSequenceClassification
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from transformers import logging
logging.set_verbosity_error()

#半精度
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

class LineByLineTextDataset(Dataset):
    def __init__(self, file_path: str, train_num=0):
        assert os.path.isfile(file_path)
        # print("read data file at:", file_path)

        with open(file_path, encoding="utf-8") as f:
            self.lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        # 截断测试
        if(train_num != 0):
            self.lines = self.lines[:train_num]

        self.text_lines = []
        self.code_lines = []
        self.labels = []

        for line in self.lines:
            temp_line = line.split("<CODESPLIT>")
            if (len(temp_line)) == 5:  # 确保<CODESPLIT>分开的每个部分都有值，不是Null
                # if(str(temp_line[0]) == "1"): #1表示代码和注释对应着，0表示没对应
                self.text_lines.append(temp_line[-2].lower()) #注释
                self.code_lines.append(temp_line[-1].lower()) #代码
                self.labels.append(int(temp_line[0]))

        # print(self.text_lines[0])
        # print(self.code_lines[0])


        # print("TRAIN注释和代码总行数:", len(self.text_lines), len(self.code_lines))

    def __len__(self):
        return len(self.text_lines)  # 注意这个len本质是数据的数量

    def __getitem__(self, i):
        a = self.text_lines[i]
        b = self.code_lines[i]
        c = self.labels[i]
        return a, b, c


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

#随机采样路径
def random_sample_path():
    control_layters = []
    choice_list = np.random.rand(12).round().astype(np.uint8) #共11层，选择的值是0/1，0代表不插入该层，1代表插入

    for index, value in enumerate(choice_list):
        if (value == 0):  # 如果value是0，则表示移除
            control_layters.append(index)

    # print("control_layters{}, choice_list{} ".format(control_layters, choice_list))

    return control_layters

#随机采样一个路径，获得一个子网
def get_valid_model(base_valid_model, super_adapter_path, device):
    base_valid_model.delete_adapter('bottleneck_adapter')

    # 随机采样
    control_layters = random_sample_path()

    # 模型
    # super_model = AutoModelForSequenceClassification.from_pretrained(base_valid_model)
    adapter_name = base_valid_model.load_adapter(super_adapter_path, leave_out=control_layters, with_head=True)
    base_valid_model.train_adapter("bottleneck_adapter")  #
    base_valid_model.set_active_adapters(adapter_name)

    base_valid_model.to(device)
    base_valid_model.eval()

    return base_valid_model, control_layters

def validate_acc(base_valid_model, super_adapter_path, valid_file_path, device, tokenizer):

    #数据
    valid_dataset = LineByLineTextDataset(file_path=valid_file_path)
    valid_dataLoader = DataLoader(valid_dataset, 32, shuffle=False)

    #评估
    all_correct = 0
    all_num = 0

    # progress_bar_in = tqdm(range(len(valid_dataLoader) * 1))
    super_valid_model, control_layters = get_valid_model(base_valid_model, super_adapter_path, device)

    for text, code, labels in valid_dataLoader:

        label_list = labels.to(device)

        with autocast():
            batch_tokenized = tokenizer(list(text), list(code), add_special_tokens=True,
                                        padding=True, max_length=180,
                                        truncation=True, return_tensors="pt")  # tokenize、add special token、pad
            batch_tokenized = batch_tokenized.to(device)
            outputs = super_valid_model(**batch_tokenized)
            _, predict = torch.max(outputs.logits, 1)

        corret_num = sum((predict == label_list))
        all_num += len(predict)
        all_correct += corret_num

        # progress_bar_in.update(32)

    currnt_accuracy = all_correct / all_num
    # print('准确率为%.8f' % (currnt_accuracy))

    return currnt_accuracy, control_layters



def run_search(search_num, tokenizer_path, base_model_path, best_super_adapter_dir, valid_file_path):
    print("run")

    # 设置GPU运行
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("train_device: ", device)


    ########## MODEL ##############################
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)  # 分词器

    base_valid_model = AutoModelForSequenceClassification.from_pretrained(base_model_path)


    ########valid########

    max_acc = 0
    best_path = []
    for num in range(search_num):
        currnt_accuracy, control_layters = validate_acc(base_valid_model, best_super_adapter_dir, valid_file_path, device, tokenizer)

        print()
        print("currnt acc {}, path {}".format(currnt_accuracy, control_layters))

        #保存最优 path
        if(currnt_accuracy > max_acc):
            max_acc = currnt_accuracy
            best_path = control_layters
            print("max acc {}, best path {}".format(max_acc, best_path))


    return max_acc, best_path


if __name__ == '__main__':
    set_seed(1) #固定随机种子

    lang = "sql" #代码搜索时的编程语言

    #数据
    valid_file_path = "../../data/train_valid/"+ lang +"/valid.txt"

    #模型
    tokenizer_path = "../../../graph_code_bert"
    base_model_path = "initial_base_model_adapter/python_model_hardonce_unified_topk5_sqlvalid_2e5"

    best_super_adapter_dir = "./{}_best_super_adapter".format(lang) #在第一阶段训练的超网adapter保存位置

    search_num = 100 #搜索次数

    max_acc, best_path = run_search(search_num, tokenizer_path, base_model_path, best_super_adapter_dir, valid_file_path)
    print("max acc {}, best path {}".format(max_acc, best_path))

