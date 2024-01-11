import random

import torch
import os
from transformers import AutoModelForSequenceClassification, AdapterConfig, get_linear_schedule_with_warmup
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
#半精度
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

from util import remove_conflicting_zero_pairs, balance_samples_with_distinct_zero_pairs
from .spos_adaptor_inference import mrr_inference, pure_mrr_inference
from .spos_mrr import get_mrr


class LineByLineTextDataset(Dataset):
    def __init__(self, file_path: str, train_num=0, isTrain=False):
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


        print("TRAIN注释和代码总行数:", len(self.text_lines), len(self.code_lines))
        if (isTrain):
            # 去除冲突标签 #平衡类别
            self.text_lines, self.code_lines, self.labels = remove_conflicting_zero_pairs(self.text_lines,
                                                                                          self.code_lines, self.labels)
            print("去除冲突标签后的行数:", len(self.text_lines), len(self.code_lines))
            self.text_lines, self.code_lines, self.labels = balance_samples_with_distinct_zero_pairs(self.text_lines,
                                                                                                     self.code_lines,
                                                                                                     self.labels)
            print("平衡类别 label 1 num {}, label 0 num {}".format(sum(self.labels), len(self.labels) - sum(self.labels)))

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

def run_retrain(train_num, lr, lang, adaptor_save_dir, train_file_path, infer_file_path, output_infer_file, best_path, best_super_adapter_dir,
                re_num_epochs, re_batch_size, tokenizer_path, base_model_path):
    print("run")
    print("init_adapter_path: ", best_super_adapter_dir)
    # 配置
    num_epochs = re_num_epochs #arg1 #训练的epoch
    batch_size = re_batch_size #arg2 #batch_size

    ########################## 数据 #########################
    # 读取数据
    train_dataset = LineByLineTextDataset(file_path=train_file_path, train_num=train_num, isTrain=True)
    train_dataLoader = DataLoader(train_dataset, batch_size, shuffle=True)

    # 设置GPU运行
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("train_device: ", device)


    ########## MODEL ##############################
    #加载MODEL
    model = AutoModelForSequenceClassification.from_pretrained(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) #分词器

    ############################# Adapter ############################
    model.delete_adapter('bottleneck_adapter')
    model.load_adapter(adapter_name_or_path=best_super_adapter_dir, load_as="bottleneck_adapter", leave_out=best_path, with_head=True)

    #下面两行，实现了固定预训练模型的参数，在训练时只训练adapter的参数
    model.train_adapter("bottleneck_adapter") #
    model.set_active_adapters("bottleneck_adapter") #

    # 看模型的总参数结构
    # for name, param in model.named_parameters():
    #     print(name, param.size())

    print("*" * 30)
    print('\n')

    # 看哪些参数参与训练
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.size())

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    total_steps = len(train_dataLoader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0.05 * total_steps,
                                                num_training_steps=total_steps)

    model.to(device)
    lossfuction = nn.CrossEntropyLoss()

    ######################### train #########################
    scaler = GradScaler()
    # progress_bar_out = tqdm(range(num_epochs))
    progress_bar_in = tqdm(range(len(train_dataLoader) * num_epochs))
    tag = 0
    model.train()
    max_mrr = 0
    for epoch in range(num_epochs):
        epoch_all_loss = 0
        for text, code, labels in train_dataLoader:
            model.train()
            targets = labels.to(device)

            with autocast():
                batch_tokenized = tokenizer(list(text), list(code), add_special_tokens=True,
                                                 padding=True, max_length=180,
                                                 truncation=True, return_tensors="pt")  # tokenize、add special token、pad
                batch_tokenized = batch_tokenized.to(device)

                outputs = model(**batch_tokenized,output_hidden_states=True)

                loss = lossfuction(outputs.logits, targets)

            # 修改为半精度
            scaler.scale(loss).backward()
            epoch_all_loss += loss.item()

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            progress_bar_in.update(1)

        print("PRE5 - epoch: %d, loss: %.8f" % (epoch + 1, epoch_all_loss / len(train_dataLoader)))

        ########valid########
        # 每个epoch都验证一下
        model.eval()
        pure_mrr_inference(tokenizer_path, model, infer_file_path, output_infer_file)
        current_mrr = get_mrr(lang)

        print("epoch: ",epoch)
        print('currnt mrr %.8f, max mrr %.8f' % (current_mrr, max_mrr))

        #保存最优
        if(current_mrr > max_mrr):
            # torch.save(model, model_save_dir) #这样就更换了，问题不大
            model.save_adapter(adaptor_save_dir, "bottleneck_adapter") #如果目前的model好于max_mrr，那么就存为best。  如果后续没超过，则不更新best，否则更新
            max_mrr = current_mrr
            print('max mrr %.8f' % (max_mrr))


    # deactivate all adapters
    model.set_active_adapters(None)
    # delete the added adapter
    model.delete_adapter('bottleneck_adapter')

#86?
if __name__ == '__main__':
    set_seed(1) #固定随机种子

    #配置
    lang = "sql" #代码搜索时的编程语言
    re_num_epochs = 10
    re_batch_size = 64

    #数据
    adaptor_save_dir = "../../save_model/"+ lang +"/" + str(lang) + "_adaptor" #adapter的保存位置
    train_file_path = "../../data/train_valid/"+ lang +"/train.txt"  # 训练文件目录

    infer_file_path = "../../data/test/"+ lang +"/batch_0.txt" # 推理文件目录
    output_infer_file = "../../results/"+ lang +"/adaptor_batch_0.txt"  # 推理结果目录

    #模型
    # init_adapter_path = "./{}_best_super_adapter".format(lang)
    best_super_adapter_dir = "../initial_base_model_adapter/php_adapter_sqlvalid_0.75hardgraph_code_bert_2e-05"

    tokenizer_path = "../../../graph_code_bert"
    base_model_path = "../initial_base_model_adapter/php_model_sqlvalid_0.75hardgraph_code_bert_2e-05"

    best_path = [1, 3, 5, 7, 8, 9, 11]

    for train_num in [14000]: #训练数据量
        for lr in [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]: #lr
        # for lr in [6e-4, 5e-4, 4e-4, 3e-4]:
            #train
            run_retrain(train_num, lr, lang, adaptor_save_dir, train_file_path, infer_file_path, output_infer_file,
                        best_path, best_super_adapter_dir,
                        re_num_epochs, re_batch_size, tokenizer_path, base_model_path)

            #inference
            mrr_inference(base_model_path, tokenizer_path, adaptor_save_dir, infer_file_path, output_infer_file, 0)

            #获取结果
            get_mrr(lang)

            print("lr {}, train_num {}".format(lr, train_num))
