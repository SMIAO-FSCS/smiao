import random

import torch
import os
import pandas as pd
import sklearn
from transformers import BertModel, BertTokenizer, RobertaConfig, RobertaModelWithHeads, PrefixTuningConfig, \
    AutoModelForSequenceClassification, TextClassificationPipeline, AdapterConfig, ConfigUnion, ParallelConfig, \
    UniPELTConfig, get_linear_schedule_with_warmup
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset, RandomSampler
import pickle
from transformers import AutoTokenizer, AutoModel
#半精度
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from transformers.adapters import Fuse

class TestLineByLineTextDataset(Dataset):
    def __init__(self, file_path: str):
        assert os.path.isfile(file_path)
        print("read data file at:", file_path)

        with open(file_path, encoding="utf-8") as f:
            self.lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        # 截断测试
        # self.lines = self.lines[:1000] #test数据集也会被截断,所以需要处理一下

        self.text_lines = []
        self.code_lines = []
        self.labels = []

        for line in self.lines:
            temp_line = line.split("<CODESPLIT>")
            if (len(temp_line)) == 5:  # 确保<CODESPLIT>分开的每个部分都有值，不是Null
                # if(str(temp_line[0]) == "1"): #1表示代码和注释对应着，0表示每对应
                self.text_lines.append(temp_line[-2].lower()) #注释
                self.code_lines.append(temp_line[-1].lower()) #代码
                self.labels.append(int(temp_line[0]))

        print("TEST注释和代码总行数:", len(self.text_lines), len(self.code_lines))

    def __len__(self):
        return len(self.text_lines)  # 注意这个len本质是数据的数量

    def __getitem__(self, i):
        a = self.text_lines[i]
        b = self.code_lines[i]
        c = self.labels[i]
        return a, b, c

class LineByLineTextDataset(Dataset):
    def __init__(self, file_path: str, init_train_num = 0):
        assert os.path.isfile(file_path)
        print("read data file at:", file_path)

        with open(file_path, encoding="utf-8") as f:
            self.lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        # 截断测试
        if(init_train_num != 0):
            self.lines = self.lines[:init_train_num]

        self.text_lines = []
        self.code_lines = []
        self.labels = []

        for line in self.lines:
            temp_line = line.split("<CODESPLIT>")
            if (len(temp_line)) == 5:  # 确保<CODESPLIT>分开的每个部分都有值，不是Null
                # if(str(temp_line[0]) == "1"): #1表示代码和注释对应着，0表示每对应
                self.text_lines.append(temp_line[-2].lower()) #注释
                self.code_lines.append(temp_line[-1].lower()) #代码
                self.labels.append(int(temp_line[0]))

        print("TRAIN注释和代码总行数:", len(self.text_lines), len(self.code_lines))

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

def init_model_adapter(num_epochs, batch_size, lr, init_adaptor_save_dir, init_basemodel_save_dir, pre_train_model, train_file_path, valid_file_path, init_train_num):
    set_seed(1)
    print("run")
    # 配置
    # num_epochs = 10 #arg1
    # batch_size = 64 #arg2
    # lr = 2e-5
    #根据您的数据集大小，您可能还需要比平时更长的训练时间。 为避免过度拟合，您可以在开发集上的每个 epoch 之后评估适配器，并仅保存最佳模型。

    #train
    # init_adaptor_save_dir = "java_adapter_solidityvalid_10w_codebert_2e5"  #arg3
    # init_basemodel_save_dir = "java_model_solidityvalid_10w_codebert_2e5"

    # pre_train_model = "../../../CODEBERT"

    # train_file_path = "../../../data/train_valid/java/hardonce_topk5codebert.txt"  #

    #valid
    valid_batch_size = 1
    # valid_file_path = "../../../data/train_valid/solidity/valid.txt"  # arg1

    ########################## 数据 #########################
    # file_path = "../look"
    train_dataset = LineByLineTextDataset(file_path=train_file_path, init_train_num=init_train_num)
    train_dataLoader = DataLoader(train_dataset, batch_size, shuffle=True)

    valid_dataset = TestLineByLineTextDataset(file_path=valid_file_path)
    valid_dataLoader = DataLoader(valid_dataset, valid_batch_size, shuffle=False)

    # device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("train_device: ", device)


    ########## MODEL ##############################
    set_seed(1)
    model = AutoModelForSequenceClassification.from_pretrained(pre_train_model) #AutoAdapterModel.from_pretrained   ---> model.add_classification_head
    tokenizer = AutoTokenizer.from_pretrained(pre_train_model) #换上solidity_bpe_tokenizer，降低幅度很大，为什么呢？

    #bottleneck_adapter
    config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu") #16,5epoch: 0.641, 1epoch:0.624 #8:0.613 #
    model.add_adapter("bottleneck_adapter", config=config) #80

    # prefix_tuning
    # config = PrefixTuningConfig(flat=False, prefix_length=30, leave_out=[1])  # 30：0.601
    # model.add_adapter("bottleneck_adapter", config=config)  # 只加了5层

    # mam_adapter
    # config = ConfigUnion(
    #     PrefixTuningConfig(bottleneck_size=512),
    #     ParallelConfig(),
    # )
    # model.add_adapter("bottleneck_adapter", config=config)

    # model.train_adapter("bottleneck_adapter") # disables training of all weights outside the task adapter
    model.set_active_adapters("bottleneck_adapter") #train的时候要不要加呢？inference时是加了

    # 看有哪些参数一下
    # for name, param in model.named_parameters():
    #     print(name, param.size())

    print("*" * 30)
    print('\n')

    # 验证一下
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.size())

    # 过滤掉requires_grad = False的参数
    # adaptor是随机初始化参数的，所以lr一般比较大，常见的是1e-4,那能不能弄出个好的初始化参数呢
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
    max_accuracy = 0
    model.train()
    for epoch in range(num_epochs):
        epoch_all_loss = 0
        for text, code, labels in train_dataLoader:
            model.train()
            targets = labels.to(device)

            with autocast():
                batch_tokenized = tokenizer(list(text), list(code), add_special_tokens=True,
                                                 padding=True, max_length=128,
                                                 truncation=True, return_tensors="pt")  # tokenize、add special token、pad
                batch_tokenized = batch_tokenized.to(device)

                outputs = model(**batch_tokenized,output_hidden_states=True)

                loss = lossfuction(outputs.logits, targets)

            # 修改为半精度
            scaler.scale(loss).backward()
            epoch_all_loss += loss.item()

            scaler.step(optimizer)
            scaler.update()
            scheduler.step() # not use scheduler
            optimizer.zero_grad()

            ############### old ###########################
            # loss.backward()
            # optimizer.step()
            # epoch_all_loss += loss.item()
            ##########################################

            if(tag % 200 == 0):
                print("batch: %d, loss: %.8f" % (tag + 1, float(loss.item())))
            tag += 1

            progress_bar_in.update(1)

        # every 5 epoch print loss
        # if ((epoch + 1) % 1 == 0):
        print("PRE5 - epoch: %d, loss: %.8f" % (epoch + 1, epoch_all_loss / len(train_dataLoader)))

        ########valid########
        # 每个epoch都验证一下
        model.eval()
        all_correct = 0
        all_num = 0
        for text, code, labels in valid_dataLoader:
            label_list = labels.to(device)
            with autocast():
                batch_tokenized = tokenizer(list(text), list(code), add_special_tokens=True,
                                            padding=True, max_length=128,
                                            truncation=True, return_tensors="pt")  # tokenize、add special token、pad
                batch_tokenized = batch_tokenized.to(device)

                outputs = model(**batch_tokenized)

            _, predict = torch.max(outputs.logits, 1)

            corret_num = sum((predict == label_list))
            all_num += len(predict)
            all_correct += corret_num
        print("epoch: ",epoch)
        currnt_accuracy = all_correct / all_num
        print('currnt 准确率为%.8f, max 准确率为%.8f' % (currnt_accuracy, max_accuracy))

        #这里直接用mrr保存模型岂不更好？不过那个时间长一些，而且显存可能不够。
        if(currnt_accuracy > max_accuracy):
            # torch.save(model, model_save_dir) #这样就更换了，问题不大
            model.save_pretrained(init_basemodel_save_dir)
            model.save_adapter(init_adaptor_save_dir, "bottleneck_adapter")
            max_accuracy = currnt_accuracy
            print('max 准确率为%.8f' % (max_accuracy))


        #######SAVE MODEL####
        # torch.save(model, model_save_dir+"last")  # 这样就更换了，问题不大
        # model.save_adapter(init_adaptor_save_dir+"last", "bottleneck_adapter")

    # save last
    # model.save_pretrained(init_basemodel_save_dir + "lastepoch")
    # model.save_adapter(init_adaptor_save_dir + "lastepoch", "bottleneck_adapter")

    # deactivate all adapters
    model.set_active_adapters(None)
    # delete the added adapter
    model.delete_adapter('bottleneck_adapter')

#86?
if __name__ == '__main__':
    set_seed(1)
    num_epochs = 5  # arg1
    batch_size = 64  # arg2
    lr = 2e-5
    init_train_num = 200000

    # train
    init_adaptor_save_dir = "java_adapter_sqlvalid"  # arg3
    init_basemodel_save_dir = "java_model_sqlvalid"

    pre_train_model = "../../../graph_code_bert"

    train_file_path = "../../../data/train_valid/java/train.txt"  #

    # valid
    valid_file_path = "../../../data/train_valid/sql/valid.txt"  # arg1


    init_model_adapter(num_epochs, batch_size, lr, init_adaptor_save_dir, init_basemodel_save_dir, pre_train_model, train_file_path, valid_file_path, init_train_num)