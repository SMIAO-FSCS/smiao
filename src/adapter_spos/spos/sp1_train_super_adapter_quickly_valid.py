import random
import shutil

import torch
import os
from transformers import AutoModelForSequenceClassification
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from transformers import logging

logging.set_verbosity_error()

#半精度
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

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


        print("注释和代码总行数:", len(self.text_lines), len(self.code_lines))

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

#创建子网,复制权重方法
def create_sub_net(base_model_for_train, adapter_path):
    #移除掉已经挂载到基础模型的adapter，以挂载新的路径，否则还是之前的路径
    base_model_for_train.delete_adapter('bottleneck_adapter')

    #先随机采样一个路径
    control_layters = random_sample_path()

    #创建子网，并贡献权重
    # model = AutoModelForSequenceClassification.from_pretrained(base_model_path)
    base_model_for_train.load_adapter(adapter_name_or_path=adapter_path, load_as="bottleneck_adapter", leave_out=control_layters, with_head=True)

    # 下面两行，实现了固定预训练模型的参数，在训练时只训练adapter的参数
    base_model_for_train.train_adapter("bottleneck_adapter")  #
    base_model_for_train.set_active_adapters("bottleneck_adapter")  #

    #返回sub_net，sub_net用于去训练
    return base_model_for_train

    #sub_net训练完成后，将自身的权重，再复制给super_net对应的位置。

#将子网的权重，复制给超网
def sub_weight_2_super(sub_model, super_model, init_adapter_path, new_super_adaptor_save_dir):
    super_model.delete_adapter('bottleneck_adapter')

    #加载超网
    # super_model = AutoModelForSequenceClassification.from_pretrained(base_model_path)
    super_model.load_adapter(adapter_name_or_path=init_adapter_path, load_as="bottleneck_adapter", with_head=True) #init_adapter_path没有减少层次，当作超网

    #将子网权重，复制给超网，(相当于复制对应的adapter参数)
    for sub_name, sub_param in sub_model.named_parameters():
        if "adapters" in sub_name:
            for super_name, super_param in super_model.named_parameters():
                if sub_name == super_name and "adapters" in super_name:
                    super_param.data.copy_(sub_param.data)

    #保存超网（相当于保存新的adapter）
    super_model.save_adapter(new_super_adaptor_save_dir, "bottleneck_adapter")

def get_valid_model(base_valid_model, super_adapter_path, device):
    base_valid_model.delete_adapter('bottleneck_adapter')

    # 随机采样
    control_layters = random_sample_path()

    # 模型
    adapter_name = base_valid_model.load_adapter(super_adapter_path, leave_out=control_layters, with_head=True)
    base_valid_model.train_adapter("bottleneck_adapter")  #
    base_valid_model.set_active_adapters(adapter_name)

    base_valid_model.to(device)
    base_valid_model.eval()

    return base_valid_model

def validate(base_valid_model, super_adapter_path, valid_file_path, device, tokenizer):

    #数据
    valid_dataset = LineByLineTextDataset(file_path=valid_file_path)
    valid_dataLoader = DataLoader(valid_dataset, 32, shuffle=False)

    #评估
    all_correct = 0
    all_num = 0

    # progress_bar_in = tqdm(range(len(valid_dataLoader) * 1))
    for text, code, labels in valid_dataLoader:

        super_valid_model = get_valid_model(base_valid_model, super_adapter_path, device)

        label_list = labels.to(device)

        with autocast():
            batch_tokenized = tokenizer(list(text), list(code), add_special_tokens=True,
                                        padding=True, max_length=180,
                                        truncation=True, return_tensors="pt")  # tokenize、add special token、pad
            batch_tokenized = batch_tokenized.to(device)

            outputs = super_valid_model(**batch_tokenized)

        _, predict = torch.max(outputs.logits, 1)

        corret_num = sum((predict == label_list))
        # print('corret_num %.8f' % (corret_num))


        all_num += len(predict)
        all_correct += corret_num

        # progress_bar_in.update(32)

    currnt_accuracy = all_correct / all_num
    print('准确率为%.8f' % (currnt_accuracy))

    return currnt_accuracy



def run_train_supernet(train_num, lr, train_file_path, valid_file_path, num_epochs, batch_size,
                       tokenizer_path, base_model_path, init_adapter_path, best_super_adapter_dir):
    set_seed(1)
    print("run")

    ########################## 数据 #########################
    # 读取数据
    train_dataset = LineByLineTextDataset(file_path=train_file_path, train_num=train_num, isTrain=True)
    train_dataLoader = DataLoader(train_dataset, batch_size, shuffle=True)

    # 设置GPU运行
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("train_device: ", device)


    ########## MODEL ##############################
    #加载MODEL
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) #分词器

    adapter_path = init_adapter_path #初始的adapter路径
    super_adaptor_save_dir = adapter_path + "new" #用于将子网权重复制给超网，中转的adapter   超网保存目录，但并非最佳超网保存目录

    base_model_for_train = AutoModelForSequenceClassification.from_pretrained(base_model_path)
    base_model_for_surernet = AutoModelForSequenceClassification.from_pretrained(base_model_path)
    base_valid_model = AutoModelForSequenceClassification.from_pretrained(base_model_path)

    ############################# Adapter ############################


    print("*" * 30)
    print('\n')

    lossfuction = nn.CrossEntropyLoss()

    ######################### train #########################
    scaler = GradScaler()
    progress_bar_in = tqdm(range(len(train_dataLoader) * num_epochs))
    max_mrr = 0
    max_acc = 0
    for epoch in range(num_epochs):
        epoch_all_loss = 0
        for text, code, labels in train_dataLoader:
            #创建随机采样路径的子网
            model = create_sub_net(base_model_for_train, adapter_path)

            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
            model.to(device)
            model.train()

            targets = labels.to(device)

            with autocast():
                batch_tokenized = tokenizer(list(text), list(code), add_special_tokens=True,
                                                 padding=True, max_length=180,
                                                 truncation=True, return_tensors="pt")  # tokenize、add special token、pad
                batch_tokenized = batch_tokenized.to(device)

                outputs = model(**batch_tokenized,output_hidden_states=True)

                loss = lossfuction(outputs.logits, targets)

                # print("loss ", loss.item())

            # 修改为半精度
            scaler.scale(loss).backward()
            epoch_all_loss += loss.item()

            scaler.step(optimizer)
            scaler.update()
            # scheduler.step()
            optimizer.zero_grad()

            #更新完成后，便需要将子网adapter的参数，复制给超网
            # super_adaptor_save_dir = adapter_path + "new"
            sub_weight_2_super(sub_model = model, super_model= base_model_for_surernet, init_adapter_path = adapter_path, new_super_adaptor_save_dir = super_adaptor_save_dir)
            adapter_path = super_adaptor_save_dir #下一次选择path，则用的是上一次更新的权重，实现权重共享

            progress_bar_in.update(1)

        print("PRE5 - epoch: %d, loss: %.8f" % (epoch + 1, epoch_all_loss / len(train_dataLoader)))


        ###valid ACC ###
        current_acc = validate(base_valid_model, super_adaptor_save_dir, valid_file_path, device, tokenizer)
        print()
        print('epoch %d, currnt acc %.8f, max acc %.8f' % (epoch, current_acc, max_acc))
        print()
        #保存最优
        if(current_acc > max_acc):
            # 保存最佳超网的权重，不是子网
            # 直接通过复制当前最佳super adapter的路径即可
            # init_adapter_path = "./{}_best_super_adapter".format(lang)
            if(os.path.exists(best_super_adapter_dir)):
                shutil.rmtree(best_super_adapter_dir)
            shutil.copytree(src = super_adaptor_save_dir, dst = best_super_adapter_dir)

            max_acc = current_acc



if __name__ == '__main__':
    set_seed(1) #固定随机种子

    #配置
    lang = "sql"
    num_epochs = 20
    batch_size = 64

    #数据
    train_file_path = "../../data/train_valid/"+ lang +"/train.txt"  # 训练文件目录
    valid_file_path = "../../data/train_valid/"+ lang +"/valid.txt" #valid文件目录
    #模型
    tokenizer_path = "../../../graph_code_bert"  #分词器
    base_model_path = "../../../java_fine_turn_GraBertlastepoch_lookLoss_GraBert"  #基础模型
    init_adapter_path = "python_adaptor_onceAllFile_2e4" #初始的adapter

    #保存
    best_super_adapter_dir = "./{}_best_super_adapter".format(lang) #保存最佳性能超网的权重


    for train_num in [8000]: #训练数据量
        # for lr in [6e-4, 2e-4, 5e-5, 1e-5]: #lr
        for lr in [6e-4]:
            #train
            run_train_supernet(train_num, lr, train_file_path, valid_file_path, num_epochs, batch_size,
                               tokenizer_path, base_model_path, init_adapter_path, best_super_adapter_dir)
