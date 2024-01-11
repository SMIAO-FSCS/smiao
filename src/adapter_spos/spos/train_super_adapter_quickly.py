import random
import shutil

import torch
import os
from transformers import AutoModelForSequenceClassification, AdapterConfig, get_linear_schedule_with_warmup
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

from spos_adaptor_inference import mrr_inference, pure_mrr_inference
from spos_mrr import get_mrr


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
    def __init__(self, file_path: str, train_num=0):
        assert os.path.isfile(file_path)
        print("read data file at:", file_path)

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

        print(self.text_lines[0])
        print(self.code_lines[0])


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

#随机采样路径
def random_sample_path():
    control_layters = []
    choice_list = np.random.rand(12).round().astype(np.uint8) #共11层，选择的值是0/1，0代表不插入该层，1代表插入

    for index, value in enumerate(choice_list):
        if (value == 0):  # 如果value是0，则表示移除
            control_layters.append(index)

    print("control_layters{}, choice_list{} ".format(control_layters, choice_list))

    return control_layters

#创建子网,复制权重方法
def create_sub_net(base_model_for_train, adapter_path):
    #移除掉已经挂载到基础模型的adapter，以挂载新的路径，否则还是之前的路径
    base_model_for_train.delete_adapter('bottleneck_adapter')

    #先随机采样一个路径
    control_layters = random_sample_path()

    #创建子网，并贡献权重
    # model = AutoModelForSequenceClassification.from_pretrained(base_model_path)
    base_model_for_train.load_adapter(adapter_name_or_path=adapter_path, load_as="bottleneck_adapter", leave_out=control_layters, with_head=False)

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
    super_model.load_adapter(adapter_name_or_path=init_adapter_path, load_as="bottleneck_adapter", with_head=False) #init_adapter_path没有减少层次，当作超网

    #将子网权重，复制给超网，(相当于复制对应的adapter参数)
    for sub_name, sub_param in sub_model.named_parameters():
        if "adapters" in sub_name:
            for super_name, super_param in super_model.named_parameters():
                if sub_name == super_name and "adapters" in super_name:
                    super_param.data.copy_(sub_param.data)

    #保存超网（相当于保存新的adapter）
    super_model.save_adapter(new_super_adaptor_save_dir, "bottleneck_adapter")


#创建子网，通过disable掉要去掉层的梯度，实现在反向传播时，不更新非采样的子网的参数
#但是，这种做法，会导致 即使是本来要remove的层，还依旧在模型中参与forword计算，只是没更新。
#看SPOS的实现的代码，forword时后，不需要的层不参与forword

#由于在训练超网时，只训练，而不验证，所以这种做法不是不行。在搜索子网过程中，再把需要去除的去掉。
#有点别扭，因为训练的时候，相当于整个网参与计算，但是搜索时候，就被破除了。和SPOS在这里不一致，只是参数更新时一致。
#不过，由于搜索和训练是分开的，所以，搜索出的是去层的架构，然后retrain，性能应该还是ok的
def create_sub_net_2(supernet_path):
    #不用分析太多，实验的效果说了算

    #采样路径
    control_layters = random_sample_path()

    #disable去除层的权重，

    #去训练子网




def run(train_num, lr, lang, adaptor_save_dir, train_file_path, infer_file_path, output_infer_file, control_layters=[]):
    set_seed(1) #固定随机种子
    print("run")
    # 配置
    num_epochs = 100 #arg1 #训练的epoch
    batch_size = 64 #arg2 #batch_size

    ########################## 数据 #########################
    # 读取数据
    train_dataset = LineByLineTextDataset(file_path=train_file_path, train_num=train_num)
    train_dataLoader = DataLoader(train_dataset, batch_size, shuffle=True)

    # 设置GPU运行
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("train_device: ", device)


    ########## MODEL ##############################
    #加载MODEL
    # model = AutoModelForSequenceClassification.from_pretrained("../../java_fine_turn_GraBertlastepoch_lookLoss_GraBert")
    tokenizer = AutoTokenizer.from_pretrained("../../../graph_code_bert") #分词器

    base_model_path = "../../../java_fine_turn_GraBertlastepoch_lookLoss_GraBert"
    adapter_path = "./python_adaptor_onceAllFile_2e4" #初始的adapter路径
    super_adaptor_save_dir = adapter_path + "new" #用于将子网权重复制给超网，中转的adapter

    base_model_for_train = AutoModelForSequenceClassification.from_pretrained(base_model_path)
    base_model_for_surernet = AutoModelForSequenceClassification.from_pretrained(base_model_path)

    ############################# Adapter ############################


    print("*" * 30)
    print('\n')

    lossfuction = nn.CrossEntropyLoss()

    ######################### train #########################
    scaler = GradScaler()
    progress_bar_in = tqdm(range(len(train_dataLoader) * num_epochs))
    # model.train()
    max_mrr = 0
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

                print("loss ", loss.item())

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

        ########valid########
        # 每个epoch都验证一下
        model.eval()

        pure_mrr_inference(model, infer_file_path, output_infer_file)
        current_mrr = get_mrr(lang)

        print("epoch: ",epoch)
        print('currnt mrr %.8f, max mrr %.8f' % (current_mrr, max_mrr))

        #保存最优
        # if(current_mrr > max_mrr):
        #     # torch.save(model, model_save_dir) #这样就更换了，问题不大
        #     model.save_adapter(adaptor_save_dir, "bottleneck_adapter") #如果目前的model好于max_mrr，那么就存为best。  如果后续没超过，则不更新best，否则更新
        #     max_mrr = current_mrr
        #     print('max mrr %.8f' % (max_mrr))

        # 保存最优
        if (current_mrr > max_mrr):
            # 保存最佳超网的权重，不是子网
            # 直接通过复制当前最佳super adapter的路径即可
            save_best_super_adapter_dir = "./{}_best_super_adapter".format(lang)
            if (os.path.exists(save_best_super_adapter_dir)):
                shutil.rmtree(save_best_super_adapter_dir)
            shutil.copytree(src=super_adaptor_save_dir, dst=save_best_super_adapter_dir)
            # model.save_adapter(adaptor_save_dir, "bottleneck_adapter") #如果目前的model好于max_mrr，那么就存为best。  如果后续没超过，则不更新best，否则更新

            max_mrr = current_mrr
            print('max mrr %.8f' % (max_mrr))


    # deactivate all adapters
    # model.set_active_adapters(None)
    # # delete the added adapter
    # model.delete_adapter('bottleneck_adapter')

#86?
if __name__ == '__main__':
    lang = "sql" #代码搜索时的编程语言


    #配置
    adaptor_save_dir = "../../save_model/"+ lang +"/" + str(lang) + "_adaptor" #adapter的保存位置
    train_file_path = "../../data/train_valid/"+ lang +"/train.txt"  # 训练文件目录

    infer_file_path = "../../data/test/"+ lang +"/batch_0.txt" # 推理文件目录
    output_infer_file = "../../results/"+ lang +"/adaptor_batch_0.txt"  # 推理结果目录

    for train_num in [500]: #训练数据量
        # for lr in [6e-4, 2e-4, 5e-5, 1e-5]: #lr
        for lr in [6e-4]:
            #train
            run(train_num, lr, lang, adaptor_save_dir, train_file_path, infer_file_path, output_infer_file)

            #inference
            # mrr_inference(adaptor_save_dir, infer_file_path, output_infer_file, 0)

            #获取结果
            get_mrr(lang)

            print("lr {}, train_num {}".format(lr, train_num))