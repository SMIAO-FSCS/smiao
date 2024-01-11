import pprint
import random

import faiss
import torch
import os

from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from transformers import AutoModelForSequenceClassification, AdapterConfig, get_linear_schedule_with_warmup, AutoModel
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
#半精度
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

from spos.spos_adaptor_inference import pure_mrr_inference, mrr_inference
from spos.spos_mrr import get_mrr


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


class ArrayDataset(Dataset):
    def __init__(self, queries, codes, lables):

        self.text_lines = queries
        self.code_lines = codes
        self.labels = lables

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

def validate_acc(tokenizer, model, valid_file_path, device):

    #数据
    valid_dataset = LineByLineTextDataset(file_path=valid_file_path)
    valid_dataLoader = DataLoader(valid_dataset, 32, shuffle=False)

    #评估
    all_correct = 0
    all_num = 0

    for text, code, labels in valid_dataLoader:

        label_list = labels.to(device)

        with autocast():
            batch_tokenized = tokenizer(list(text), list(code), add_special_tokens=True,
                                        padding=True, max_length=180,
                                        truncation=True, return_tensors="pt")  # tokenize、add special token、pad
            batch_tokenized = batch_tokenized.to(device)
            outputs = model(**batch_tokenized)
            _, predict = torch.max(outputs.logits, 1)

        corret_num = sum((predict == label_list))
        all_num += len(predict)
        all_correct += corret_num

        # progress_bar_in.update(32)

    currnt_accuracy = all_correct / all_num
    # print('准确率为%.8f' % (currnt_accuracy))

    return currnt_accuracy

def read_file(file_path, train_num=0, str_label=""):
    with open(file_path, encoding="utf-8") as f:
        lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    # 截断测试
    if (train_num != 0):
        lines = lines[:train_num]

    text_lines = []
    code_lines = []
    labels = []

    for line in lines:
        temp_line = line.split("<CODESPLIT>")
        if (len(temp_line)) == 5:  # 确保<CODESPLIT>分开的每个部分都有值，不是Null
            text_lines.append(temp_line[-2])  # 注释
            code_lines.append(temp_line[-1])  # 代码
            labels.append(int(temp_line[0]))


    return np.array(code_lines), np.array(text_lines), np.array(labels)

def kmeans_sample_data(codes, queries, labels, num_clusters=100):
    codes = list(codes)
    queries = list(queries)
    labels = list(labels)
    """
    对给定的代码进行聚类，并从每个聚类中采样对应的查询。
    """
    # 确保CUDA可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载RoBERTa模型和分词器
    tokenizer_path = "../../graph_code_bert"  # 分词器
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModel.from_pretrained(tokenizer_path, return_dict=True).to(device)
    model.eval()

    def extract_features_in_batches(codes, batch_size=32):
        """ 分批次提取代码特征 """
        all_features = []
        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i:i + batch_size]
            inputs = tokenizer(batch_codes, padding=True, truncation=True, return_tensors="pt", max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_features.append(features)
        return np.vstack(all_features)

    # 提取特征
    features = extract_features_in_batches(codes)

    # 使用KMeans进行聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)
    centroids = kmeans.cluster_centers_

    # 找出每个类别的质心点对应的最近数据
    sampled_indices = []
    for centroid in centroids:
        distances = np.linalg.norm(features - centroid, axis=1)
        sampled_indices.append(np.argmin(distances))

    # 从原始数据中获取采样结果
    sampled_codes = [codes[i] for i in sampled_indices]
    sampled_queries = [queries[i] for i in sampled_indices]
    sampled_labels = [labels[i] for i in sampled_indices]

    return np.array(sampled_codes), np.array(sampled_queries), np.array(sampled_labels)


def model_train(model, device, scaler, optimizer, lossfuction, tokenizer, batch_size, train_queries, train_codes, train_lables):
    model.train()

    train_dataset = ArrayDataset(queries=train_queries, codes=train_codes, lables=train_lables)
    train_dataLoader = DataLoader(train_dataset, batch_size, shuffle=True)

    epoch_all_loss = 0
    for text, code, labels in train_dataLoader:
        targets = labels.to(device)

        with autocast():
            batch_tokenized = tokenizer(list(text), list(code), add_special_tokens=True,
                                        padding=True, max_length=180,
                                        truncation=True, return_tensors="pt")  # tokenize、add special token、pad
            batch_tokenized = batch_tokenized.to(device)

            outputs = model(**batch_tokenized, output_hidden_states=True)

            loss = lossfuction(outputs.logits, targets)

        # 修改为半精度
        scaler.scale(loss).backward()
        epoch_all_loss += loss.item()

        scaler.step(optimizer)
        scaler.update()
        # scheduler.step()
        optimizer.zero_grad()

    print("loss: %.8f" % (epoch_all_loss / len(train_dataLoader)))

def model_predict(tokenizer, model, test_queries, test_codes, test_lables, device):
    model.eval()

    #数据
    test_dataset = ArrayDataset(queries=test_queries, codes=test_codes, lables=test_lables)
    test_dataLoader = DataLoader(test_dataset, 32, shuffle=True)

    #评估
    all_predictions = []
    for text, code, labels in test_dataLoader:

        label_list = labels.to(device)

        with autocast():
            batch_tokenized = tokenizer(list(text), list(code), add_special_tokens=True,
                                        padding=True, max_length=180,
                                        truncation=True, return_tensors="pt")  # tokenize、add special token、pad
            batch_tokenized = batch_tokenized.to(device)
            outputs = model(**batch_tokenized)
            _, predict = torch.max(outputs.logits, 1)
            all_predictions.extend(predict.cpu().numpy())

    return all_predictions

def find_most_similar_codes(code_sources, code_targets, queries, labels):
    # 加载RoBERTa模型和分词器
    tokenizer = AutoTokenizer.from_pretrained("../../graph_code_bert")
    roberta_model = AutoModel.from_pretrained("../../graph_code_bert").to('cuda')

    def encode_texts(texts, batch_size=32):
        """将文本批量转换为特征向量，并在GPU上运行"""
        roberta_model.eval()
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(
                'cuda')
            with torch.no_grad():
                outputs = roberta_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                all_embeddings.append(embeddings)

        # 在将所有特征向量合并为一个张量之前，先将它们移至CPU
        all_embeddings = [embedding.cpu() for embedding in all_embeddings]
        return torch.cat(all_embeddings, dim=0).numpy()

    # 将文本转换为特征向量
    source_vectors = encode_texts(list(code_sources))
    target_vectors = encode_texts(list(code_targets))

    # 使用FAISS进行相似性搜索（使用GPU）
    d = target_vectors.shape[1]  # 特征维度
    res = faiss.StandardGpuResources()  # 声明一个GPU资源对象
    index_flat = faiss.IndexFlatL2(d)
    index = faiss.index_cpu_to_gpu(res, 0, index_flat)  # 将索引转移到GPU上
    index.add(target_vectors)

    similar_queries = []
    similar_codes = []
    similar_labels = []
    k = 100  # 查找最近的一个邻居
    for code_vector, code_i in zip(source_vectors, code_sources):
        distances, indices = index.search(np.expand_dims(code_vector, axis=0), k)
        sim_index = 2
        closest_idx = indices[0][sim_index] # 不能拿到自身
        similar_code = code_targets[closest_idx]
        while(True): #避免拿到自己
            if(similar_code == code_i):
                sim_index += 1
                closest_idx = indices[0][sim_index]
                similar_code = code_targets[closest_idx]
            else:
                break

            if(sim_index == 90):
                break

        query = queries[closest_idx]
        label = labels[closest_idx]

        similar_queries.append(query)
        similar_codes.append(similar_code)
        similar_labels.append(label)

    return similar_codes, similar_queries, similar_labels

def dedupe(current_codes, all_codes, all_queries, all_labels):
    # 创建一个集合来存储已经在 current_codes 中出现过的元素
    current_codes_set = set(current_codes)

    # 初始化新的 all_queries、all_labels 和 new_all_codes 列表
    new_all_queries = []
    new_all_labels = []
    new_all_codes = []

    # 遍历 all_codes，检查是否已经在 current_codes_set 中出现过
    for i in range(len(all_codes)):
        if all_codes[i] not in current_codes_set:
            new_all_queries.append(all_queries[i])
            new_all_codes.append(all_codes[i])
            new_all_labels.append(all_labels[i])

            # print(all_queries[i], all_codes[i], all_labels[i])

    return new_all_codes, new_all_queries, new_all_labels

def run_retrain_active(train_num, lr, lang, adaptor_save_dir, train_file_path, infer_file_path, output_infer_file, best_path, best_super_adapter_dir,
                re_num_epochs, re_batch_size, tokenizer_path, base_model_path):

    # 配置
    # num_epochs = re_num_epochs #arg1 #训练的epoch
    # batch_size = re_batch_size #arg2 #batch_size

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
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.size())

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # total_steps = len(train_dataLoader) * num_epochs
    # scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0.05 * total_steps,
    #                                             num_training_steps=total_steps)

    model.to(device)
    model.train()
    lossfuction = nn.CrossEntropyLoss()

    # 读取数据
    current_codes, current_queries, current_labels = read_file(train_file_path, train_num=int(train_num/5)) #初始是标记的数量
    all_codes, all_queries, all_labels = read_file(train_file_path, train_num=0)
    # kmeans 多样性采样
    #current_codes, current_queries, current_labels = kmeans_sample_data(current_codes, current_queries, current_labels)
    print("kmeans label 1: {}, label 0 {}".format(sum(current_labels), len(current_labels) - sum(current_labels)))



    scaler = GradScaler()
    batch_size = re_batch_size
    max_acc = 0
    max_mrr = 0
    #TODO 在到了最大数据量后，epoch还怎么弄？主要是怕过拟合，上mrr？
    for epoch in range(re_num_epochs):
        # k折交叉验证方式的训练
        kf = KFold(n_splits=2)

        all_misclassified_samples = []
        for train_index, test_index in kf.split(current_codes):
            #分割数据
            train_queires, train_codes, train_lables = current_queries[train_index], current_codes[train_index], current_labels[train_index]
            test_queires, test_codes, test_lables = current_queries[test_index], current_codes[test_index], current_labels[test_index]
            # print(test_index)

            #训练模型
            #TODO 这种训练方式合理吗
            model_train(model, device, scaler, optimizer, lossfuction, tokenizer, batch_size, train_queires, train_codes,
                        train_lables)
            #评估模型
            predictions = model_predict(tokenizer, model, test_queires, test_codes, test_lables, device)

            #找到被错误分类的样本
            misclassified = (predictions != test_lables)

            # print("len(misclassified): ", len(test_codes[misclassified]))

            all_misclassified_samples.extend(test_codes[misclassified])

            print("len(all_misclassified_samples): ",len(all_misclassified_samples))

            ########valid########
            # 每个epoch都验证一下
            model.eval()
            pure_mrr_inference(tokenizer_path, model, infer_file_path, output_infer_file)
            current_mrr = get_mrr(lang)

            print("epoch: ", epoch)
            print('currnt mrr %.8f, max mrr %.8f' % (current_mrr, max_mrr))

            # 保存最优
            if (current_mrr > max_mrr):
                # torch.save(model, model_save_dir) #这样就更换了，问题不大
                model.save_adapter(adaptor_save_dir,
                                   "bottleneck_adapter")  # 如果目前的model好于max_mrr，那么就存为best。  如果后续没超过，则不更新best，否则更新
                max_mrr = current_mrr
                print('max mrr %.8f' % (max_mrr))

        if(len(current_codes) <= train_num): #小于标记预算
            # 采样出错误分类样本的相似样本，然后和原始训练集混合，再进行训练。
            #需要去重，已经采样过的，不会再被采样
            all_codes, all_queries, all_labels = dedupe(current_codes, all_codes, all_queries, all_labels)
            similar_codes, similar_queries, similar_labels = find_most_similar_codes(all_misclassified_samples,
                                                                                     all_codes, all_queries, all_labels)
            #TODO 有一个问题，在后续的epoch找的时候，会不会找到以前被找过的？需要处理
            # 没有对0或者1分开处理，会不会label 不平衡
            current_codes, current_queries, current_labels = list(current_codes), list(current_queries), list(current_labels)
            current_codes.extend(similar_codes)
            current_queries.extend(similar_queries)
            current_labels.extend(similar_labels)
            current_codes, current_queries, current_labels = np.array(current_codes), np.array(current_queries), np.array(current_labels)
            current_codes, current_queries, current_labels = current_codes[:train_num], current_queries[:train_num], current_labels[:train_num]
        else:
            current_codes, current_queries, current_labels = current_codes[:train_num], current_queries[:train_num], current_labels[:train_num]

        # TODO 要考虑到样本不平衡如何处理, 先通过这个观察观察
        print("label 1: {}, label 0 {}".format(sum(current_labels), len(current_labels) - sum(current_labels)))
        print("len(current_codes): ", len(current_codes))

    # deactivate all adapters
    # model.set_active_adapters(None)
    # delete the added adapter
    # model.delete_adapter('bottleneck_adapter')

#86?
if __name__ == '__main__':
    set_seed(1) #固定随机种子

    #配置
    lang = "sql" #代码搜索时的编程语言
    re_num_epochs = 10
    re_batch_size = 32

    #数据
    adaptor_save_dir = "../../save_model/"+ lang +"/" + str(lang) + "_adaptor" #adapter的保存位置
    train_file_path = "../../data/train_valid/"+ lang +"/train.txt"  # 训练文件目录

    infer_file_path = "../../data/test/"+ lang +"/batch_0.txt" # 推理文件目录
    output_infer_file = "../../results/"+ lang +"/adaptor_batch_0.txt"  # 推理结果目录

    #模型
    tokenizer_path = "../../graph_code_bert"  # 分词器
    base_model_path = "./initial_base_model_adapter/java_model_sqlvalid_0.95hard100000_5e-06"  # 基础模型
    best_super_adapter_dir = "./initial_base_model_adapter/java_adapter_sqlvalid_0.95hard100000_5e-06"  # 初始的adapter

    best_path = []

    for train_num in [5000]: #
        for lr in [1e-3, 6e-4, 2e-4, 5e-5]: #lr
        # for lr in [5e-5, 2e-5, 1e-5, 8e-6, 5e-6]:
            #train
            run_retrain_active(train_num, lr, lang, adaptor_save_dir, train_file_path, infer_file_path, output_infer_file,
                        best_path, best_super_adapter_dir,
                        re_num_epochs, re_batch_size, tokenizer_path, base_model_path)

            #inference
            mrr_inference(base_model_path, tokenizer_path, adaptor_save_dir, infer_file_path, output_infer_file, 0)

            #获取结果
            get_mrr(lang)

            print("lr {}, train_num {}".format(lr, train_num))
