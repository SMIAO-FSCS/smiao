import torch
import os

from torch.cuda.amp import autocast
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from .spos_mrr import get_mrr

class LineByLineTextDataset(Dataset):
    def __init__(self, file_path: str, split_num=0):
        print("read data file at:", file_path)
        assert os.path.isfile(file_path)

        with open(file_path, encoding="utf-8") as f:
            self.lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        # 截断测试
        if(split_num != 0):
            self.lines = self.lines[:split_num]

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


        print("注释和代码总行数:", len(self.text_lines), len(self.code_lines))

    def __len__(self):
        return len(self.text_lines)  # 注意这个len本质是数据的数量

    def __getitem__(self, i):
        a = self.text_lines[i]
        b = self.code_lines[i]
        c = self.labels[i]
        return a, b, c




def write_result_to_file(output_test_file, all_result, test_data_dir, test_num):
    assert os.path.isfile(test_data_dir)
    # assert os.path.isfile(output_test_file) #是即将要新建的文件，不需要判断存不存在
    print("read test file at:", test_data_dir)

    with open(test_data_dir, encoding="utf-8") as f:
        lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())] #每一行都读进来
    assert (len(lines) % test_num == 0) #要和test_num对的上，不然怕写错了

    with open(output_test_file, "w") as writer:
        print("***** Output test results *****")
        for i, logit in tqdm(enumerate(all_result), desc='Testing'):
            # instance_rep = '<CODESPLIT>'.join(
            #     [item.encode('ascii', 'ignore').decode('ascii') for item in lines[i]])
            writer.write(lines[i] + '<CODESPLIT>' + '<CODESPLIT>'.join([str(l) for l in logit]) + '\n')

#组装预训练模型 + adaptor
def mrr_inference(base_model_name, tokenizer_name, adaptor_save_dir, infer_file_path, output_infer_file, split_num = 0):
    print("run mrr inference")
    # 配置
    batch_size = 32

    # adaptor_save_dir = "../save_model/go/go_adaptor"
    fine_turn_model_name = base_model_name
    tokenizer_name = tokenizer_name

    # infer_file_path = "../data/test/go/batch_0.txt" #arg2
    # output_infer_file = "../results/go/go_adaptor_batch_0.txt"  #arg3 last arg

    print("infer_file_path: ", infer_file_path)
    print("adaptor_save_dir ", adaptor_save_dir)

    ########################## 数据 ############65 #############
    # infer_file_path = "../look"
    infer_dataset = LineByLineTextDataset(file_path=infer_file_path, split_num=split_num)
    infer_dataLoader = DataLoader(infer_dataset, batch_size, shuffle=False)

    # device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("train_device: ", device)

    ########## MODEL ##############################
    # model = torch.load(inference_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(fine_turn_model_name)
    model.delete_adapter('bottleneck_adapter')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    adapter_name = model.load_adapter(adaptor_save_dir)
    model.train_adapter("bottleneck_adapter")  #
    model.set_active_adapters(adapter_name)

    model.to(device)

    ######################### Inference #########################

    all_result = []

    model.eval()
    size = len(infer_dataLoader)
    test_progress_bar = tqdm(range(size), desc='mrr_inference', mininterval=10)
    for text, code, labels in infer_dataLoader:

        batch_tokenized = tokenizer(list(text), list(code), add_special_tokens=True,
                                    padding=True, max_length=256,
                                    truncation=True, return_tensors="pt")  # tokenize、add special token、pad
        batch_tokenized = batch_tokenized.to(device)

        with torch.no_grad():
            with autocast():
                outputs = model(**batch_tokenized)
            logits = outputs.logits
            all_result.extend(logits.detach().cpu().numpy())

        test_progress_bar.update(1)

    test_data_dir = infer_file_path

    test_num = 1000
    write_result_to_file(output_infer_file, all_result, test_data_dir, test_num) #看了以下，输出和之前的代码完全一致了，inference的result算是搞定了

    print("mrr inference end")

#在训练的时候，将预训练模型和adaptor看作同一个model，进来inference，节省内存
def pure_mrr_inference(tokenizer_name, model, infer_file_path, output_infer_file):
    print("run mrr inference")
    # 配置
    batch_size = 32

    # model_name = "../../graph_code_bert"

    ########################## 数据 ############65 #############
    infer_dataset = LineByLineTextDataset(file_path=infer_file_path, split_num=50000)
    infer_dataLoader = DataLoader(infer_dataset, batch_size, shuffle=False)

    # device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("train_device: ", device)

    ########## MODEL ##############################
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    ######################### Inference #########################

    all_result = []

    model.eval()
    size = len(infer_dataLoader)
    # test_progress_bar = tqdm(range(size))
    test_progress_bar = tqdm(range(size), desc='train_model_mrr_inference_ing', mininterval=10)
    for text, code, labels in infer_dataLoader:

        batch_tokenized = tokenizer(list(text), list(code), add_special_tokens=True,
                                    padding=True, max_length=128, #之前看设置的是512
                                    truncation=True, return_tensors="pt")  # tokenize、add special token、pad
        batch_tokenized = batch_tokenized.to(device)

        with torch.no_grad():
            with autocast():
                outputs = model(**batch_tokenized)
            logits = outputs.logits
            all_result.extend(logits.detach().cpu().numpy())

        test_progress_bar.update(1)

    test_data_dir = infer_file_path

    test_num = 1000
    write_result_to_file(output_infer_file, all_result, test_data_dir, test_num) #看了以下，输出和之前的代码完全一致了，inference的result算是搞定了

    print("mrr inference end")


if __name__ == '__main__':
    tokenizer_path = "../../../graph_code_bert"  # 分词器
    base_model_path = "initial_base_model_adapter/python_model_hardonce_unified_topk5_2e5"  # 基础模型

    adaptor_save_dir = "../../../save_model/solidity/solidity_adaptor"  # arg3

    infer_file_path = "../../../data/test/solidity/batch_0.txt"
    output_infer_file = "../../../results/solidity/adaptor_batch_0.txt"
    mrr_inference(base_model_path, tokenizer_path, adaptor_save_dir, infer_file_path, output_infer_file, 50000)

    get_mrr('solidity')