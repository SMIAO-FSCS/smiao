import random

def balance_samples_with_distinct_zero_pairs(text_lines: object, code_lines: object, labels: object) -> object:
    if not (len(text_lines) == len(code_lines) == len(labels)):
        raise ValueError("All input lists must have the same length.")

    count_label_0 = labels.count(0)
    count_label_1 = labels.count(1)

    # 如果0的数量大于1的数量，则对1进行过采样
    if count_label_0 > count_label_1:
        indices_label_1 = [i for i, label in enumerate(labels) if label == 1]
        num_samples_to_add = count_label_0 - count_label_1
        for _ in range(num_samples_to_add):
            index = random.choice(indices_label_1)
            text_lines.append(text_lines[index])
            code_lines.append(code_lines[index])
            labels.append(1)
    # 如果1的数量大于0的数量，随机匹配text和code构成新的0标签样本对，确保其不与任何标签为1的样本对相同
    elif count_label_1 > count_label_0:
        num_samples_to_add = count_label_1 - count_label_0
        # 获取所有标签为1的样本对
        label_1_pairs = set((text_lines[i], code_lines[i]) for i, label in enumerate(labels) if label == 1)
        all_indices = set(range(len(text_lines)))
        for _ in range(num_samples_to_add):
            while True:
                # 随机选择不同的text和code索引
                text_index, code_index = random.sample(all_indices, 2)
                text_sample = text_lines[text_index]
                code_sample = code_lines[code_index]
                # 确保新的样本对不在标签为1的样本对中
                if (text_sample, code_sample) not in label_1_pairs:
                    text_lines.append(text_sample)
                    code_lines.append(code_sample)
                    labels.append(0)
                    break

    return text_lines, code_lines, labels

def remove_conflicting_zero_pairs(text_lines, code_lines, labels):
    if not (len(text_lines) == len(code_lines) == len(labels)):
        raise ValueError("All input lists must have the same length.")

    # 创建一个字典来存储每个<query, code>对应的label
    pair_to_label = {}

    # 遍历所有样本
    for text, code, label in zip(text_lines, code_lines, labels):
        pair = (text, code)

        # 如果这个pair已经存在于字典中
        if pair in pair_to_label:
            # 如果当前label为1且之前存储的label为0，则更新label为1
            if label == 1 and pair_to_label[pair] == 0:
                pair_to_label[pair] = 1
        else:
            # 如果pair不存在于字典中，直接添加
            pair_to_label[pair] = label

    # 现在pair_to_label包含了没有冲突的<query, code, label>
    # 重新构建text_lines, code_lines, labels数组
    new_text_lines = []
    new_code_lines = []
    new_labels = []

    for pair, label in pair_to_label.items():
        new_text_lines.append(pair[0])
        new_code_lines.append(pair[1])
        new_labels.append(label)

    return new_text_lines, new_code_lines, new_labels