import os
import random
import torch
from tokenizer import Tokenizer

tokenizerPath = 'TextModel/NewsModel.model'
tok = Tokenizer(tokenizerPath)
newsAllItems = []  # 单条新闻的文本，一共767010条新闻
for i in range(0, 14):
    for filename in os.listdir(f'../../liweih/NewsClassification/{i}'):
        with open(f"../../liweih/NewsClassification/{i}/{filename}", 'r', encoding='utf-8') as f:
            content = f.read()
            item = [content, i]
            newsAllItems.append(item)
# 打乱顺序，引入随机性
random.shuffle(newsAllItems)
val_test_data_len = int(0.1 * len(newsAllItems))  # 验证集大小 和 测试集大小
newsValItems = newsAllItems[:val_test_data_len]
newsTestItems = newsAllItems[val_test_data_len : val_test_data_len * 2]
newsTrainItems = newsAllItems[val_test_data_len * 2 :]
print("***data has loaded!")

def getVocabSize():
    return tok.getVocabSize()

def getABatch(split, batch_size, block_size):
    # 规范输入
    accept_string = {'val', 'train', 'test'}
    assert split in accept_string, f"Invalid input."
    pre_index = 0  # 从第0个下标开始
    assert split
    if split == 'train':
        data = newsTrainItems
    elif split == 'val':
        data = newsValItems
    else:
        data = newsTestItems
    batch_num = len(data) // batch_size  # 一共有多少个batch
    for _ in range(0, batch_num):
        Xs = [item[0] for item in data[pre_index: pre_index + batch_size]]  # 此时取出一个batch中的新闻
        Ys = [item[1] for item in data[pre_index: pre_index + batch_size]]  # 取出当前batch中的新闻标签
        pre_index += batch_size
        x = [tok.encode(s, eos=True, bos=True) for s in Xs]  # 自然语言转化为数字
        for index in range(0, len(x)):
            if len(x[index]) < block_size:  # 如果当前的新闻长度小于block_size，就需要填充
                padData = [tok.pad_id] * (block_size - len(x[index]))
                x[index] = x[index] + padData
            else:   # 如果当前的新闻长度大于block_size，就需要截断
                x[index] = x[index][0:block_size]
        x = torch.tensor(x)
        y = torch.tensor(Ys).reshape(batch_size, -1)
        yield x, y