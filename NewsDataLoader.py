import os
import random
import torch
from tokenizer import Tokenizer

path = "/chengge/liweih"
if os.path.exists(path):
    pass
else:
    path = ".."
tokenizerPath = 'TextModel/NewsModel.model'
tok = Tokenizer(tokenizerPath)
newsTrainItems = []  # 单条新闻的文本，一共767010条新闻
for i in range(0, 14):
    for filename in os.listdir(os.path.join(path, f'/NewsClassification/train/{i}')):
        with open(os.path.join(path, f'/NewsClassification/train/{i}/{filename}'), 'r', encoding='utf-8') as f:
            content = f.read()
            item = [content, i]
            newsTrainItems.append(item)
# 打乱顺序，引入随机性
random.shuffle(newsTrainItems)
newsTestItems = []  # 单条新闻的文本，一共767010条新闻
for i in range(0, 14):
    for filename in os.listdir(os.path.join(path, f'/NewsClassification/train/{i}')):
        with open(os.path.join(path, f'/NewsClassification/train/{i}/{filename}'), 'r', encoding='utf-8') as f:
            content = f.read()
            item = [content, i]
            newsTestItems.append(item)
# 打乱顺序，引入随机性
random.shuffle(newsTestItems)
print("数据加载完毕")


def getVocabSize():
    return tok.getVocabSize()


def getABatch(split, batch_size, block_size):
    pre_index = 0  # 从第0个下标开始
    data = newsTrainItems if split == 'train' else newsTestItems
    batch_num = len(data) // batch_size  # 一共有多少个batch
    print(batch_num)
    for _ in range(0, batch_num):
        Xs = [item[0] for item in data[pre_index: pre_index + batch_size]]  # 此时取出一个batch中的新闻
        Ys = [item[1] for item in data[pre_index: pre_index + batch_size]]  # 取出当前batch中的新闻标签
        pre_index += batch_size
        x = [tok.encode(s, eos=True, bos=True) for s in Xs]
        for index in range(0, len(x)):
            if len(x[index]) < block_size:  # 如果当前的新闻长度小于block_size，就需要填充
                padData = [tok.pad_id] * (block_size - len(x[index]))
                x[index] = x[index] + padData
            else:
                x[index] = x[index][0:block_size]
        x = torch.tensor(x)
        y = torch.tensor(Ys).reshape(batch_size, -1)
        yield x, y
