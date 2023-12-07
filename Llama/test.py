import torch
from NewsDataLoader import getABatch, getVocabSize
import torch.nn.functional as F
import numpy as np


m = torch.load("NewsModel.pth", map_location='cpu')
allCount = 0
accCount = 0
for x, y in getABatch('val', 512, 1024):
    x = x.to('cpu')
    y = y.to('cpu')
    allCount += 512
    output = m.pre(x)
    acc = y == output
    acc = acc.sum().item()
    accCount += acc
print(f"总数据{allCount}")
print(f"准确分类数据{accCount}")
print(f"准确率{accCount / allCount}")