# -*- encoding==utf-8 -*-
import model
from model import LLamaOnlyPERO,device
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
# from data import getDataLoader, getVocabSize
import matplotlib.pyplot as plt
import logging
from NewsDataLoader import getABatch, getVocabSize

n_embedding: int = 168  # 嵌入维度
# 注意力相关参数
n_heads: int = 4  # 注意力头
head_dim: int = n_embedding // n_heads  # 每个注意力头的维度
vocab_size: int = -1  # 词表大小
multiple_of: int = 4  # make SwiGLU hidden layer size multiple of large power of 2
batch_size: int = 128  # 一个批量大小
block_size: int = 1024  # 一个批量中包含的字符数
dropout: int = 0.2
#device: str = 'cuda:4' if torch.cuda.is_available() else 'cpu'
#device="cpu"
max_iter: int = 5

# 创建logger对象
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

# 创建FileHandler并设置日志格式、保存路径等参数
file_handler = logging.FileHandler('log', mode='w')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 添加FileHandler到logger对象
logger.addHandler(file_handler)


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    losses = []
    for x, y in getABatch('val', batch_size, block_size):
        x = x.to(device)
        y = y.to(device)
        logits, loss = model(x, y)
        losses.append(loss.item())
    out = np.mean(losses)
    model.train()
    return out


vocab_size = getVocabSize()
print(vocab_size)

feed_forward_mode="swish"
norm="rms"
pos_embed_method="repo"
#14分类任务
m = model.LLama1(vocab_size, out_features=14, n_heads=4,
                 n_embedding=n_embedding, block_size=block_size, dropout=dropout,
                 feed_forward_mode = feed_forward_mode, norm=norm,
                 pos_embed_method=pos_embed_method)
optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)
trainLosses = []
val_losses = []
count = 0
logger.info("start training")
print("start training")
params_num=sum(p.numel() for p in m.parameters()) / 1e6
print(params_num)
for step in range(max_iter):
    m.to(device)
    trainLoss = []
    logger.info(f"The step is {step}")
    for X, Y in getABatch('train', batch_size, block_size):
        X, Y = X.to(device), Y.to(device)
        logits, loss = m(X,Y)
        trainLoss.append(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if step != -1:
        val_loss = estimate_loss(m)
        t_loss = np.mean(trainLoss)
        trainLosses.append(t_loss)
        val_losses.append(val_loss)
        count += 1
        logger.info(f"step{step}: train loss {t_loss}, val loss {val_loss}")
    m.to("cpu")
    allCount = 0
    accCount = 0
    for x, y in getABatch('val', 512, block_size):
        x = x.to('cpu')
        y = y.to('cpu')
        allCount += 512
        output = m.pre(x)
        y=y.view(block_size)
        acc = y == output
        acc = acc.sum().item()
        accCount += acc
    print(f"step:{step},total:{allCount}")
    print(f"acc num:{accCount}")
    print(f"acc:{accCount / allCount}")
    logger.info(f"step:{step},toal num:{allCount}")
    logger.info(f"acc num:{accCount}")
    logger.info(f"acc:{accCount / allCount}")
    torch.save(m,f'./output/mod-_{feed_forward_mode}_{norm}_{pos_embed_method}-{block_size}_{n_embedding}-{step}-{params_num}_{accCount / allCount}.pth')

torch.save(m.cpu(), 'NewsModel.pth')
plt.plot(trainLosses, label='train_loss')
plt.plot(val_losses, label='val_loss')
plt.xlabel('step')
plt.ylabel('loss')
plt.title('train and val loss')
plt.legend()

# 显示图形
plt.show()
plt.savefig(f"loss_{feed_forward_mode}_{norm}_{pos_embed_method}-{block_size}_{n_embedding}-{step}-{params_num}_{accCount / allCount}.png")
