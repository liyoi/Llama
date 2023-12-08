"""
IMDb数据集（情感分析）的训练和测试***
"""
import logging

import numpy as np
import torch
from torch import nn

from IMDb_dataloader import load_data_imdb
from NewsDataLoader import getVocabSize
from model import LLama1
from train import estimate_loss

dropout = 0.2
learning_rate = 1e-3  # 学习率
block_size = 200  # 滑动窗口大小
eval_interval = 500
eval_iters = 200
n_layer = 6
n_embedding = 168
max_iters = 20
batch_size = 32
head_size = 16
limit_top_number = 5  # no use
# n_embedding = 160
device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_embedding: int = 360  # 嵌入维度
# 注意力相关参数
n_heads: int = 4  # 注意力头
head_dim: int = n_embedding // n_heads  # 每个注意力头的维度
vocab_size: int = -1  # 词表大小
multiple_of: int = 4  # make SwiGLU hidden layer size multiple of large power of 2
batch_size: int = 128  # 一个批量大小
block_size: int = 512  # 一个批量中包含的字符数
dropout: int = 0.2
device: str = 'cuda:6' if torch.cuda.is_available() else 'cpu'
# device="cpu"
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


vocab_size = getVocabSize()
print(vocab_size)

train_iter, test_iter, vocab = load_data_imdb(batch_size)

los = nn.CrossEntropyLoss().to(device)
feed_forward_mode = "relu"
norm = "none"
pos_embed_method = "repo"
model = LLama1(vocab_size, out_features=2, n_heads=4,
               n_embedding=n_embedding, block_size=block_size, dropout=dropout,
               feed_forward_mode=feed_forward_mode, norm=norm,
               pos_embed_method=pos_embed_method)

# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("开始训练")
# Train 训练第一遍
for i in range(max_iters):
    current_iter = 0
    for data in enumerate(train_iter):
        current_iter = current_iter + 1
        # every once in a while evaluate the loss on train and val sets
        # sample a batch of data
        xb = data[1][0].to(device)
        yb = data[1][1].to(device)
        # evaluate the loss
        logits, loss = model(xb, yb)
        if data[0] % 100 == 0:
            print(f"current_iter:{current_iter}  loss {loss}")
        # loss.cpu()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    acc = estimate_loss(model)
    torch.save(model, f'model_llama1_{i}_{current_iter}_{acc}.pth')

# 四次的正确率 16 num_step 19233 0.769073896353167  20090 0.8033429302623161    20720 0.8288  20751 0.8288
