from NewsDataLoader import getABatch, getVocabSize
import torch

import train
from GQAModel import GQALLama1

n_embedding: int = 360  # 嵌入维度
# 注意力相关参数
n_heads: int = 4  # 注意力头
head_dim: int = n_embedding // n_heads  # 每个注意力头的维度
vocab_size = getVocabSize()
print(vocab_size)  # 词表大小
multiple_of: int = 4  # make SwiGLU hidden layer size multiple of large power of 2
batch_size: int = 128  # 一个批量大小
block_size: int = 512  # 一个批量中包含的字符数
dropout: float = 0.2
device: str = 'cuda:5' if torch.cuda.is_available() else 'cpu'
# device="cpu"
max_iter: int = 10

feed_forward_mode = "relu"
norm = "none"
pos_embed_method = "sin"
# 14分类任务
m = GQALLama1(train.vocab_size, out_features=14, n_heads=4,
                  n_embedding=n_embedding, block_size=block_size, dropout=dropout,
                  feed_forward_mode=feed_forward_mode, norm=norm,
                  pos_embed_method=pos_embed_method)

optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)
train.train(model=m, model_name='GQA_qkv011', optimizer=optimizer, max_iter=max_iter, batch_size=batch_size,
                block_size=block_size,
                n_embedding=n_embedding,
                getABatch=train.getABatch, device=device)