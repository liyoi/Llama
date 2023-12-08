# -*- encoding==utf-8 -*-
import time

# from data import getDataLoader, getVocabSize
import matplotlib.pyplot as plt
import numpy as np
import torch

import model

from utils import get_logger


# 暂时将训练和验证合在一起
def train(model, model_name, optimizer, max_iter, batch_size, block_size, n_embedding, getABatch, device="cuda"):
    m = model
    params_num = sum(p.numel() for p in m.parameters()) / 1e6
    print(params_num)
    trainLosses = []
    val_losses = []
    logger = get_logger()
    logger.info(f"train.py,n_embedding:{m.n_embedding},block_size:{m.block_size}")
    logger.info("start training")
    print("start training")
    train_start_time = time.time()
    count = 0
    for step in range(max_iter):
        m.to(device)
        trainLoss = []
        logger.info(f"The step is {step}")
        for X, Y in getABatch('train', batch_size, m.block_size):
            X, Y = X.to(device), Y.to(device)
            logits, loss = m(X, Y)
            trainLoss.append(loss.item())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if step != -1:
            # val_loss = estimate_loss(m)
            val_loss = 0
            t_loss = np.mean(trainLoss)
            trainLosses.append(t_loss)
            val_losses.append(val_loss)
            count += 1
            logger.info(f"step：{step}: train loss {t_loss}, val loss {val_loss}")
        train_time_elapsed = time.time()
        all_count = 0
        acc_count = 0
        for x, y in getABatch('val', 512, m.block_size):
            x = x.to(device)
            y = y.to(device)
            all_count += 512
            output = m.pre(x)
            y = y.view(512)
            acc = y == output
            acc = acc.sum().item()
            acc_count += acc
        # 打印结果日志
        logger.info(
            f"model:{model_name},{m.feed_forward_mode}_{m.norm}_{m.pos_embed_method}-{m.block_size}_{m.n_embedding}-{step}")
        print(f"step:{step},total:{all_count}")
        print(f"acc num:{acc_count}")
        print(f"acc:{acc_count / all_count}")
        logger.info(f"step:{step},toal num:{all_count}")
        logger.info(f"acc num:{acc_count}")
        logger.info(f"acc:{acc_count / all_count}")
        torch.save(m,
                   f'./output/{model_name}-_{m.feed_forward_mode}_{m.norm}_{m.pos_embed_method}-{m.block_size}_{m.n_embedding}-{step}-{params_num}_{acc_count / all_count}.pth')
        val_time_elapsed = time.time()

        # 显示图形
        plt.plot(trainLosses, label='train_loss')
        plt.plot(val_losses, label='val_loss')
        plt.xlabel('step')
        plt.ylabel('loss')
        plt.title('train and val loss')
        plt.legend()
        plt.show()
        plt.savefig(
            f"loss_{model_name}_{m.feed_forward_mode}_{m.norm}_{m.pos_embed_method}-{m.block_size}_{m.n_embedding}-{step}-{params_num}_{acc_count / all_count}.png")

        # 打印训练测试时间
        logger.info(
            f"train_time_elapsed:{train_time_elapsed - train_start_time},val_time_elapsed:{val_time_elapsed - train_time_elapsed}")
        print(
            f"train_time_elapsed:{train_time_elapsed - train_start_time},val_time_elapsed:{val_time_elapsed - train_time_elapsed}")
        return 1


# def train_and_val(model, model_name, optimizer, max_iter, batch_size, block_size, n_embedding, getABatch, device="cuda"):


if __name__ == '__main__':
    from NewsDataLoader import getABatch, getVocabSize

    n_embedding: int = 168  # 嵌入维度
    # 注意力相关参数
    n_heads: int = 4  # 注意力头
    head_dim: int = n_embedding // n_heads  # 每个注意力头的维度
    vocab_size = getVocabSize()
    print(vocab_size)  # 词表大小
    multiple_of: int = 4  # make SwiGLU hidden layer size multiple of large power of 2
    batch_size: int = 128  # 一个批量大小
    block_size: int = 512  # 一个批量中包含的字符数
    dropout: int = 0.2
    device: str = 'cuda:4' if torch.cuda.is_available() else 'cpu'
    # device="cpu"
    max_iter: int = 10
    feed_forward_mode = "relu"
    norm = "none"
    pos_embed_method = "sin"
    # 14分类任务
    m = model.LLama1(vocab_size, out_features=14, n_heads=4,
                     n_embedding=n_embedding, block_size=block_size, dropout=dropout,
                     feed_forward_mode=feed_forward_mode, norm=norm,
                     pos_embed_method=pos_embed_method)
    optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)

    train(model=m, model_name='Llama1', optimizer=optimizer, max_iter=max_iter, batch_size=batch_size,
          block_size=block_size,
          n_embedding=n_embedding,
          getABatch=getABatch, device=device)

    torch.save(m.cpu(), 'NewsModel.pth')
