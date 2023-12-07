"""
IMDb数据集（情感分析）的训练和测试***
"""
import torch
from torch import nn

from IMDb_dataloader import load_data_imdb
from model import LLama1

dropout = 0.2
learning_rate = 3e-4  # 学习率
block_size = 200  # 滑动窗口大小
eval_interval = 500
eval_iters = 200
n_layer = 6
n_embedding = 360
max_iters = 20
batch_size = 32
head_size = 16
limit_top_number = 5  # no use
# n_embedding = 160
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_iter, test_iter, vocab = load_data_imdb(batch_size)

los = nn.CrossEntropyLoss().to(device)

model = LLama1(vocab_size=49346).to(device)
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
    torch.save(model, f'model_llama1_{i}_{current_iter}.pth')
# # 训练第二遍
# model.train()
# for data in enumerate(train_iter):
#     # every once in a while evaluate the loss on train and val sets
#     # sample a batch of data
#     xb = data[1][0].to(device)
#     yb = data[1][1].to(device)
#     # evaluate the loss
#     logits, loss = model(xb, yb)
#     if data[0] % 100 == 0:
#         print(f"loss {loss}")
#     # loss.cpu()
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()
# torch.save(model, 'model1.pth')

# Test

acc_num = 0
total = 0
for data in enumerate(train_iter):
    model.train(False)
    # every once in a while evaluate the loss on train and val sets
    # sample a batch of data
    xb = data[1][0].to(device)
    yb = data[1][1].to(device)
    output = model.pre(xb)
    acc_num = acc_num + torch.sum(output == yb).item()
    total = total + 16
    print(acc_num)

# 四次的正确率 16 num_step 19233 0.769073896353167  20090 0.8033429302623161    20720 0.8288  20751 0.8288
