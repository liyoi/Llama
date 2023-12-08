#!/usr/bin/env python
# coding: utf-8
import train
from model import precompute_freqs_cis, apply_rotary_emb, RMSNorm, FeedForward
import torch
from torch import nn
import torch.nn.functional as F


class GQAHead(nn.Module):
    """ 掩藏注意力 """
    freqs_cis = None

    def __init__(self, head_size, n_embedding, block_size, dropout=0.2, pos_embed_method="rope"):
        super().__init__()
        self.query = nn.Linear(n_embedding, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.pos_embed_method = pos_embed_method
        self.freqs_cis = precompute_freqs_cis(dim=head_size, seq_len=block_size)

    def forward(self, x, key, value):
        B, T, C = x.shape
        query = self.query(x)
        if self.pos_embed_method == 'rope':
            # Reformer相对位置编码
            #           if self.freqs_cis is None:
            #               self.freqs_cis = precompute_freqs_cis(dim=key.shape[-1], seq_len=T).to(key.device)
            xq, xk = apply_rotary_emb(key, query, self.freqs_cis.to(x.device))
            # key*value/(d**-0.5)
            query, key = xq, xk
        wei = key @ query.transpose(-2, -1) * (key.shape[-1] ** -0.5)
        # 掩藏操作，使注意力只能看到前面数据
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, -1)
        wei = self.dropout(wei)
        outputs = wei @ value
        return outputs


class MultiQueryAttention(nn.Module):
    """ 多组查询注意力 """

    def __init__(self, num_heads, head_size, block_size=256, n_embedding=360, pos_embed_method="rope"):
        super().__init__()
        # 多头注意力由多个注意力叠加
        self.heads = nn.ModuleList(
            [GQAHead(head_size=head_size, block_size=block_size, n_embedding=n_embedding,
                     pos_embed_method=pos_embed_method) for _ in
             range(num_heads)])
        self.linear = nn.Linear(n_embedding, n_embedding)
        self.key = nn.Linear(n_embedding, head_size, bias=False)
        self.value = nn.Linear(n_embedding, head_size, bias=False)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # 拼接各个注意力的输出结果
        key = self.key(x)
        value = self.value(x)
        output = torch.cat([h(x, key, value) for h in self.heads], dim=-1)
        output = self.dropout(self.linear(output))
        return output


class MultiQueryAttention(nn.Module):
    """ 多组查询注意力 """

    def __init__(self, num_heads, head_size, block_size=256, n_embedding=360, pos_embed_method="rope"):
        super().__init__()
        # 多头注意力由多个注意力叠加
        self.heads = nn.ModuleList(
            [GQAHead(head_size=head_size, block_size=block_size, n_embedding=n_embedding,
                     pos_embed_method=pos_embed_method) for _ in
             range(num_heads)])
        self.linear = nn.Linear(n_embedding, n_embedding)
        self.key = nn.Linear(n_embedding, head_size, bias=False)
        self.value = nn.Linear(n_embedding, head_size, bias=False)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # 拼接各个注意力的输出结果
        key = self.key(x)
        value = self.value(x)
        output = torch.cat([h(x, key, value) for h in self.heads], dim=-1)
        output = self.dropout(self.linear(output))
        return output


class GQABlock(nn.Module):
    """
    feed_forward_mode "swish"-前馈神经网络使用swish激活函数，”relu“-使用relu激活函数
    norm ："RMS"-使用RMSNorm归一化，其他值则使用nn.LayerNorm
    """

    def __init__(self, n_emb, n_head, block_size, dropout=0.2, feed_forward_mode: str = "swish", norm="rms",
                 pos_embed_method="repo"):
        super().__init__()
        self.norm = norm
        self.swish = feed_forward_mode
        head_size = n_emb // n_head
        self.heads = MultiQueryAttention(n_head, head_size, block_size, n_emb, pos_embed_method=pos_embed_method)
        self.fb = FeedForward(n_emb, 4 * n_emb, 2, dropout, mode=feed_forward_mode)
        if norm == "rms":
            self.l1 = RMSNorm(n_emb)
            self.l2 = RMSNorm(n_emb)
        else:
            self.l1 = nn.LayerNorm(n_emb)
            self.l2 = nn.LayerNorm(n_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.norm == "rms":
            x = self.l1(x)
            x = x + self.heads(x)
            x = self.l2(x)
            x = x + self.fb(x)
        else:
            x = x + self.heads(x)
            x = self.l1(x)
            x = x + self.fb(x)
            x = self.l2(x)
        return x


class GQALLama(nn.Module):
    # 完整的Llama模型
    def __init__(self, vocab_size, out_features=2, n_heads=4,
                 n_embedding=360, block_size=200, dropout=0.2,
                 feed_forward_mode: str = "swish", norm="rms",
                 pos_embed_method="repo"):
        super().__init__()
        self.vocab_size = vocab_size
        self.out_features = out_features
        self.n_heads = n_heads
        self.n_embedding = n_embedding
        self.pos_embed_method = pos_embed_method
        self.block_size = block_size
        self.embedding = torch.nn.Embedding(vocab_size, n_embedding)
        if pos_embed_method == "sin":
            self.position_emb = nn.Embedding(block_size, n_embedding)
        self.heads = nn.Sequential(
            GQABlock(n_embedding, n_heads, block_size, pos_embed_method=pos_embed_method,
                     feed_forward_mode=feed_forward_mode, norm=norm),
            GQABlock(n_embedding, n_heads, block_size, pos_embed_method=pos_embed_method,
                     feed_forward_mode=feed_forward_mode, norm=norm),
            GQABlock(n_embedding, n_heads, block_size, pos_embed_method=pos_embed_method,
                     feed_forward_mode=feed_forward_mode, norm=norm),
            nn.Dropout(dropout)
        )
        # self.l1 = torch.nn.Linear(n_embedding, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        word_embedding = self.embedding(idx)
        x = word_embedding
        if self.pos_embed_method == "sin":
            pos_embedding = self.position_emb(torch.arange(0, T).to(idx.device))
            pos_embedding = torch.repeat_interleave(pos_embedding, B).view(B, T, self.n_embedding)
            x = x + pos_embedding
        x = self.heads(x)
        return x


class GQALLama1(nn.Module):
    """
    pos_embed_method 限制取值 "rope"|"sin"
    """

    def __init__(self, vocab_size, out_features=2, n_heads=4,
                 n_embedding=360, block_size=200, dropout=0.2,
                 feed_forward_mode: str = "swish", norm="rms",
                 pos_embed_method="repo"):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embedding = n_embedding
        self.feed_forward_mode = feed_forward_mode
        self.norm = norm
        self.pos_embed_method = pos_embed_method
        self.l1 = torch.nn.Linear(n_embedding, out_features)
        self.llama = GQALLama(vocab_size, out_features, n_heads,
                              n_embedding, block_size, dropout,
                              feed_forward_mode, norm,
                              pos_embed_method)

    def forward(self, idx, targets=None):
        x = self.llama(idx, targets)
        logits = self.l1(x)
        loss = None
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits[:, -1, :]
            logits = logits.view(B, C)
            targets = targets.view(B)
            loss = torch.nn.functional.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, block_size=16):
        # 生成文本
        for _ in range(max_new_tokens):
            # get the predictions
            idx_conv = idx[:, -block_size:]
            logits, loss = self(idx_conv)
            logits = logits[:, -1, :]
            # apply softmax to get the probability
            probs = torch.nn.functional.softmax(logits, dim=-1)
            top_probs = torch.topk(probs, 5, dim=-1)
            next_index = torch.multinomial(top_probs[0], num_samples=1)[0][0]
            idx_next = top_probs[1][0][next_index.item()].unsqueeze(0).unsqueeze(0)
            # idx_next = torch.argmax(probs, dim=-1).unsqueeze(0)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def pre(self, idx):
        # 预测分类
        logits, loss = self(idx)
        logits = logits[:, -1, :]
        output = torch.argmax(logits, -1)
        return output
