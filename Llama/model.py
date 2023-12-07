#!/usr/bin/env python
# coding: utf-8

from typing import Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F

device: str = 'cuda:4' if torch.cuda.is_available() else 'cpu'

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2]
    freqs = torch.outer(t, freqs).float()
    # torch.polar 的文档
    # https://pytorch.org/docs/stable/generated/torch.polar.html
    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


# PERO旋转位置嵌入
def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq.shape = [batch_size, seq_len, dim]
    # xq_.shape = [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2).to(device)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2).to(device)

    # 转为复数域
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)
    # 应用旋转操作，然后将结果转回实数域
    # xq_out.shape = [batch_size, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Head(nn.Module):
    """ 掩藏注意力 """
    freqs_cis = None

    def __init__(self, head_size, n_embedding,block_size, dropout=0.2, pos_embed_method="rope"):
        super().__init__()
        self.key = nn.Linear(n_embedding, head_size, bias=False)
        self.query = nn.Linear(n_embedding, head_size, bias=False)
        self.value = nn.Linear(n_embedding, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.pos_embed_method = pos_embed_method
        self.freqs_cis=precompute_freqs_cis(dim=head_size, seq_len=block_size).to(device)

    def forward(self, x):
        B, T, C = x.shape
        key = self.key(x)
        query = self.query(x)
        if self.pos_embed_method == 'rope':
            # Reformer相对位置编码
 #           if self.freqs_cis is None:
 #               self.freqs_cis = precompute_freqs_cis(dim=key.shape[-1], seq_len=T).to(key.device)
            xq, xk = apply_rotary_emb(key, query, self.freqs_cis)
            # key*value/(d**-0.5)
            query, key = xq, xk
        wei = key @ query.transpose(-2, -1) * (key.shape[-1] ** -0.5)
        # 掩藏操作，使注意力只能看到前面数据
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, -1)
        wei = self.dropout(wei)
        value = self.value(x)
        outputs = wei @ value
        return outputs


class MultiHeadAttention(nn.Module):
    """ 多头注意力 """

    def __init__(self, num_heads, head_size, block_size=256,n_embedding=360, pos_embed_method="rope"):
        super().__init__()
        # 多头注意力由多个注意力叠加
        self.heads = nn.ModuleList([Head(head_size, block_size=block_size,n_embedding=n_embedding,pos_embed_method=pos_embed_method) for _ in range(num_heads)])
        self.linear = nn.Linear(n_embedding, n_embedding)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # 拼接各个注意力的输出结果
        output = torch.cat([h(x) for h in self.heads], dim=-1)
        output = self.dropout(self.linear(output))
        return output


class RMSNorm(nn.Module):
    """ RMSNorm均方层归一化 """

    def __init__(self, n_emb, eps: float = 1e-6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_embedding = n_emb
        self.weights = nn.Parameter(torch.ones(n_emb))
        self.eps = eps

    def _norm(self, x):
        return x / torch.sqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        o = self._norm(x)
        return o * self.weights


def _Swish(x, beta=1):
    return x * (1 / 1 + torch.exp(-beta * x))


class SwishGLU(nn.Module):
    def __init__(self, dim, hidden_dim, beta=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)

    def forward(self, x):
        return _Swish(self.w1(x), self.beta) * self.w2(x)


class FeedForwardWithRELU(nn.Module):
    """ 前馈神经网络 """

    def __init__(self, dim: int, hidden_dim, dropout: float):
        super().__init__()
        self.w3 = nn.Linear(dim, hidden_dim)
        self.swish = nn.ReLU()
        self.w4 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w4(self.swish(self.w3(x))))


class FeedForward(nn.Module):
    """ 前馈神经网络 """

    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 4, dropout: float = 0.2, mode='swish'):
        super().__init__()
        self.mode = mode
        if self.mode == 'swish':
            # 4*(2*3*hidden_dim/4) 缩小为2*3倍
            hidden_dim = multiple_of * ((2 * hidden_dim // 3 + multiple_of - 1) // multiple_of)
            self.w3 = nn.Linear(hidden_dim, dim)
            self.swish = SwishGLU(dim, hidden_dim)
            self.dropout = nn.Dropout(dropout)
        elif self.mode == 'relu':
            self.relu = FeedForwardWithRELU(dim, hidden_dim, dropout)

    def forward(self, x):
        if self.mode == 'swish':
            return self.dropout(self.w3(self.swish(x)))
        elif self.mode == 'relu':
            return self.relu(x)


class Block(nn.Module):
    """
    feed_forward_mode "swish"-前馈神经网络使用swish激活函数，”relu“-使用relu激活函数
    norm ："RMS"-使用RMSNorm归一化，其他值则使用nn.LayerNorm
    """

    def __init__(self, n_emb, n_head, block_size,dropout=0.2, feed_forward_mode: str = "swish", norm="rms",
                 pos_embed_method="repo"):
        super().__init__()
        self.norm = norm
        self.swish = feed_forward_mode
        head_size = n_emb // n_head
        self.heads = MultiHeadAttention(n_head, head_size, block_size,n_emb,pos_embed_method=pos_embed_method)
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


class BlockOnlyPERO(nn.Module):
    def __init__(self, n_emb, n_head, block_size=256, dropout=0.2):
        super().__init__()
        head_size = n_emb // n_head
        self.heads = MultiHeadAttention(n_head, head_size,block_size=block_size,n_embedding=n_emb)
        self.fb = FeedForwardWithRELU(n_emb, n_emb, dropout)
        self.l1 = nn.LayerNorm(n_emb)
        self.l2 = nn.LayerNorm(n_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.l1(x)
        x = x + self.heads(x)
        x = self.l2(x)
        x = x + self.fb(x)
        return x


class LLama(nn.Module):
    #完整的Llama模型
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
            Block(n_embedding, n_heads,block_size, pos_embed_method=pos_embed_method,feed_forward_mode = feed_forward_mode, norm=norm),
            Block(n_embedding, n_heads,block_size, pos_embed_method=pos_embed_method,feed_forward_mode = feed_forward_mode, norm=norm),
            Block(n_embedding, n_heads,block_size, pos_embed_method=pos_embed_method,feed_forward_mode = feed_forward_mode, norm=norm),
            nn.Dropout(dropout)
        )
        #self.l1 = torch.nn.Linear(n_embedding, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        word_embedding = self.embedding(idx)
        x = word_embedding
        if self.pos_embed_method == "sin":
            pos_embedding = self.position_emb(torch.arange(0, T))
            pos_embedding = torch.repeat_interleave(pos_embedding, B)
            x = x + pos_embedding
        x = self.heads(x)
        return x
        # logits = self.l1(x)
        # loss = None
        # if targets is None:
        #     loss = None
        # else:
        #     B, T, C = logits.shape
        #     logits = logits.view(B * T, C)
        #     targets = targets.view(B * T)
        #     loss = torch.nn.functional.cross_entropy(logits, targets)
        # return logits, loss

#     def generate(self, idx, max_new_tokens, block_size=16):
#         p = 0
#         # 生成文本
#         for _ in range(max_new_tokens):
#             # get the predictions
#             idx_conv = idx[:, -block_size:]
#             logits, loss = self(idx_conv)
#             logits = logits[:, -1, :]
#             # apply softmax to get the probability
#             probs = torch.nn.functional.softmax(logits, dim=-1)
#             top_probs = torch.topk(probs, 5, dim=-1)
#             next_index = torch.multinomial(top_probs[0], num_samples=1)[0][0]
#             idx_next = top_probs[1][0][next_index.item()].unsqueeze(0).unsqueeze(0)
#             # idx_next = torch.argmax(probs, dim=-1).unsqueeze(0)
#             p = p + torch.log(probs[0][idx_next[0][0]])
#             idx = torch.cat((idx, idx_next), dim=1)
#         zhi_xin_du = torch.pow(torch.e, -p / max_new_tokens)
#         print(f"置信度：{zhi_xin_du}")
#         return idx


class LLama1(nn.Module):
    """
    pos_embed_method 限制取值 "rope"|"sin"
    """

    def __init__(self, vocab_size, out_features=2, n_heads=4,
                 n_embedding=360, block_size=200, dropout=0.2,
                 feed_forward_mode: str = "swish", norm="rms",
                 pos_embed_method="repo"):
        super().__init__()
        # self.vocab_size = vocab_size
        # self.out_features = out_features
        # self.n_heads = n_heads
        # self.n_embedding = n_embedding
        # self.pos_embed_method = pos_embed_method
        # self.block_size = block_size
        # self.embedding = torch.nn.Embedding(vocab_size, n_embedding)
        # if pos_embed_method != "rope":
        #     self.position_emb = nn.Embedding(block_size, n_embedding)
        # self.heads = nn.Sequential(
        #     Block(n_embedding, n_heads,block_size, pos_embed_method=pos_embed_method),
        #     Block(n_embedding, n_heads,block_size, pos_embed_method=pos_embed_method),
        #     Block(n_embedding, n_heads,block_size, pos_embed_method=pos_embed_method),
        #     nn.Dropout(dropout)
        # )
        self.l1 = torch.nn.Linear(n_embedding, out_features)
        self.llama = LLama(vocab_size, out_features, n_heads,
                 n_embedding, block_size, dropout,
                 feed_forward_mode,norm,
                 pos_embed_method)

    def forward(self, idx, targets=None):
        # B, T = idx
        # word_embedding = self.embedding(idx)
        # x = word_embedding
        # # pos_embed_method=="sin" ，使用sinmoid位置编码
        # if self.pos_embed_method == "sin":
        #     pos_embedding = self.position_emb(torch.arange(0, T))
        #     pos_embedding = torch.repeat_interleave(pos_embedding, B)
        #     x = x + pos_embedding
        # x = self.heads(x)
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


class LLamaOnlyPERO(nn.Module):
    def __init__(self, vocab_size,out_features=2,block_size=256, n_embedding=360,n_heads=4, dropout=0.2):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, n_embedding)
        self.heads = nn.Sequential(
            BlockOnlyPERO(n_embedding, n_heads,block_size),
            BlockOnlyPERO(n_embedding, n_heads,block_size),
            BlockOnlyPERO(n_embedding, n_heads,block_size),
            nn.Dropout(dropout)
        )
        self.l1 = torch.nn.Linear(n_embedding, out_features)

    def forward(self, idx, targets=None):
        word_embedding = self.embedding(idx)
        x = word_embedding
        x = self.heads(x)
        logit = self.l1(x)
        loss = None
        if targets is None:
            loss = None
        else:
            B, T, C = logit.shape
            logit = logit[:, -1, :]
            logit = logit.view(B, C)
            targets = targets.view(B)
            loss = torch.nn.functional.cross_entropy(logit, targets)
        return logit, loss

    def pre(self, idx):
        logit, loss = self(idx)
        logit = logit[:, -1, :]
        output = torch.argmax(logit, -1)
        return output














# #!/usr/bin/env python
# # coding: utf-8

# from typing import Tuple, Union
# import torch
# from torch import nn
# import torch.nn.functional as F

# device: str = 'cuda:5' if torch.cuda.is_available() else 'cpu'

# def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
#     # 计算词向量元素两两分组之后，每组元素对应的旋转角度
#     freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
#     # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
#     t = torch.arange(seq_len, device=freqs.device)
#     # freqs.shape = [seq_len, dim // 2]
#     freqs = torch.outer(t, freqs).float()
#     # torch.polar 的文档
#     # https://pytorch.org/docs/stable/generated/torch.polar.html
#     # 计算结果是个复数向量
#     # 假设 freqs = [x, y]
#     # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
#     freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
#     return freqs_cis


# # PERO旋转位置嵌入
# def apply_rotary_emb(
#         xq: torch.Tensor,
#         xk: torch.Tensor,
#         freqs_cis: torch.Tensor,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     # xq.shape = [batch_size, seq_len, dim]
#     # xq_.shape = [batch_size, seq_len, dim // 2, 2]
#     xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2).to(device)
#     xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2).to(device)

#     # 转为复数域
#     xq_ = torch.view_as_complex(xq_)
#     xk_ = torch.view_as_complex(xk_)
#     # 应用旋转操作，然后将结果转回实数域
#     # xq_out.shape = [batch_size, seq_len, dim]
#     xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
#     xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
#     return xq_out.type_as(xq), xk_out.type_as(xk)


# class Head(nn.Module):
#     """ 掩藏注意力 """
#     freqs_cis = None

#     def __init__(self, head_size, n_embedding,block_size, dropout=0.2, pos_embed_method="rope"):
#         super().__init__()
#         self.key = nn.Linear(n_embedding, head_size, bias=False)
#         self.query = nn.Linear(n_embedding, head_size, bias=False)
#         self.value = nn.Linear(n_embedding, head_size, bias=False)
#         self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
#         self.dropout = nn.Dropout(dropout)
#         self.pos_embed_method = pos_embed_method
#         self.freqs_cis=precompute_freqs_cis(dim=head_size, seq_len=block_size).to(device)

#     def forward(self, x):
#         B, T, C = x.shape
#         key = self.key(x)
#         query = self.query(x)
#         if self.pos_embed_method == 'rope':
#             # Reformer相对位置编码
#  #           if self.freqs_cis is None:
#  #               self.freqs_cis = precompute_freqs_cis(dim=key.shape[-1], seq_len=T).to(key.device)
#             xq, xk = apply_rotary_emb(key, query, self.freqs_cis)
#             # key*value/(d**-0.5)
#             query, key = xq, xk
#         wei = key @ query.transpose(-2, -1) * (key.shape[-1] ** -0.5)
#         # 掩藏操作，使注意力只能看到前面数据
#         wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
#         wei = F.softmax(wei, -1)
#         wei = self.dropout(wei)
#         value = self.value(x)
#         outputs = wei @ value
#         return outputs


# class MultiHeadAttention(nn.Module):
#     """ 多头注意力 """

#     def __init__(self, num_heads, head_size, block_size=256,n_embedding=360, pos_embed_method="rope"):
#         super().__init__()
#         # 多头注意力由多个注意力叠加
#         self.heads = nn.ModuleList([Head(head_size, block_size=block_size,n_embedding=n_embedding,pos_embed_method=pos_embed_method) for _ in range(num_heads)])
#         self.linear = nn.Linear(n_embedding, n_embedding)
#         self.dropout = nn.Dropout(0.2)

#     def forward(self, x):
#         # 拼接各个注意力的输出结果
#         output = torch.cat([h(x) for h in self.heads], dim=-1)
#         output = self.dropout(self.linear(output))
#         return output


# class RMSNorm(nn.Module):
#     """ RMSNorm均方层归一化 """

#     def __init__(self, n_emb, eps: float = 1e-6, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.n_embedding = n_emb
#         self.weights = nn.Parameter(torch.ones(n_emb))
#         self.eps = eps

#     def _norm(self, x):
#         return x / torch.sqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + self.eps)

#     def forward(self, x: torch.Tensor):
#         o = self._norm(x)
#         return o * self.weights


# def _Swish(x, beta=1):
#     return x * (1 / 1 + torch.exp(-beta * x))


# class SwishGLU(nn.Module):
#     def __init__(self, dim, hidden_dim, beta=1, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.beta = beta
#         self.w1 = nn.Linear(dim, hidden_dim)
#         self.w2 = nn.Linear(dim, hidden_dim)

#     def forward(self, x):
#         return _Swish(self.w1(x), self.beta) * self.w2(x)


# class FeedForwardWithRELU(nn.Module):
#     """ 前馈神经网络 """

#     def __init__(self, dim: int, hidden_dim, dropout: float):
#         super().__init__()
#         self.w3 = nn.Linear(dim, hidden_dim)
#         self.swish = nn.ReLU()
#         self.w4 = nn.Linear(hidden_dim, dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         return self.dropout(self.w4(self.swish(self.w3(x))))


# class FeedForward(nn.Module):
#     """ 前馈神经网络 """

#     def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 4, dropout: float = 0.2, mode='swish'):
#         super().__init__()
#         self.mode = mode
#         if self.mode == 'swish':
#             # 4*(2*3*hidden_dim/4) 缩小为2*3倍
#             hidden_dim = multiple_of * ((2 * hidden_dim // 3 + multiple_of - 1) // multiple_of)
#             self.w3 = nn.Linear(hidden_dim, dim)
#             self.swish = SwishGLU(dim, hidden_dim)
#             self.dropout = nn.Dropout(dropout)
#         elif self.mode == 'relu':
#             self.relu = FeedForwardWithRELU(dim, hidden_dim, dropout)

#     def forward(self, x):
#         if self.mode == 'swish':
#             return self.dropout(self.w3(self.swish(x)))
#         elif self.mode == 'relu':
#             return self.relu(x)


# class Block(nn.Module):
#     """
#     feed_forward_mode "swish"-前馈神经网络使用swish激活函数，”relu“-使用relu激活函数
#     norm ："RMS"-使用RMSNorm归一化，其他值则使用nn.LayerNorm
#     """

#     def __init__(self, n_emb, n_head, block_size,dropout=0.2, feed_forward_mode: str = "swish", norm="rms",
#                  pos_embed_method="repo"):
#         super().__init__()
#         self.norm = norm
#         self.swish = feed_forward_mode
#         head_size = n_emb // n_head
#         self.heads = MultiHeadAttention(n_head, head_size, block_size,n_emb,pos_embed_method=pos_embed_method)
#         self.fb = FeedForward(n_emb, 2 * n_emb, 2, dropout, mode=feed_forward_mode)
#         if norm == "rms":
#             self.l1 = RMSNorm(n_emb)
#             self.l2 = RMSNorm(n_emb)
#         else:
#             self.l1 = nn.LayerNorm(n_emb)
#             self.l2 = nn.LayerNorm(n_emb)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         if self.norm == "rms":
#             x = self.l1(x)
#             x = x + self.heads(x)
#             x = self.l2(x)
#             x = x + self.fb(x)
#         else:
#             x = x + self.heads(x)
#             x = self.l1(x)
#             x = x + self.fb(x)
#             x = self.l2(x)
#         return x


# class BlockOnlyPERO(nn.Module):
#     def __init__(self, n_emb, n_head, block_size=256, dropout=0.2):
#         super().__init__()
#         head_size = n_emb // n_head
#         self.heads = MultiHeadAttention(n_head, head_size,block_size=block_size,n_embedding=n_emb)
#         self.fb = FeedForwardWithRELU(n_emb, n_emb, dropout)
#         self.l1 = nn.LayerNorm(n_emb)
#         self.l2 = nn.LayerNorm(n_emb)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         x = self.l1(x)
#         x = x + self.heads(x)
#         x = self.l2(x)
#         x = x + self.fb(x)
#         return x


# class LLama(nn.Module):
#     def __init__(self, vocab_size, block_size=256,n_embedding=360,n_heads=4,dropout=0.2):
#         super().__init__()
#         self.embedding = torch.nn.Embedding(vocab_size, n_embedding)
#         # self.position_emb = nn.Embedding(block_size, n_embedding)
#         self.heads = nn.Sequential(
#             Block(n_embedding, n_heads,block_size,n_embedding),
#             Block(n_embedding, n_heads,block_size,n_embedding),
#             Block(n_embedding, n_heads,block_size,n_embedding),
#             nn.Dropout(dropout)
#         )
#         self.l1 = torch.nn.Linear(n_embedding, vocab_size)

#     def forward(self, idx, targets=None):
#         B, T = idx.shape
#         word_embedding = self.embedding(idx)
#         x = word_embedding
#         x = self.heads(x)
#         logits = self.l1(x)
#         loss = None
#         if targets is None:
#             loss = None
#         else:
#             B, T, C = logits.shape
#             logits = logits.view(B * T, C)
#             targets = targets.view(B * T)
#             loss = torch.nn.functional.cross_entropy(logits, targets)
#         return logits, loss

#     def generate(self, idx, max_new_tokens, block_size=16):
#         p = 0
#         # 生成文本
#         for _ in range(max_new_tokens):
#             # get the predictions
#             idx_conv = idx[:, -block_size:]
#             logits, loss = self(idx_conv)
#             logits = logits[:, -1, :]
#             # apply softmax to get the probability
#             probs = torch.nn.functional.softmax(logits, dim=-1)
#             top_probs = torch.topk(probs, 5, dim=-1)
#             next_index = torch.multinomial(top_probs[0], num_samples=1)[0][0]
#             idx_next = top_probs[1][0][next_index.item()].unsqueeze(0).unsqueeze(0)
#             # idx_next = torch.argmax(probs, dim=-1).unsqueeze(0)
#             p = p + torch.log(probs[0][idx_next[0][0]])
#             idx = torch.cat((idx, idx_next), dim=1)
#         zhi_xin_du = torch.pow(torch.e, -p / max_new_tokens)
#         print(f"置信度：{zhi_xin_du}")
#         return idx


# class LLama1(nn.Module):
#     """
#     pos_embed_method 限制取值 "rope"|"sin"
#     """

#     def __init__(self, vocab_size, out_features=2, n_heads=4,
#                  n_embedding=360, block_size=200, dropout=0.2,
#                  pos_embed_method="rope"):
#         super().__init__()
#         self.vocab_size = vocab_size
#         self.out_features = out_features
#         self.n_heads = n_heads
#         self.n_embedding = n_embedding
#         self.pos_embed_method = pos_embed_method
#         self.block_size = block_size
#         self.embedding = torch.nn.Embedding(vocab_size, n_embedding)
#         if pos_embed_method != "rope":
#             self.position_emb = nn.Embedding(block_size, n_embedding)
#         self.heads = nn.Sequential(
#             Block(n_embedding, n_heads,block_size, pos_embed_method=pos_embed_method),
#             Block(n_embedding, n_heads,block_size, pos_embed_method=pos_embed_method),
#             Block(n_embedding, n_heads,block_size, pos_embed_method=pos_embed_method),
#             nn.Dropout(dropout)
#         )
#         self.l1 = torch.nn.Linear(n_embedding, out_features)

#     def forward(self, idx, targets=None):
#         B, T = idx
#         word_embedding = self.embedding(idx)
#         x = word_embedding
#         # pos_embed_method=="sin" ，使用sinmoid位置编码
#         if self.pos_embed_method == "sin":
#             pos_embedding = self.position_emb(torch.arange(0, T))
#             pos_embedding = torch.repeat_interleave(pos_embedding, B)
#             x = x + pos_embedding
#         x = self.heads(x)
#         logits = self.l1(x)
#         loss = None
#         if targets is None:
#             loss = None
#         else:
#             B, T, C = logits.shape
#             logits = logits[:, -1, :]
#             logits = logits.view(B, C)
#             targets = targets.view(B)
#             loss = torch.nn.functional.cross_entropy(logits, targets)
#         return logits, loss

#     def generate(self, idx, max_new_tokens, block_size=16):
#         # 生成文本
#         for _ in range(max_new_tokens):
#             # get the predictions
#             idx_conv = idx[:, -block_size:]
#             logits, loss = self(idx_conv)
#             logits = logits[:, -1, :]
#             # apply softmax to get the probability
#             probs = torch.nn.functional.softmax(logits, dim=-1)
#             top_probs = torch.topk(probs, 5, dim=-1)
#             next_index = torch.multinomial(top_probs[0], num_samples=1)[0][0]
#             idx_next = top_probs[1][0][next_index.item()].unsqueeze(0).unsqueeze(0)
#             # idx_next = torch.argmax(probs, dim=-1).unsqueeze(0)
#             idx = torch.cat((idx, idx_next), dim=1)
#         return idx

#     def pre(self, idx):
#         # 预测分类
#         logits, loss = self(idx)
#         logits = logits[:, -1, :]
#         output = torch.argmax(logits, -1)
#         return output


# class LLamaOnlyPERO(nn.Module):
#     def __init__(self, vocab_size,out_features=2,block_size=256, n_embedding=360,n_heads=4, dropout=0.2):
#         super().__init__()
#         self.embedding = torch.nn.Embedding(vocab_size, n_embedding)
#         self.heads = nn.Sequential(
#             BlockOnlyPERO(n_embedding, n_heads,block_size),
#             BlockOnlyPERO(n_embedding, n_heads,block_size),
#             BlockOnlyPERO(n_embedding, n_heads,block_size),
#             nn.Dropout(dropout)
#         )
#         self.l1 = torch.nn.Linear(n_embedding, out_features)

#     def forward(self, idx, targets=None):
#         word_embedding = self.embedding(idx)
#         x = word_embedding
#         x = self.heads(x)
#         logit = self.l1(x)
#         loss = None
#         if targets is None:
#             loss = None
#         else:
#             B, T, C = logit.shape
#             logit = logit[:, -1, :]
#             logit = logit.view(B, C)
#             targets = targets.view(B)
#             loss = torch.nn.functional.cross_entropy(logit, targets)
#         return logit, loss

#     def pre(self, idx):
#         logit, loss = self(idx)
#         logit = logit[:, -1, :]
#         output = torch.argmax(logit, -1)
#         return output











# #!/usr/bin/env python
# # coding: utf-8

# from typing import Tuple, Union
# import torch
# from torch import nn
# import torch.nn.functional as F

# device: str = 'cuda:5' if torch.cuda.is_available() else 'cpu'

# def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
#     # 计算词向量元素两两分组之后，每组元素对应的旋转角度
#     freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
#     # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
#     t = torch.arange(seq_len, device=freqs.device)
#     # freqs.shape = [seq_len, dim // 2]
#     freqs = torch.outer(t, freqs).float()
#     # torch.polar 的文档
#     # https://pytorch.org/docs/stable/generated/torch.polar.html
#     # 计算结果是个复数向量
#     # 假设 freqs = [x, y]
#     # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
#     freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
#     return freqs_cis


# # PERO旋转位置嵌入
# def apply_rotary_emb(
#         xq: torch.Tensor,
#         xk: torch.Tensor,
#         freqs_cis: torch.Tensor,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     # xq.shape = [batch_size, seq_len, dim]
#     # xq_.shape = [batch_size, seq_len, dim // 2, 2]
#     xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2).to(device)
#     xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2).to(device)

#     # 转为复数域
#     xq_ = torch.view_as_complex(xq_)
#     xk_ = torch.view_as_complex(xk_)
#     # 应用旋转操作，然后将结果转回实数域
#     # xq_out.shape = [batch_size, seq_len, dim]
#     xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
#     xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
#     return xq_out.type_as(xq), xk_out.type_as(xk)


# class Head(nn.Module):
#     """ 掩藏注意力 """
#     freqs_cis = None

#     def __init__(self, head_size, n_embedding,block_size, dropout=0.2, pos_embed_method="rope"):
#         super().__init__()
#         self.key = nn.Linear(n_embedding, head_size, bias=False)
#         self.query = nn.Linear(n_embedding, head_size, bias=False)
#         self.value = nn.Linear(n_embedding, head_size, bias=False)
#         self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
#         self.dropout = nn.Dropout(dropout)
#         self.pos_embed_method = pos_embed_method
#         self.freqs_cis=precompute_freqs_cis(dim=head_size, seq_len=block_size).to(device)

#     def forward(self, x):
#         B, T, C = x.shape
#         key = self.key(x)
#         query = self.query(x)
#         if self.pos_embed_method == 'rope':
#             # Reformer相对位置编码
#  #           if self.freqs_cis is None:
#  #               self.freqs_cis = precompute_freqs_cis(dim=key.shape[-1], seq_len=T).to(key.device)
#             xq, xk = apply_rotary_emb(key, query, self.freqs_cis)
#             # key*value/(d**-0.5)
#             query, key = xq, xk
#         wei = key @ query.transpose(-2, -1) * (key.shape[-1] ** -0.5)
#         # 掩藏操作，使注意力只能看到前面数据
#         wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
#         wei = F.softmax(wei, -1)
#         wei = self.dropout(wei)
#         value = self.value(x)
#         outputs = wei @ value
#         return outputs


# class MultiHeadAttention(nn.Module):
#     """ 多头注意力 """

#     def __init__(self, num_heads, head_size, block_size=256,n_embedding=360, pos_embed_method="rope"):
#         super().__init__()
#         # 多头注意力由多个注意力叠加
#         self.heads = nn.ModuleList([Head(head_size, block_size=block_size,n_embedding=n_embedding,pos_embed_method=pos_embed_method) for _ in range(num_heads)])
#         self.linear = nn.Linear(n_embedding, n_embedding)
#         self.dropout = nn.Dropout(0.2)

#     def forward(self, x):
#         # 拼接各个注意力的输出结果
#         output = torch.cat([h(x) for h in self.heads], dim=-1)
#         output = self.dropout(self.linear(output))
#         return output


# class RMSNorm(nn.Module):
#     """ RMSNorm均方层归一化 """

#     def __init__(self, n_emb, eps: float = 1e-6, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.n_embedding = n_emb
#         self.weights = nn.Parameter(torch.ones(n_emb))
#         self.eps = eps

#     def _norm(self, x):
#         return x / torch.sqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + self.eps)

#     def forward(self, x: torch.Tensor):
#         o = self._norm(x)
#         return o * self.weights


# def _Swish(x, beta=1):
#     return x * (1 / 1 + torch.exp(-beta * x))


# class SwishGLU(nn.Module):
#     def __init__(self, dim, hidden_dim, beta=1, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.beta = beta
#         self.w1 = nn.Linear(dim, hidden_dim)
#         self.w2 = nn.Linear(dim, hidden_dim)

#     def forward(self, x):
#         return _Swish(self.w1(x), self.beta) * self.w2(x)


# class FeedForwardWithRELU(nn.Module):
#     """ 前馈神经网络 """

#     def __init__(self, dim: int, hidden_dim, dropout: float):
#         super().__init__()
#         self.w3 = nn.Linear(dim, hidden_dim)
#         self.swish = nn.ReLU()
#         self.w4 = nn.Linear(hidden_dim, dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         return self.dropout(self.w4(self.swish(self.w3(x))))


# class FeedForward(nn.Module):
#     """ 前馈神经网络 """

#     def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 4, dropout: float = 0.2, mode='swish'):
#         super().__init__()
#         self.mode = mode
#         if self.mode == 'swish':
#             # 4*(2*3*hidden_dim/4) 缩小为2*3倍
#             hidden_dim = multiple_of * ((2 * hidden_dim // 3 + multiple_of - 1) // multiple_of)
#             self.w3 = nn.Linear(hidden_dim, dim)
#             self.swish = SwishGLU(dim, hidden_dim)
#             self.dropout = nn.Dropout(dropout)
#         elif self.mode == 'relu':
#             self.relu = FeedForwardWithRELU(dim, hidden_dim, dropout)

#     def forward(self, x):
#         if self.mode == 'swish':
#             return self.dropout(self.w3(self.swish(x)))
#         elif self.mode == 'relu':
#             return self.relu(x)


# class Block(nn.Module):
#     """
#     feed_forward_mode "swish"-前馈神经网络使用swish激活函数，”relu“-使用relu激活函数
#     norm ："RMS"-使用RMSNorm归一化，其他值则使用nn.LayerNorm
#     """

#     def __init__(self, n_emb, n_head, block_size,embedding_size,,dropout=0.2, feed_forward_mode: str = "swish", norm="rms",
#                  pos_embed_method="repo"):
#         super().__init__()
#         self.norm = norm
#         self.swish = feed_forward_mode
#         head_size = n_emb // n_head
#         self.heads = MultiHeadAttention(n_head, head_size, block_size,embedding_size,pos_embed_method=pos_embed_method)
#         self.fb = FeedForward(n_emb, 2 * n_emb, 2, dropout, mode=feed_forward_mode)
#         if norm == "rms":
#             self.l1 = RMSNorm(n_emb)
#             self.l2 = RMSNorm(n_emb)
#         else:
#             self.l1 = nn.LayerNorm(n_emb)
#             self.l2 = nn.LayerNorm(n_emb)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         if self.norm == "rms":
#             x = self.l1(x)
#             x = x + self.heads(x)
#             x = self.l2(x)
#             x = x + self.fb(x)
#         else:
#             x = x + self.heads(x)
#             x = self.l1(x)
#             x = x + self.fb(x)
#             x = self.l2(x)
#         return x


# class BlockOnlyPERO(nn.Module):
#     def __init__(self, n_emb, n_head, block_size=256, n_embedding=360, dropout=0.2):
#         super().__init__()
#         head_size = n_emb // n_head
#         self.heads = MultiHeadAttention(n_head, head_size,block_size=block_size,n_embedding=n_embedding, block_size=block_size)
#         self.fb = FeedForwardWithRELU(n_emb, n_emb, dropout)
#         self.l1 = nn.LayerNorm(n_emb)
#         self.l2 = nn.LayerNorm(n_emb)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         x = self.l1(x)
#         x = x + self.heads(x)
#         x = self.l2(x)
#         x = x + self.fb(x)
#         return x


# class LLama(nn.Module):
#     def __init__(self, vocab_size, block_size=256,n_embedding=360, dropout=0.2):
#         super().__init__()
#         self.embedding = torch.nn.Embedding(vocab_size, n_embedding)
#         # self.position_emb = nn.Embedding(block_size, n_embedding)
#         self.heads = nn.Sequential(
#             Block(n_embedding, n_head=4,block_size,n_embedding),
#             Block(n_embedding, n_head=4,block_size,n_embedding),
#             Block(n_embedding, n_head=4,block_size,n_embedding),
#             nn.Dropout(dropout)
#         )
#         self.l1 = torch.nn.Linear(n_embedding, vocab_size)

#     def forward(self, idx, targets=None):
#         B, T = idx.shape
#         word_embedding = self.embedding(idx)
#         x = word_embedding
#         x = self.heads(x)
#         logits = self.l1(x)
#         loss = None
#         if targets is None:
#             loss = None
#         else:
#             B, T, C = logits.shape
#             logits = logits.view(B * T, C)
#             targets = targets.view(B * T)
#             loss = torch.nn.functional.cross_entropy(logits, targets)
#         return logits, loss

#     def generate(self, idx, max_new_tokens, block_size=16):
#         p = 0
#         # 生成文本
#         for _ in range(max_new_tokens):
#             # get the predictions
#             idx_conv = idx[:, -block_size:]
#             logits, loss = self(idx_conv)
#             logits = logits[:, -1, :]
#             # apply softmax to get the probability
#             probs = torch.nn.functional.softmax(logits, dim=-1)
#             top_probs = torch.topk(probs, 5, dim=-1)
#             next_index = torch.multinomial(top_probs[0], num_samples=1)[0][0]
#             idx_next = top_probs[1][0][next_index.item()].unsqueeze(0).unsqueeze(0)
#             # idx_next = torch.argmax(probs, dim=-1).unsqueeze(0)
#             p = p + torch.log(probs[0][idx_next[0][0]])
#             idx = torch.cat((idx, idx_next), dim=1)
#         zhi_xin_du = torch.pow(torch.e, -p / max_new_tokens)
#         print(f"置信度：{zhi_xin_du}")
#         return idx


# class LLama1(nn.Module):
#     """
#     pos_embed_method 限制取值 "rope"|"sin"
#     """

#     def __init__(self, vocab_size, out_features=2, n_heads=4,
#                  n_embedding=360, block_size=200, dropout=0.2,
#                  pos_embed_method="rope"):
#         super().__init__()
#         self.vocab_size = vocab_size
#         self.out_features = out_features
#         self.n_heads = n_heads
#         self.n_embedding = n_embedding
#         self.pos_embed_method = pos_embed_method
#         self.block_size = block_size
#         self.embedding = torch.nn.Embedding(vocab_size, n_embedding)
#         if pos_embed_method != "rope":
#             self.position_emb = nn.Embedding(block_size, n_embedding)
#         self.heads = nn.Sequential(
#             Block(n_embedding, n_head=n_heads,block_size,n_embedding, pos_embed_method=pos_embed_method),
#             Block(n_embedding, n_head=n_heads,block_size,n_embedding, pos_embed_method=pos_embed_method),
#             Block(n_embedding, n_head=n_heads,block_size,n_embedding, pos_embed_method=pos_embed_method),
#             nn.Dropout(dropout)
#         )
#         self.l1 = torch.nn.Linear(n_embedding, out_features)

#     def forward(self, idx, targets=None):
#         B, T = idx
#         word_embedding = self.embedding(idx)
#         x = word_embedding
#         # pos_embed_method=="sin" ，使用sinmoid位置编码
#         if self.pos_embed_method == "sin":
#             pos_embedding = self.position_emb(torch.arange(0, T))
#             pos_embedding = torch.repeat_interleave(pos_embedding, B)
#             x = x + pos_embedding
#         x = self.heads(x)
#         logits = self.l1(x)
#         loss = None
#         if targets is None:
#             loss = None
#         else:
#             B, T, C = logits.shape
#             logits = logits[:, -1, :]
#             logits = logits.view(B, C)
#             targets = targets.view(B)
#             loss = torch.nn.functional.cross_entropy(logits, targets)
#         return logits, loss

#     def generate(self, idx, max_new_tokens, block_size=16):
#         # 生成文本
#         for _ in range(max_new_tokens):
#             # get the predictions
#             idx_conv = idx[:, -block_size:]
#             logits, loss = self(idx_conv)
#             logits = logits[:, -1, :]
#             # apply softmax to get the probability
#             probs = torch.nn.functional.softmax(logits, dim=-1)
#             top_probs = torch.topk(probs, 5, dim=-1)
#             next_index = torch.multinomial(top_probs[0], num_samples=1)[0][0]
#             idx_next = top_probs[1][0][next_index.item()].unsqueeze(0).unsqueeze(0)
#             # idx_next = torch.argmax(probs, dim=-1).unsqueeze(0)
#             idx = torch.cat((idx, idx_next), dim=1)
#         return idx

#     def pre(self, idx):
#         # 预测分类
#         logits, loss = self(idx)
#         logits = logits[:, -1, :]
#         output = torch.argmax(logits, -1)
#         return output


# class LLamaOnlyPERO(nn.Module):
#     def __init__(self, vocab_size,out_features=2,block_size=256, n_embedding=360, dropout=0.2):
#         super().__init__()
#         self.embedding = torch.nn.Embedding(vocab_size, n_embedding)
#         self.heads = nn.Sequential(
#             BlockOnlyPERO(n_embedding, n_head=4,block_size,n_embedding),
#             BlockOnlyPERO(n_embedding, n_head=4,block_size,n_embedding),
#             BlockOnlyPERO(n_embedding, n_head=4,block_size,n_embedding),
#             nn.Dropout(dropout)
#         )
#         self.l1 = torch.nn.Linear(n_embedding, out_features)

#     def forward(self, idx, targets=None):
#         word_embedding = self.embedding(idx)
#         x = word_embedding
#         x = self.heads(x)
#         logit = self.l1(x)
#         loss = None
#         if targets is None:
#             loss = None
#         else:
#             B, T, C = logit.shape
#             logit = logit[:, -1, :]
#             logit = logit.view(B, C)
#             targets = targets.view(B)
#             loss = torch.nn.functional.cross_entropy(logit, targets)
#         return logit, loss

#     def pre(self, idx):
#         logit, loss = self(idx)
#         logit = logit[:, -1, :]
#         output = torch.argmax(logit, -1)
#         return output
