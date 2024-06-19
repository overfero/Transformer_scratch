import math

import torch
import torch.nn as nn


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.h = h
        self.d_model = d_model
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(q, k, v, mask, dropout: nn.Dropout):
        d_k = q.shape[-1]

        att_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            att_score = att_score.masked_fill_(mask == 0, -1e-9)
        att_score = att_score.softmax(dim=-1)

        if dropout is not None:
            att_score = dropout(att_score)
        return att_score.matmul(v), att_score

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        x, att_score = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(x.shape[0], x.shape[1], self.h * self.d_k)
        )

        return self.w_o(x)
