import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        """前向传播.
            Args:
                q: Queries张量，形状为[B, L_q, D_q]
                k: Keys张量，形状为[B, L_k, D_k]
                v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
                mask: Masking张量，形状为[B, L_q, L_k]

            Returns:
                上下文张量和attention张量
        """
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            # 给需要mask的地方设置一个负无穷 将 mask中为1的 元素所在的索引，在a中相同的的索引处替换为 value
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn
