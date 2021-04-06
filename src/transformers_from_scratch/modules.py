import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ScaledDotProductAttention(nn.Module):
    '''Scaled Dot-Product Attention'''

    def __init__(self, d_k: float):
        super().__init__()
        self.d_k: float = d_k

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, mask=None
    ) -> Tuple[Tensor, Tensor]:
        '''
        Args:
            q, k, v (Tensor): (batch_size, nun_sequences, embedding_dim).
        Returns:
            output (Tensor): Output of this layer. (batch, num_sequences, embed_dim)
            weights (Tensor): Attention weights. (batch, num_sequences, embed_dim)
        '''
        # -> (batch, num_seq, num_seq)
        weights = torch.matmul(q / self.d_k, k.transpose(1, 2))

        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)

        weights = F.softmax(weights, dim=2)
        # -> (batch, num_seq, embed_dim)
        output = torch.matmul(weights, v)
        return output, weights


if __name__ == '__main__':
    batch_size: int = 2
    num_sequences: int = 128
    d_k: float = math.sqrt(num_sequences)
    embed_dim: int = 256

    q = torch.zeros((batch_size, num_sequences, embed_dim))
    k = torch.zeros((batch_size, num_sequences, embed_dim))
    v = torch.zeros((batch_size, num_sequences, embed_dim))

    attention = ScaledDotProductAttention(d_k)
    out, attn = attention(q, k, v)
    print(out.size(), attn.size())
