import torch
from torch import nn


class conv_low_rank(nn.Conv2d):
    def __init__(self, token_len=10, num_heads=12, **kwargs):
        super(conv_low_rank, self).__init__(**kwargs)
        self.low_rank = nn.Parameter(torch.zeros(num_heads, token_len), requires_grad=True)
