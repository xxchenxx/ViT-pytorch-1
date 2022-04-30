# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import numpy as np
# from models.custom_functions.sparse_matrix import SparseTensor

from models.custom_functions.custom_fc import LinearSparse
from models.custom_functions.custom_softmax import SoftmaxSparse
from models.custom_functions.custom_gelu import GELUSparse
from models.custom_functions.custom_matmul import MatMulSparse
from models.custom_functions.custom_softmax_matmul import SoftmaxMatMulSparse

from torch.nn import Dropout, Softmax, Linear


class MlpActPrune(nn.Module):
    def __init__(self, config, masker):
        super(MlpActPrune, self).__init__()

        self.fc1 = LinearSparse(config.hidden_size, config.transformer["mlp_dim"], quantize=config.quantize, masker=masker)
        self.fc2 = LinearSparse(config.transformer["mlp_dim"], config.hidden_size, quantize=config.quantize, masker=masker)
        self.act_fn = GELUSparse(quantize=config.quantize, masker=masker)
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class AttentionActPrune(nn.Module):
    def __init__(self, config, vis, masker):
        super(AttentionActPrune, self).__init__()
        self.vis = vis
        self.masker = masker
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = LinearSparse(config.hidden_size, self.all_head_size, quantize=config.quantize, masker=masker)
        self.key = LinearSparse(config.hidden_size, self.all_head_size, quantize=config.quantize, masker=masker)
        self.value = LinearSparse(config.hidden_size, self.all_head_size, quantize=config.quantize, masker=masker)

        self.out = LinearSparse(config.hidden_size, config.hidden_size, quantize=config.quantize, masker=masker)

        assert config.transformer["attention_dropout_rate"] == 0
        # self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.mm1 = MatMulSparse(quantize=config.quantize, masker=masker)
        self.softmax_mm2 = SoftmaxMatMulSparse(quantize=config.quantize, masker=masker, dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        #print(query_layer.shape)
        attention_scores = self.mm1(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        weights = None
        context_layer = self.softmax_mm2(attention_scores, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


def test_activation_prune():
    a = torch.rand(2, 100, 100)
    prune_ratio = 0.7
    mask = activation_prune(a, prune_ratio)
    print("prune ratio set is {}, real is {}".format(prune_ratio, 1 - mask.float().mean()))


if __name__ == "__main__":
    test_activation_prune()
