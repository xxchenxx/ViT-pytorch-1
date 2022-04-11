# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm

from pdb import set_trace
@torch.no_grad()
def activation_prune(activation, prune_ratio):
    num_small = int(np.clip(activation[0].numel() * prune_ratio, 1, activation[0].numel()))
    activation_mag = torch.abs(activation)
    threshold, _ = torch.kthvalue(activation_mag.flatten(1), num_small)
    while len(threshold.shape) < len(activation_mag.shape):
        threshold = threshold.unsqueeze(-1)
    mask = activation_mag >= threshold
    return mask


class SoftmaxActivationPrune(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_dense, prune_ratio_attn_mat_store=0):
        dense_out = Softmax(dim=-1)(x_dense)

        mask = activation_prune(dense_out, prune_ratio_attn_mat_store)
        sparse_out = mask * dense_out
        # print("attn prune ratio: {}".format(1 - mask.float().mean()))

        ctx.sparse_out = sparse_out
        # save sparse activation, but forward with dense
        return dense_out, sparse_out

    @staticmethod
    def backward(ctx, grad_in, *args):
        shape_A = ctx.sparse_out.shape
        unsqueeze_cnt = len(shape_A) - 1
        eye = torch.eye(shape_A[-1]).to(ctx.sparse_out.device)
        for _ in range(unsqueeze_cnt):
            eye = eye.unsqueeze(0)
        A = ctx.sparse_out

        # grad_Softmax = (A * (1 - A)).unsqueeze(-1) * eye - (A.unsqueeze(-1) * A.unsqueeze(-2)) * (1 - eye)
        # grad_out = (grad_in.unsqueeze(-2) @ grad_Softmax).squeeze(-2)

        # merge togather
        # set_trace()
        grad_out = grad_in * A - ((grad_in.unsqueeze(-2) @ A.unsqueeze(-1)) @ A.unsqueeze(-2)).squeeze(-2)

        return grad_out, None


class AttentionStoreActivationPrune(nn.Module):
    def __init__(self, config, vis, prune_ratio_attn_mat_store=0, prune_ratio_act_store=0):
        super(AttentionStoreActivationPrune, self).__init__()
        self.vis = vis
        # prune ratio for stored attn matrix
        self.prune_ratio_attn_mat_store = prune_ratio_attn_mat_store
        # prune ratio for query, key, value, input activation matrix
        self.prune_ratio_act_store = prune_ratio_act_store
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        # self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        # magnitude based prune activation
        mask = activation_prune(hidden_states, self.prune_ratio_act_store)
        hidden_states_prune = mask.detach() * hidden_states

        # dense activate forward
        with torch.no_grad():
            mixed_query_layer = self.query(hidden_states).detach()
            mixed_key_layer = self.key(hidden_states).detach()
            mixed_value_layer = self.value(hidden_states).detach()

        # pruned activate backward
        mixed_query_layer += self.query(hidden_states_prune) - self.query(hidden_states_prune).detach()
        mixed_key_layer += self.key(hidden_states_prune) - self.key(hidden_states_prune).detach()
        mixed_value_layer += self.value(hidden_states_prune) - self.value(hidden_states_prune).detach()

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        #print(query_layer.shape)

        # magnitude based prune activation
        with torch.no_grad():
            mask_query = activation_prune(query_layer, self.prune_ratio_act_store)
            mask_key = activation_prune(key_layer, self.prune_ratio_act_store)

        query_layer_prune = mask_query.detach() * query_layer
        key_layer_prune = mask_key.detach() * key_layer

        # dense activate forward
        with torch.no_grad():
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)).detach()

        # pruned activate backward
        attention_scores += torch.matmul(query_layer_prune, key_layer_prune.transpose(-1, -2)) - \
                            torch.matmul(query_layer_prune, key_layer_prune.transpose(-1, -2)).detach()

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # dense activation forward and prune activation backward
        attention_probs, attention_probs_prune = SoftmaxActivationPrune.apply(attention_scores, self.prune_ratio_attn_mat_store)

        # debug use
        # attention_probs = Softmax(dim=-1)(attention_scores)
        # attention_probs_prune = attention_probs

        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)


        # magnitude based prune activation
        mask = activation_prune(value_layer, self.prune_ratio_act_store)
        value_layer_prune = mask.detach() * value_layer

        # dense activate forward
        with torch.no_grad():
            context_layer = torch.matmul(attention_probs, value_layer).detach()

        # pruned activate backward
        context_layer += torch.matmul(attention_probs_prune, value_layer_prune) - \
                         torch.matmul(attention_probs_prune, value_layer_prune).detach()

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # magnitude based prune activation
        mask = activation_prune(context_layer, self.prune_ratio_act_store)
        context_layer_prune = mask.detach() * context_layer

        # dense activate forward
        with torch.no_grad():
            attention_output = self.out(context_layer).detach()

        # pruned activate backward
        attention_output += self.out(context_layer_prune) - self.out(context_layer_prune).detach()

        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


def test_activation_prune():
    a = torch.rand(2, 100, 100)
    prune_ratio = 0.7
    mask = activation_prune(a, prune_ratio)
    print("prune ratio set is {}, real is {}".format(prune_ratio, 1 - mask.float().mean()))


if __name__ == "__main__":
    test_activation_prune()
