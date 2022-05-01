# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import numpy as np
# from .sparse_matrix_old import SparseTensor
from .custom_functions.sparse_matrix import sparsify, unsparsify

from torch.nn import Dropout, Softmax, Linear


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

        # ctx.sparse_out = SparseTensor(sparse_out, mask)
        shape, mask_, sparse = sparsify(sparse_out, mask)
        ctx.save_for_backward(shape, mask_, sparse)
        # save sparse activation, but forward with dense
        return dense_out, mask

    @staticmethod
    def backward(ctx, grad_in, *args):
        shape, mask, sparse = ctx.saved_tensors
        sparse_out = unsparsify(shape, mask, sparse)

        # sparse_out = ctx.sparse_out
        # sparse_out = sparse_out.to_dense()
        # del ctx.sparse_out

        shape_A = sparse_out.shape
        unsqueeze_cnt = len(shape_A) - 1
        eye = torch.eye(shape_A[-1]).to(sparse_out.device)
        for _ in range(unsqueeze_cnt):
            eye = eye.unsqueeze(0)
        A = sparse_out

        # grad_Softmax = (A * (1 - A)).unsqueeze(-1) * eye - (A.unsqueeze(-1) * A.unsqueeze(-2)) * (1 - eye)
        # grad_out = (grad_in.unsqueeze(-2) @ grad_Softmax).squeeze(-2)

        # merge togather
        # set_trace()
        grad_out = grad_in * A - ((grad_in.unsqueeze(-2) @ A.unsqueeze(-1)) @ A.unsqueeze(-2)).squeeze(-2)

        return grad_out, None


class LinearFunctionActivationPrune(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, input_prune=None, mask_prune=None):
        # ctx.save_for_backward(input_prune, weight, bias)
        # ctx.save_for_backward(weight, bias)
        # ctx.input_prune = SparseTensor(input_prune, mask_prune)

        shape, mask, sparse = sparsify(input_prune, mask_prune)
        print("mask float mean is {}".format(mask.float().mean()))
        ctx.save_for_backward(weight, bias, shape, mask, sparse)

        output = torch.matmul(input, weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        # input_prune, weight, bias = ctx.saved_tensors

        # weight, bias = ctx.saved_tensors
        # input_prune = ctx.input_prune.to_dense()
        # del ctx.input_prune

        weight, bias, shape, mask, sparse = ctx.saved_tensors
        input_prune = unsparsify(shape, mask, sparse)

        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        # print("grad_output shape is {}".format(grad_output.shape))
        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output, weight)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.matmul(grad_output.transpose(-2, -1), input_prune)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None, None


class MatMulActivationPrune(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, A, B, A_prune=None, A_prune_mask=None, B_prune=None, B_prune_mask=None):
        # ctx.A_prune = SparseTensor(A_prune, A_prune_mask)
        # ctx.B_prune = SparseTensor(B_prune, B_prune_mask)

        A_shape, A_mask, A_sparse = sparsify(A_prune, A_prune_mask)
        B_shape, B_mask, B_sparse = sparsify(B_prune, B_prune_mask)
        ctx.save_for_backward(A_shape, A_mask, A_sparse, B_shape, B_mask, B_sparse)

        output = torch.matmul(A, B)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.

        # A_prune = ctx.A_prune.to_dense()
        # del ctx.A_prune
        # B_prune = ctx.B_prune.to_dense()
        # del ctx.B_prune
        grad_A = grad_B = None

        A_shape, A_mask, A_sparse, B_shape, B_mask, B_sparse = ctx.saved_tensors
        A_prune = unsparsify(A_shape, A_mask, A_sparse)
        B_prune = unsparsify(B_shape, B_mask, B_sparse)

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_A = torch.matmul(grad_output, B_prune.transpose(-2, -1))
        if ctx.needs_input_grad[1]:
            grad_B = torch.matmul(A_prune.transpose(-2, -1), grad_output)

        return grad_A, grad_B, None, None, None, None


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class LinearActivationPrune(Linear):
    def forward(self, input, input_prune, mask_prune):
        return LinearFunctionActivationPrune.apply(input, self.weight, self.bias, input_prune, mask_prune)


class MlpActivationPrune(nn.Module):
    def __init__(self, config, prune_ratio_act_store):
        super(MlpActivationPrune, self).__init__()

        self.prune_ratio_act_store = prune_ratio_act_store

        self.fc1 = LinearActivationPrune(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = LinearActivationPrune(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        mask = activation_prune(x, self.prune_ratio_act_store)
        x_prune = mask.detach() * x
        # print("mask.detach() prune ratio is {}".format(mask.detach().float().mean()))
        x = self.fc1(x, x_prune, mask.detach())
        x = self.act_fn(x)
        x = self.dropout(x)

        mask = activation_prune(x, self.prune_ratio_act_store)
        x_prune = mask.detach() * x

        x = self.fc2(x, x_prune, mask.detach())
        x = self.dropout(x)
        return x


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

        self.query = LinearActivationPrune(config.hidden_size, self.all_head_size)
        self.key = LinearActivationPrune(config.hidden_size, self.all_head_size)
        self.value = LinearActivationPrune(config.hidden_size, self.all_head_size)

        self.out = LinearActivationPrune(config.hidden_size, config.hidden_size)
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
        mixed_query_layer = self.query(hidden_states, hidden_states_prune, mask.detach())
        mixed_key_layer = self.key(hidden_states, hidden_states_prune, mask.detach())
        mixed_value_layer = self.value(hidden_states, hidden_states_prune, mask.detach())

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

        # # dense activate forward
        # with torch.no_grad():
        #     attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)).detach()
        #
        # # pruned activate backward
        # attention_scores += torch.matmul(query_layer_prune, key_layer_prune.transpose(-1, -2)) - \
        #                     torch.matmul(query_layer_prune, key_layer_prune.transpose(-1, -2)).detach()

        attention_scores = MatMulActivationPrune.apply(query_layer, key_layer.transpose(-1, -2),
                                                       query_layer_prune, mask_query.detach(),
                                                       key_layer_prune.transpose(-1, -2), mask_key.detach().transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # dense activation forward and prune activation backward
        attention_probs, mask_attention_probs = SoftmaxActivationPrune.apply(attention_scores, self.prune_ratio_attn_mat_store)
        attention_probs_prune = attention_probs * mask_attention_probs

        # debug use
        # attention_probs = Softmax(dim=-1)(attention_scores)
        # attention_probs_prune = attention_probs

        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)


        # magnitude based prune activation
        mask = activation_prune(value_layer, self.prune_ratio_act_store)
        value_layer_prune = mask.detach() * value_layer

        # # dense activate forward
        # with torch.no_grad():
        #     context_layer = torch.matmul(attention_probs, value_layer).detach()
        #
        # # pruned activate backward
        # context_layer += torch.matmul(attention_probs_prune, value_layer_prune) - \
        #                  torch.matmul(attention_probs_prune, value_layer_prune).detach()

        context_layer = MatMulActivationPrune.apply(attention_probs, value_layer,
                                                    attention_probs_prune, mask_attention_probs,
                                                    value_layer_prune, mask.detach())

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # magnitude based prune activation
        mask = activation_prune(context_layer, self.prune_ratio_act_store)
        context_layer_prune = mask.detach() * context_layer
        attention_output = self.out(context_layer, context_layer_prune, mask.detach())

        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


def test_activation_prune():
    a = torch.rand(2, 100, 100)
    prune_ratio = 0.7
    mask = activation_prune(a, prune_ratio)
    print("prune ratio set is {}, real is {}".format(prune_ratio, 1 - mask.float().mean()))


if __name__ == "__main__":
    test_activation_prune()
