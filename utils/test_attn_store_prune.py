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
import sys
sys.path.append(".")

from models.modeling_attn_store_prune import SoftmaxActivationPrune, AttentionStoreActivationPrune
from models.modeling import Attention


def testSoftMax():
    A = torch.rand(1, 4, 32, 32)
    A.requires_grad = True

    # origin softmax
    A_softmax_ori = Softmax(dim=-1)(A)
    A_softmax_ori.sum().backward()

    A_grad_ori = A.grad

    # our softmax
    A.grad = None

    # when prune ratio is 0, the two should be equal
    A_softmax_our, _ = SoftmaxActivationPrune.apply(A, 0)
    A_softmax_our.sum().backward()

    A_grad_our = A.grad

    print("activation dist is {}".format(torch.norm(A_softmax_ori - A_softmax_our)))
    print("grad dist is {}".format(torch.norm(A_grad_ori - A_grad_our)))


class Config(object):
    def __init__(self):
        self.hidden_size = 32

        self.transformer = dict()
        self.transformer["attention_dropout_rate"] = 0
        self.transformer["num_heads"] = 4


def testAttnStoreActivationPrune():
    config = Config()

    attn_origin = Attention(config, False)
    attn_our = AttentionStoreActivationPrune(config, False, prune_ratio_attn_mat_store=0, prune_ratio_act_store=0)
    attn_our.load_state_dict(attn_origin.state_dict())

    input = torch.rand(2, 10, 32)
    input.requires_grad = True

    attn_origin_out = attn_origin(input)
    attn_origin_out[0].sum().backward()
    input_grad_ori = input.grad

    # our softmax
    input.grad = torch.zeros_like(input.grad)
    input.requires_grad = True

    # when prune ratio is 0, the two should be equal
    attn_our_out = attn_our(input)
    attn_our_out[0].sum().backward()

    input_grad_our = input.grad

    print("############ prune ratio of 0 #############")
    print("activation dist is {}".format(torch.norm(attn_our_out[0] - attn_origin_out[0])))
    # it would be good for slightly different, custom softmax layer often introduce some difference
    print("grad dist is {}".format(torch.norm(input_grad_ori - input_grad_our)))


if __name__ == "__main__":
    # testSoftMax()
    testAttnStoreActivationPrune()
