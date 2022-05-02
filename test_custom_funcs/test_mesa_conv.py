import torch
from torch import nn

from pdb import set_trace

import sys
sys.path.append("../utils")

from models.custom_functions.custom_fc import LinearSparse
from models.custom_functions.custom_softmax import SoftmaxSparse

from models.modeling_new_prune import MlpActPrune, AttentionActPrune
from models.modeling import Mlp, Attention

from utils.memory_cost_profiler import profile_memory_cost

from models.custom_functions.masker import Masker

from torch.nn import Conv2d
from models.custom_functions.custom_conv import SparseConv2d

import mesa
from test_mesa_all import testGradDif

class ConfigMemoryTest(object):
    def __init__(self):
        self.hidden_size = 384
        self.quantize = False

        self.transformer = dict()
        self.transformer["attention_dropout_rate"] = 0
        self.transformer["num_heads"] = 12

        self.transformer["mlp_dim"] = 384
        self.transformer["dropout_rate"] = 0


def testMlpStoreActivationPrune():
    # configMemTest = ConfigMemoryTest()
    mlp_origin = Conv2d(96, 96, (3, 3), stride=1, padding=1).cuda()

    masker = Masker(prune_ratio=0.9)
    mlp_our = SparseConv2d(96, 96, (3, 3), stride=1, padding=1, masker=masker).cuda()

    mlp_our.load_state_dict(mlp_origin.state_dict(), strict=False)

    input = torch.rand(64, 96, 36, 36).cuda()
    input.requires_grad = True

    mlp_origin_out = mlp_origin(input)
    mlp_origin_out.sum().backward()
    input_grad_ori = input.grad

    # our softmax
    input.grad = torch.zeros_like(input.grad)
    input.requires_grad = True

    # when prune ratio is 0, the two should be equal
    mlp_our_out = mlp_our(input)
    mlp_our_out.sum().backward()

    input_grad_our = input.grad

    print("############ prune ratio of 0 #############")
    print("activation dist is {}".format(torch.norm(mlp_our_out[0] - mlp_origin_out[0])))
    # it would be good for slightly different, custom softmax layer often introduce some difference
    print("input grad dist is {}".format(torch.norm(input_grad_ori - input_grad_our)))
    testGradDif(mlp_origin, mlp_our)


if __name__ == "__main__":
    # testMemoryAttention()
    testMlpStoreActivationPrune()
