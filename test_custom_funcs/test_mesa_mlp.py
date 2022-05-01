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

import mesa


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
    configMemTest = ConfigMemoryTest()
    mlp_origin = Mlp(configMemTest).cuda()

    masker = Masker(prune_ratio=0.0)
    mlp_our = MlpActPrune(configMemTest, masker).cuda()

    mlp_our.load_state_dict(mlp_origin.state_dict(), strict=False)

    input = torch.rand(64, 196, 384).cuda()
    input.requires_grad = True

    mlp_origin_out = mlp_origin(input)
    mlp_origin_out[:, :, 6:9].sum().backward()
    input_grad_ori = input.grad
    fc1_grad_origin = mlp_origin.fc1.weight.grad

    # our softmax
    input.grad = torch.zeros_like(input.grad)
    input.requires_grad = True

    # when prune ratio is 0, the two should be equal
    mlp_our_out = mlp_our(input)
    mlp_our_out[:, :, 6:9].sum().backward()
    fc1_grad_our = mlp_our.fc1.weight.grad

    input_grad_our = input.grad

    print("############ prune ratio of 0 #############")
    print("activation dist is {}".format(torch.norm(mlp_our_out[0] - mlp_origin_out[0])))
    # it would be good for slightly different, custom softmax layer often introduce some difference
    print("input grad dist is {}".format(torch.norm(input_grad_ori - input_grad_our)))
    print("fc grad dist is {}".format(torch.norm(fc1_grad_origin - fc1_grad_our)))
    print("fc1 grad norm is {}".format(torch.norm(fc1_grad_origin)))


if __name__ == "__main__":
    # testMemoryAttention()
    testMlpStoreActivationPrune()
