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


def testMesaConv():
    # masker = None
    # masker = Masker(prune_ratio=0)

    # 665.1 M
    print("test mesa memory")
    model = nn.Sequential(*[mesa.Conv2d(in_channels=256,
                                        out_channels=256,
                                        kernel_size=3,
                                        stride=1) for _ in range(5)])
    mesa.policy.deploy_on_init(model, 'model_mesa/policy_tiny-8bit.txt', verbose=print, override_verbose=False)
    model.cuda()

    # remove gelu
    for module in model:
        module.act_fn = nn.Identity()

    model = model.cuda()
    input = torch.rand(16, 256, 64, 64).cuda()
    MB = 1024.0 * 1024.0
    print("input usage is {:.1f} MB".format(input.element_size() * input.nelement() / MB))

    mlp_origin_out = model(input)
    mlp_origin_out.sum().backward()

    print("############ mesa mlp #############")
    print("max memory is {:.1f} MB".format(torch.cuda.max_memory_allocated() / MB))

    # activation_bits = 32
    # memory_cost, memory_cost_dict = profile_memory_cost(model, input_size=(1, 196, 384), require_backward=True,
    #                                                     activation_bits=activation_bits, trainable_param_bits=32,
    #                                                     frozen_param_bits=8, batch_size=64)
    # MB = 1024 * 1024
    # print("memory_cost is {:.1f} MB, param size is {:.1f} MB, act_size each sample is {:.1f} MB".
    #       format(memory_cost / MB, memory_cost_dict["param_size"] / MB, memory_cost_dict["act_size"] / MB))


def testStdConv():
    # masker = None
    # masker = Masker(prune_ratio=0)

    # 665.1 M
    print("test mesa memory")
    model = nn.Sequential(*[nn.Conv2d(in_channels=256,
                                        out_channels=256,
                                        kernel_size=3,
                                        stride=1) for _ in range(5)])
    model.cuda()

    # remove gelu
    for module in model:
        module.act_fn = nn.Identity()

    model = model.cuda()
    input = torch.rand(16, 256, 64, 64).cuda()
    MB = 1024.0 * 1024.0
    print("input usage is {:.1f} MB".format(input.element_size() * input.nelement() / MB))

    mlp_origin_out = model(input)
    mlp_origin_out.sum().backward()

    print("############ std conv #############")
    print("max memory is {:.1f} MB".format(torch.cuda.max_memory_allocated() / MB))

    # activation_bits = 32
    # memory_cost, memory_cost_dict = profile_memory_cost(model, input_size=(1, 196, 384), require_backward=True,
    #                                                     activation_bits=activation_bits, trainable_param_bits=32,
    #                                                     frozen_param_bits=8, batch_size=64)
    # MB = 1024 * 1024
    # print("memory_cost is {:.1f} MB, param size is {:.1f} MB, act_size each sample is {:.1f} MB".
    #       format(memory_cost / MB, memory_cost_dict["param_size"] / MB, memory_cost_dict["act_size"] / MB))


def testMlpStoreActivationPrune():
    # configMemTest = ConfigMemoryTest()
    mlp_origin = Conv2d(96, 96, (3, 3), stride=1, padding=1).cuda()

    masker = Masker(prune_ratio=0.9)
    mlp_our = SparseConv2d(96, 96, (3, 3), stride=1, padding=1, masker=masker).cuda()

    mlp_our.load_state_dict(mlp_origin.state_dict(), strict=False)

    input = torch.rand(64, 96, 64, 64).cuda()
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
    testMesaConv()
    # testStdConv()
    # testMlpStoreActivationPrune()
