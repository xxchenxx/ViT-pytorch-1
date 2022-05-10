import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from actnn import config, QConv1d, QConv2d, QConv3d, QConvTranspose2d, QConvTranspose3d

torch.manual_seed(0)


def testActNNConv():
    config.activation_compression_bits = [2]
    # config.empty_cache_threshold = 0.2
    # config.perlayer = False
    # config.initial_bits = 2
    # config.pergroup = False


    # 4448.5 M
    # print("test actnn memory")
    # model = nn.Sequential(*[QConv2d(in_channels=256,
    #                                 out_channels=256,
    #                                 kernel_size=3,
    #                                 stride=1) for _ in range(10)])

    # OOM
    print("test baseline memory")
    model = nn.Sequential(*[nn.Conv2d(in_channels=256,
                                      out_channels=256,
                                      kernel_size=3,
                                      stride=1) for _ in range(5)])
    model.cuda()


    model = model.cuda()
    input = torch.rand(256, 256, 64, 64).cuda()
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


if __name__ == "__main__":
    testActNNConv()
