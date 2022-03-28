
import math
import os.path

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from pdb import set_trace
from collections import OrderedDict
import argparse
from functools import partial

from utils import AverageMeter, logger

import sys
sys.path.append(".")
from models.conv_low_rank import conv_low_rank


def conv2mat(conv, matrix_side):
    """
    conv a norm to the attn matrix format
    method:
    W is target matrix, x is input in vector format, o is output of the conv layer in vector format
    Wx = Conv(x)
    W [x1, x2, ..., xn] = Conv([x1, x2, ..., xn])
    W X = Conv(X)
    let X be I
    W I = Conv(I) = W
    :param conv: conv layer with channel of 1
    :return: O_I
    """
    patch_num_side = int(math.sqrt(matrix_side))
    eye_mat = torch.eye(matrix_side, matrix_side, dtype=conv.weight.dtype, device=conv.weight.device)
    eye_mat = eye_mat.reshape(matrix_side, 1, patch_num_side, patch_num_side)
    eye_mat = eye_mat.expand(matrix_side, conv.in_channels, patch_num_side, patch_num_side)
    W = conv(eye_mat).reshape(conv.out_channels, matrix_side, matrix_side)
    return W


def entropy_regularization_loss(conv_candidates, eps=1e-6):
    # init each conv
    P_convs = []
    for name, conv in conv_candidates.items():
        P_conv = torch.norm(conv.weight) / len(conv.weight.view(-1))
        P_convs.append(P_conv)

    P_convs = torch.stack(P_convs)
    P_convs = F.softmax(P_convs, dim=0)
    H = -torch.sum(P_convs *  torch.log(torch.clamp(P_convs, min=eps)))
    return H, P_convs


def attn_mat_approx(attn_mat, conv_candidates_funs, log=None, regularization_coef=0.1, update_iter=10000, lr=0.1):
    # init each conv
    params = []
    head_dim = attn_mat.shape[0]
    conv_candidates = {k: c(in_channels=head_dim, out_channels=head_dim, groups=head_dim, num_heads=head_dim).cuda() for k, c in conv_candidates_funs.items()}
    for name, conv in conv_candidates.items():
        conv.weight.data = conv.weight.data.normal_(mean=0.0, std=0.01)
        params.append({'params': conv.parameters()})

    optimizer = torch.optim.SGD(params, lr=lr, momentum=0)

    attn_mat = attn_mat.cuda()

    attn_mat_side = attn_mat.shape[-1]
    attn_mat_patch_side = int(math.sqrt(attn_mat_side))
    if attn_mat_side - (attn_mat_patch_side ** 2) > 1:
        raise ValueError("No such attn conv")

    # with cls token, we need to use conv with two mlps in forward
    if attn_mat_side - (attn_mat_patch_side ** 2) == 1:
        log.info("conv with cls token, assume cls token is the first token")
        attn_mat = attn_mat[:, -attn_mat_patch_side ** 2:, -attn_mat_patch_side ** 2:]

    # loss_ave = AverageMeter()
    attn_norm = torch.norm(attn_mat)
    for iter in range(update_iter):
        matrix_sum = 0
        for name, conv in conv_candidates.items():
            matrix_sum += conv2mat(conv, attn_mat.shape[-1])
            # low rank
            matrix_sum += conv.low_rank.unsqueeze(1)

        dif_norm = torch.norm(attn_mat - matrix_sum)
        H, P_convs = entropy_regularization_loss(conv_candidates)
        loss = dif_norm + regularization_coef * H

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # loss_ave.update(loss.item())

        if iter % 100 == 0:
            set_trace()
            log.info("Iter {}/{}, dif_norm {:.3f}, attn mat norm {:.3f}".format(iter, update_iter,
                                                                                dif_norm.item(), attn_norm.item()))
            msg = "Prob each conv: "
            for cnt, (k, _) in enumerate(conv_candidates.items()):
                msg += "{}: {:.5f}\t".format(k, P_convs.detach()[cnt].item())
            log.info(msg)
            # loss_ave.reset()

    return conv_candidates, dif_norm.item(), attn_norm.item()


def validateConvCandidates(conv_candidates):
    patch_num_side = 16
    feature = torch.rand(1, 1, patch_num_side, patch_num_side)

    for name, conv in conv_candidates.items():
        feature = feature.to(conv.weight.device)
        out =  conv(feature)
        if not (len(out.view(-1)) == patch_num_side ** 2):
            raise ValueError("conv of {} is not a valid conv".format(name))


def test_conv():
    patch_num_side = 16
    shape = (3, 3)
    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=shape, padding=2, dilation=2, bias=False)
    weight = conv.weight.data
    conv.weight.data = torch.arange(len(weight.reshape(-1)), device=weight.device, dtype=weight.dtype).reshape(weight.shape)
    print("conv.weight.data is {}".format(conv.weight.data))
    W = conv2mat(conv, patch_num_side ** 2)
    print(W)

    rand = torch.rand(1,1,16, 16)
    out_mul = (rand.reshape(-1) @ W).resize(1, 1,16, 16)
    out = conv(rand)
    print("out from conv is {}".format(out))
    print("out_mul from conv is {}".format(out_mul))
    pass


def parse_args():
    parser = argparse.ArgumentParser("parse path")
    parser.add_argument("exp", type=str)
    parser.add_argument("--save_dir", default="checkpoints_approx_conv", type=str)
    parser.add_argument("--attn_model_path", default="", type=str)

    parser.add_argument("--update_iter", default=10000, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--regularization_coef", default=0.0, type=float)
    parser.add_argument("--conv_size", default=7, type=int)
    return parser.parse_args()


def main():
    args = parse_args()

    save_dir = args.save_dir
    exp = args.exp
    attn_model_path = args.attn_model_path

    update_iter = args.update_iter
    lr = args.lr
    regularization_coef=args.regularization_coef

    save_dir = os.path.join(save_dir, exp)
    if not os.path.isdir(save_dir):
        os.system("mkdir -p {}".format(save_dir))
    log = logger(save_dir)

    conv_size = args.conv_size
    conv_candidates_funs = {
        # "conv3x3": partial(nn.Conv2d, kernel_size=(3,3), padding=1, dilation=1, bias=False),
        # "conv5x5": partial(nn.Conv2d, kernel_size=(5,5), padding=2, dilation=1, bias=False),
        "conv{}x{}".format(conv_size, conv_size): partial(conv_low_rank, token_len=196, kernel_size=(conv_size,conv_size), padding=conv_size // 2, dilation=1, bias=False),
        # "conv3x3_dil2": partial(nn.Conv2d, kernel_size=(3, 3), padding=2, dilation=2, bias=False),
        # "conv3x3_dil3": partial(nn.Conv2d, kernel_size=(3, 3), padding=3, dilation=3, bias=False),
    }
    conv_candidates_vali = {k: v(in_channels=1, out_channels=1) for k,v in conv_candidates_funs.items()}
    validateConvCandidates(conv_candidates_vali)

    # read the attn matrix
    attn_model = torch.load(attn_model_path, map_location="cpu")

    # approximate and save one-by-one
    conv_state_dict = OrderedDict()
    conv_state_dict_info = OrderedDict()
    for name, attn_mat in attn_model.items():
        if ".A" in name:
            print("approximate attn {}".format(name))
            conv_candidates, dif_norm, attn_norm = attn_mat_approx(attn_mat, conv_candidates_funs, log=log,
                                                                   regularization_coef=regularization_coef, update_iter=update_iter, lr=lr)
            conv_state_dict[name] = {k: v.weight.data.clone() for k, v in conv_candidates.items()}
            conv_state_dict_info[name] = {"dif_norm": dif_norm, "attn_norm": attn_norm}

    torch.save(conv_state_dict, os.path.join(save_dir, "conv_state_dict.pth"))
    torch.save(conv_state_dict_info, os.path.join(save_dir, "approx_info.pth"))

    pass


if __name__ == "__main__":
    main()

