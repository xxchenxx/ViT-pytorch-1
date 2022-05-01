import torch
from torch import nn
from torch.nn import functional as F

from pdb import set_trace

import sys
sys.path.append("../utils")

from models.modeling import VisionTransformer, CONFIGS

from models.custom_functions.masker import Masker


def testGradDif(model1, model2):
    max_dif = -1
    max_dif_name = None

    grad_dict_model1 = {k: v.grad for k, v in zip(model1.state_dict(), model1.parameters()) if v.requires_grad}
    grad_dict_model2 = {k: v.grad for k, v in zip(model2.state_dict(), model2.parameters()) if v.requires_grad}

    for name1, grad1 in grad_dict_model1.items():
        if grad1 is not None:
            assert grad_dict_model2[name1] is not None
        dif = torch.norm(grad1 - grad_dict_model2[name1])
        print("grad name is {}, value is {}".format(name1, dif))
        if dif > max_dif:
            max_dif = dif
            max_dif_name = name1

    print("max grad name is {}, value is {}".format(max_dif_name, max_dif))


def testAllGradDif():
    config = CONFIGS["ViT-B_16"]

    masker = Masker(prune_ratio=0.5)
    config = CONFIGS["ViT-B_16"]
    model_our = VisionTransformer(config, 224, zero_head=True, num_classes=100,
                                  attn_store_prune=True,
                                  masker=masker, quantize=False).cuda()
    model_our.eval()

    config = CONFIGS["ViT-B_16"]
    model_origin = VisionTransformer(config, 224, zero_head=True, num_classes=100,
                                     attn_store_prune=False,
                                      masker=None, quantize=False).cuda()
    model_origin.eval()

    msg = model_our.load_state_dict(model_origin.state_dict(), strict=False)
    print(msg)

    input = torch.rand(8, 3, 224, 224).cuda()
    label = torch.LongTensor([0,12,33,4,6,8,9,1]).to(input.device)

    mlp_origin_out, _ = model_origin(input)
    loss_origin = F.cross_entropy(mlp_origin_out, label)
    loss_origin.backward()

    # when prune ratio is 0, the two should be equal
    mlp_our_out, _ = model_our(input)
    loss_our = F.cross_entropy(mlp_our_out, label)
    loss_our.backward()

    print("loss_our - loss_origin is {}".format((loss_our - loss_origin).item()))
    testGradDif(model_origin, model_our)


if __name__ == "__main__":
    testAllGradDif()
    # testMlpStoreActivationPrune()
    # testMesaMlp()
    # testMlpStoreActivationPrune()
