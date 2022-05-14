import random
import os
import numpy as np
import torch
from models.modeling import VisionTransformer, CONFIGS
from model_mesa.modeling_mesa import VisionTransformer as VisionTransformerMesa
from models.custom_functions.masker import Masker
from models.resnet import resnet

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model, log):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(log.path, "checkpoint_best.pth")
    torch.save(model_to_save.state_dict(), model_checkpoint)
    log.info("Saved model checkpoint to [DIR: {}]".format(os.path.join(log.path, args.name)))


def setup(args, log, num_classes):
    if args.model_type in CONFIGS:
        # Prepare model
        config = CONFIGS[args.model_type]
        if args.mesa:
            log.info("Employ mesa transformer")
            model = VisionTransformerMesa(config, args.img_size, zero_head=True, num_classes=num_classes)
        else:
            masker = None if not args.new_backrazor else Masker(prune_ratio=args.back_prune_ratio)
            if args.new_backrazor:
                assert not args.attn_store_prune

            backrazor_items = ["fc", "matmul", "softmax"]

            if args.backrazor_with_layernorm:
                backrazor_items.append("layernorm")

            if args.backrazor_with_gelu:
                backrazor_items.append("gelu")

            model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes,
                                      prune_mode=args.prune, prune_after_softmax=args.prune_after_softmax,
                                      attn_store_prune=args.attn_store_prune,
                                      masker=masker, quantize=args.quantize, half=args.backrazor_half,
                                      new_backrazor=args.new_backrazor, new_backrazor_item=backrazor_items)

        if not args.train_from_scratch:
            model.load_from(np.load(args.pretrained_dir))
        log.info("{}".format(config))
    else:
        masker = None if not args.new_backrazor else Masker(prune_ratio=args.back_prune_ratio)
        model = resnet.__dict__[args.model_type](pretrained=not args.train_from_scratch, num_classes=num_classes,
                                                new_backrazor=args.new_backrazor, masker=masker)

        if args.new_backrazor:
            print("when pruning backpropogation of resnet, fix the bn")
            for name, parameter in model.named_parameters():
                if "bn" in name:
                    print("fix {}".format(name))
                    parameter.requires_grad = False

    model.to(args.device)
    num_params = count_parameters(model)

    log.info("Training parameters {}".format(args))
    log.info("Total Parameter: \t {}M".format(num_params))

    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    torch.manual_seed(args.seed + args.local_rank)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed + args.local_rank)


def get_second_path(path, insert_name="_logs4.17"):
    dir = ""
    root=path
    while dir == "":
        root, dir = os.path.split(root)
    return os.path.join(root, insert_name, dir)


class logger(object):
    def __init__(self, path, log_name="log.txt", local_rank=0):
        self.path = path
        self.second_path = get_second_path(path)
        self.local_rank = local_rank
        self.log_name = log_name

        if local_rank == 0:
            os.system("mkdir -p {}".format(self.second_path))

    def info(self, msg):
        if self.local_rank in [0, -1]:
            print(msg)
            with open(os.path.join(self.path, self.log_name), 'a') as f:
                f.write(msg + "\n")
            with open(os.path.join(self.second_path, self.log_name), 'a') as f:
                f.write(msg + "\n")


class Mat_Avg_Var_Cal(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = None
        self.var = None
        self.count = 0

    def update(self, mat,):
        '''
        :param mat: [b, ...]
        :return:
        '''
        # update avg
        n = mat.shape[0]
        avg = mat.mean(0)
        torch.distributed.all_reduce(avg)
        avg = avg / torch.distributed.get_world_size()

        if self.avg is None:
            self.avg = avg
            self.count += n
        else:
            self.avg = self.avg * (self.count / (self.count + n)) + mat.sum(0) * (n / (self.count + n))
            self.count += n

        # update var
        n = mat.shape[0]
        var = torch.pow(mat - self.avg.unsqueeze(0), 2).mean(dim=0).detach().mean(0)
        torch.distributed.all_reduce(var)
        var = var / torch.distributed.get_world_size()

        if self.var is None:
            self.var = var
        else:
            self.var = self.var * (self.count / (self.count + n)) + var.sum(0) * (n / (self.count + n))


class Taylor_Cal(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = None
        self.var = None
        self.count = 0

    def update(self, weight, grad):
        '''
        :param mat: [b, ...]
        :return:
        '''
        # update avg
        n = weight.shape[0]

        mat = (weight.detach() * grad.detach()).abs()
        avg = mat.mean(0)
        torch.distributed.all_reduce(avg)
        avg = avg / torch.distributed.get_world_size()

        if self.avg is None:
            self.avg = avg
            self.count += n
        else:
            self.avg = self.avg * (self.count / (self.count + n)) + mat.sum(0) * (n / (self.count + n))
            self.count += n

        
        n = mat.shape[0]
        var = torch.pow(mat - self.avg.unsqueeze(0), 2).mean(dim=0).detach().mean(0)
        torch.distributed.all_reduce(var)
        var = var / torch.distributed.get_world_size()

        if self.var is None:
            self.var = var
        else:
            self.var = self.var * (self.count / (self.count + n)) + var.sum(0) * (n / (self.count + n))
