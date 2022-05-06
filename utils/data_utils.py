import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from dataset.init_datasets import init_datasets

from .co_tune import get_transforms


def get_loader(args):
    if "resnet" in args.model_type:
        print("norm of resnet")
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        normalization = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    if args.cotuning_trans:
        print("employ the transform of co-tuning")
        data_transforms = get_transforms(normalization)
        transform_train, transform_test = data_transforms['train'], data_transforms['val']
    else:
        train_trans = [
            transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
            transforms.ToTensor(),
            normalization,
        ]

        if args.train_resize_first:
            train_trans = [transforms.Resize((args.img_size, args.img_size)), ] + train_trans

        transform_train = transforms.Compose(train_trans)
        transform_test = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            normalization,
        ])

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    trainset, valset, testset = init_datasets(args, transform_train, transform_test)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = DistributedSampler(trainset)
    val_sampler = SequentialSampler(valset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(valset,
                            sampler=val_sampler,
                            batch_size=args.eval_batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=args.num_workers,
                             pin_memory=True)

    return train_loader, val_loader, test_loader
