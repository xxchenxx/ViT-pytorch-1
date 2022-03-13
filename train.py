# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# from apex import amp

from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size
from utils.utils import *


def valid(args, model, writer, test_loader, global_step, log):
    # Validation!
    eval_losses = AverageMeter()

    log.info("***** Running Validation *****")
    log.info("  Num steps = {}".format(len(test_loader)))
    log.info("  Batch size = {}".format(args.eval_batch_size))

    model.eval()
    all_preds, all_label = [], []
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(test_loader):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )

        if step % 50 == 0:
            log.info("Validating {}/{} (loss={:2.5f})".format(step, len(test_loader), eval_losses.val))

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    log.info("\n")
    log.info("Validation Results")
    log.info("Global Steps: {}".format(global_step))
    log.info("Valid Loss: {}".format(eval_losses.avg))
    log.info("Valid Accuracy: {}".format(accuracy))

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy


def train(args, model, log, writer):


    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # if args.fp16:
    #     model, optimizer = amp.initialize(models=model,
    #                                       optimizers=optimizer,
    #                                       opt_level=args.fp16_opt_level)
    #     amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    # Distributed training
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    # Train!
    log.info("***** Running training *****")
    log.info("  Total optimization steps = {}".format(args.num_steps))
    log.info("  Instantaneous batch size per GPU = {}".format(args.train_batch_size))
    log.info("  Total train batch size (w. parallel, distributed & accumulation) = {}".format(
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1)))
    log.info("  Gradient Accumulation steps = {}".format(args.gradient_accumulation_steps))

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    while True:
        model.train()
        for step, batch in enumerate(train_loader):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            loss = model(x, y)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            # if args.fp16:
            #     with amp.scale_loss(loss, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                # if args.fp16:
                #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                # else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if step % 50 == 0:
                    log.info("Training ({} / {} Steps) (loss={:2.5f})".format(global_step, t_total, losses.val))
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    accuracy = valid(args, model, writer, test_loader, global_step, log)
                    if best_acc < accuracy:
                        save_model(args, model, log)
                        best_acc = accuracy
                    model.train()

                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break

    if args.local_rank in [-1, 0]:
        writer.close()
    log.info("Best Accuracy: \t{}".format(best_acc))
    log.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True, help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default=".", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    # if args.local_rank == -1:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     args.n_gpu = torch.cuda.device_count()
    # else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend='nccl',
                                         timeout=timedelta(minutes=60))
    args.n_gpu = torch.distributed.get_world_size()
    if args.train_batch_size % args.n_gpu != 0:
        raise ValueError("batch size of {} is not divisible by gpu number of {}".format(args.train_batch_size, args.n_gpu))
    args.train_batch_size = args.train_batch_size // args.n_gpu
    args.device = device

    # Setup logging
    """ Train the model """
    save_dir = os.path.join(args.output_dir, "checkpoints", args.name)
    writer = None
    if args.local_rank in [-1, 0]:
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs", args.name))

    log = logger(save_dir, log_name="log.txt", local_rank=args.local_rank)
    log.info("Process rank: {}, device: {}, n_gpu: {}, distributed training: {}, 16-bits training: {}".format
        (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args, log)

    # Training
    train(args, model, log, writer)


if __name__ == "__main__":
    main()
