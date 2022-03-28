from __future__ import absolute_import, print_function
import torch.distributed as dist
from utils.utils import *
import time
from pdb import set_trace


from models.modeling import Attention


def collect_est_attn_distance(model):
    context_distance_dict = {}
    context_norm_dict = {}
    attn_distance_dict = {}
    attn_norm_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            context_distance_dict[name] = module.context_dist
            context_norm_dict[name] = module.context_norm
            attn_distance_dict[name] = module.attn_dist
            attn_norm_dict[name] = module.attn_norm

            module.attn_dist = None
            module.context_dist = None
            module.attn_norm = None
            module.context_norm = None

    return context_distance_dict, context_norm_dict, attn_distance_dict, attn_norm_dict


def validate_distill(dataloader, model, log):
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            module.distance_measuring_mode = True

    model.eval()
    end = time.time()

    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()

    context_distance = AverageDict()
    context_norm = AverageDict()
    attn_distance = AverageDict()
    attn_norm = AverageDict()

    for step, batch in enumerate(dataloader):
        data_time.update(time.time() - end)
        batch = tuple(t.cuda() for t in batch)

        x, y = batch
        loss = model(x, y)

        context_distance_dict, context_norm_dict, attn_distance_dict, attn_norm_dict = collect_est_attn_distance(model)
        context_distance.update(context_distance_dict, n=len(x))
        context_norm.update(context_norm_dict, n=len(x))
        attn_distance.update(attn_distance_dict, n=len(x))
        attn_norm.update(attn_norm_dict, n=len(x))
        # loss.backward()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # losses.update(loss.item())
        # if args.fp16:
        #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        # else:
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        # scheduler.step()
        # optimizer.step()
        # optimizer.zero_grad()
        # global_step += 1

        if step % 50 == 0:
            log.info(
                "Training ({}/{} Steps)\t(attn dist={:2.4f}({:2.4f}))\t(attn norm={:2.4f}({:2.4f}))\t(context dist={:2.5f}({:2.4f}))\t"
                "(context norm={:2.5f}({:2.4f}))\tData time={:.2f}({:.2f})\tBatch time={:.2f}({:.2f})".format(
                    step, len(dataloader),
                    attn_distance.val, attn_distance.avg, attn_norm.val, attn_norm.avg,
                    context_distance.val, context_distance.avg, context_norm.val, context_norm.avg,
                    data_time.val, data_time.avg, batch_time.val, batch_time.avg))
        # if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
        #     accuracy = valid(args, model, writer, test_loader, global_step, log)
        #     if best_acc < accuracy:
        #         save_model(args, model, log)
        #         best_acc = accuracy
        #     model.train()
        # torch.distributed.barrier()

    losses.reset()

    for name, module in model.named_modules():
        if isinstance(module, Attention):
            module.distance_measuring_mode = False
