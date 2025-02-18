from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

import numpy as np
from pdb import set_trace
import math
from utils.utils import Mat_Avg_Var_Cal, Taylor_Cal

from models.modeling import Attention

def exp_prune_ratio_cal(current_step, all_steps, final_prune):
    prune_each_step = 1 - math.exp(math.log(max(1 - final_prune, 1e-4)) / all_steps)
    prune_current = 1 - ((1 - prune_each_step) ** current_step)
    return prune_current

class Masking(object):
    def __init__(self, model, death_rate=0.3, growth_death_ratio=1.0, density=0.5, death_rate_decay=None,
                 init_method='avg_magni_var', init_iter_time=5, death_mode='avg_magni_var',
                 growth_mode='random', args=None, avg_magni_var_alpha=0, log=None):
        growth_modes = ['random']
        if growth_mode not in growth_modes:
            print('Growth mode: {0} not supported!'.format(growth_mode))
            print('Supported modes are:', str(growth_modes))

        self.args = args
        self.device = torch.device("cuda")
        self.growth_mode = growth_mode
        self.death_mode = death_mode
        self.growth_death_ratio = growth_death_ratio
        self.death_rate_decay = death_rate_decay
        self.log = log
        self.avg_magni_var_alpha = avg_magni_var_alpha
        self.init_method = init_method
        self.init_iter_time = init_iter_time

        assert growth_mode == 'random'

        self.masks = {}
        self.names = []

        # stats
        self.death_rate = death_rate
        self.density = density

        self.baseline_nonzero = 0
        for name, module in model.named_modules():
            if isinstance(module, Attention):
                self.masks[name] = module.attention_mask

        self.print_nonzero_counts(target_density=1.0)

    def init(self, train_loader, model):
        if self.init_method == "avg_magni_var":
            scores = self.score_collect(train_loader, model)
            self.truncate_weights(scores, model, first_time=True)
            self.print_nonzero_counts()
        elif self.init_method == "taylor_change_magni_var":
            # init a pretrain model
            model_pretrain = copy.deepcopy(model)
            for param in model_pretrain.parameters():
                param.requires_grad = False
            model_pretrain.eval()
            model.eval()

            # prune to target dim
            scores = self.score_collect(train_loader, model)
            self.truncate_weights(scores, model, first_time=True)

            # iteratively truncate the weight to make the distance to previous pre-train small
            for iter in range(self.init_iter_time):
                # rigL to search the pruning method with least affect to the pretrained model
                scores = self.score_collect_taylor_distance(train_loader, model, model_pretrain, distance_mode=True)
                self.truncate_weights(scores, model)
                self.print_nonzero_counts()
        else:
            raise ValueError("No init method of {}".format(self.init_method))

    def step(self, train_loader, model):
        if self.death_mode == "avg_magni_var":
            scores = self.score_collect(train_loader, model)
        elif self.death_mode == "taylor_magni_var":
            scores = self.score_collect_taylor_distance(train_loader, model)
        else:
            raise ValueError("No death_mode of {}".format(self.death_mode))
        self.truncate_weights(scores, model)
        self.print_nonzero_counts()


    def truncate_weights(self, scores, model, first_time=False, first_time_claim_density=-1):
        # death
        for name, module in model.named_modules():
            if name in scores:
                if not first_time:
                    new_rest_num = int((1 - self.death_rate) * module.attention_mask.float().sum().item())
                else:
                    density = self.density if first_time_claim_density < 0 else first_time_claim_density
                    new_rest_num = int(density * module.attention_mask.float().sum().item())

                threshold, _ = torch.topk(scores[name].flatten(), new_rest_num, sorted=True)
                if len(threshold) > 0:
                    module.attention_mask.data = (scores[name] >= threshold[-1]) & module.attention_mask.data
                else:
                    self.log.info("Warning: there is no module pruned")
                    module.attention_mask.data = torch.zeros_like(module.attention_mask.data)

        if not first_time:
            self.log.info("After death")
            self.print_nonzero_counts(target_density=(1 - self.death_rate) * self.density)

            # grow
            for name, module in model.named_modules():
                if name in scores:
                    total_regrowth = self.density * module.attention_mask.numel() - module.attention_mask.float().sum().item()
                    n = (~module.attention_mask.data).float().sum().item()
                    expeced_growth_probability = (total_regrowth / n)
                    new_weights = torch.rand(module.attention_mask.data.shape).cuda() < expeced_growth_probability
                    module.attention_mask.data = module.attention_mask.data | new_weights
    '''
                    Collect score
    '''
    def score_collect(self, train_loader, model):
        for name, module in model.named_modules():
            if isinstance(module, Attention):
                module.record_attn_mean_var = Mat_Avg_Var_Cal()

        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(train_loader):
                batch = tuple(t.to(self.args.device) for t in batch)
                x, y = batch
                loss = model(x, y)
                if step % 50 == 0:
                    self.log.info("collecting score {}/{}".format(step, len(train_loader)))

        scores_dict = {}
        for name, module in model.named_modules():
            if isinstance(module, Attention):
                # calculate score
                avg = module.record_attn_mean_var.avg
                var = module.record_attn_mean_var.var
                # normalize
                avg_norm = (avg - avg.mean()) / torch.clamp(avg.std(), min=1e-7)
                var_norm = (var - var.mean()) / torch.clamp(var.std(), min=1e-7)
                score = self.avg_magni_var_alpha * avg_norm + (1 - self.avg_magni_var_alpha) * var_norm

                scores_dict[name] = score
                module.record_attn_mean_var = None

        return scores_dict

    def score_collect_taylor_distance(self, train_loader, model, model_pretrain=None, distance_mode=False):
        criterion = torch.nn.MSELoss()

        for name, module in model.named_modules():
            if isinstance(module, Attention):
                module.record_attn_taylor = Taylor_Cal()
                module.record_attention_probs = True
        model.zero_grad()
        for step, batch in enumerate(train_loader):
            # print(step)
            batch = tuple(t.to(self.args.device) for t in batch)
            x, y = batch

            if distance_mode:
                with torch.no_grad():
                    feature_pretrain = model_pretrain(x, return_encoded_feature=True)
                feature_pruned = model(x, return_encoded_feature=True)
                loss = criterion(feature_pruned, feature_pretrain)
            else:
                loss = model(x, y)

            to_grads = []
            for module in model.modules():
                if isinstance(module, Attention):
                    to_grads.append(module.attention_probs)

            grads = torch.autograd.grad(loss, to_grads, only_inputs=True, retain_graph=False)
            idx = 0
            for module in model.modules():
                if isinstance(module, Attention):
                    module.record_attn_taylor.update(module.attention_probs, grads[idx])
                    idx += 1

            if step % 50 == 0:
                self.log.info("collecting score {}/{}".format(step, len(train_loader)))

        scores_dict = {}
        for name, module in model.named_modules():
            if isinstance(module, Attention):
                # calculate score
                avg = module.record_attn_taylor.avg
                var = module.record_attn_taylor.var

                # normalize
                avg_norm = (avg - avg.mean()) / torch.clamp(avg.std(), min=1e-7)
                var_norm = (var - var.mean()) / torch.clamp(var.std(), min=1e-7)
                score = self.avg_magni_var_alpha * avg_norm + (1 - self.avg_magni_var_alpha) * var_norm

                scores_dict[name] = score

        return scores_dict
    '''
                    Utils
    '''
    def print_nonzero_counts(self, target_density=-1):
        total_size = 0
        for name, weight in self.masks.items():
            total_size  += weight.numel()
        self.log.info('Total Model parameters: {}'.format(total_size))

        dense_size = 0
        for name, weight in self.masks.items():
            dense_size += weight.sum().int().item()

        if target_density < 0:
            target_density = self.density

        self.log.info('Target density level is {}, current density level is {}'.format(
            target_density, (dense_size / total_size)))

        if abs(target_density - (dense_size / total_size)) > 0.1:
            raise ValueError("Current density margin is larger than 0.1")


class TaylorMasking(Masking):
    def score_collect(self, train_loader, model):
        assert self.death_mode == "avg_magni_var"

        for name, module in model.named_modules():
            if isinstance(module, Attention):
                module.record_attn_mean_var = Mat_Avg_Var_Cal()
                module.record_attn_taylor = Taylor_Cal()
                module.record_attention_probs = True
        model.zero_grad()
        for step, batch in enumerate(train_loader):
            #print(step)
            batch = tuple(t.to(self.args.device) for t in batch)
            x, y = batch
            loss = model(x, y)
            to_grads = []
            for module in model.modules():
                if isinstance(module, Attention):
                    to_grads.append(module.attention_probs)
            grads = torch.autograd.grad(loss, to_grads, only_inputs=True, retain_graph=True)[0]
            idx = 0
            for module in model.modules():
                if isinstance(module, Attention):
                    module.record_attn_taylor.update(module.attention_probs, grads[idx])
                    idx += 1
            
            if step % 50 == 0:
                self.log.info("collecting score {}/{}".format(step, len(train_loader)))

        scores_dict = {}
        for name, module in model.named_modules():
            if isinstance(module, Attention):
                # calculate score
                avg = module.record_attn_taylor.avg
                var = module.record_attn_taylor.var

                # normalize
                avg_norm = (avg - avg.mean()) / torch.clamp(avg.std(), min=1e-7)
                var_norm = (var - var.mean()) / torch.clamp(var.std(), min=1e-7)
                score = self.avg_magni_var_alpha * avg_norm + (1 - self.avg_magni_var_alpha) * var_norm

                scores_dict[name] = score
                module.record_attn_mean_var = None

        return scores_dict