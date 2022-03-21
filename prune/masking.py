from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

import numpy as np
import math
from utils.utils import Mat_Avg_Var_Cal, Taylor_Cal

from models.modeling import Attention

class Masking(object):
    def __init__(self, model, death_rate=0.3, growth_death_ratio=1.0, density=0.5, death_rate_decay=None, death_mode='avg_magni_var',
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

        assert growth_mode == 'random'
        assert death_mode == 'avg_magni_var'

        self.masks = {}
        self.names = []

        # stats
        self.death_rate = death_rate
        self.density = density

        self.baseline_nonzero = 0
        for name, module in model.named_modules():
            if isinstance(module, Attention):
                self.masks[name] = module.attention_mask

        self.print_nonzero_counts()

    def init(self, train_loader, model):
        scores = self.score_collect(train_loader, model)
        self.truncate_weights(scores, model, first_time=True)
        self.print_nonzero_counts()

    def step(self, train_loader, model):
        scores = self.score_collect(train_loader, model)
        self.truncate_weights(scores, model)
        self.print_nonzero_counts()


    def truncate_weights(self, scores, model, first_time=False):
        # death
        for name, module in model.named_modules():
            if name in scores:
                if not first_time:
                    new_rest_num = int((1 - self.death_rate) * module.attention_mask.float().sum().item())
                else:
                    new_rest_num = int(self.density * module.attention_mask.float().sum().item())

                threshold, _ = torch.topk(scores[name].flatten(), new_rest_num, sorted=True)
                if len(threshold) > 1:
                    module.attention_mask.data = (scores[name] >= threshold[-1]) & module.attention_mask.data
                else:
                    module.attention_mask.data = (scores[name] >= threshold) & module.attention_mask.data

        if not first_time:
            # grow
            for name, module in model.named_modules():
                if name in scores:
                    total_regrowth = self.density * module.attention_mask.numel() - module.attention_mask.float().sum().item()
                    print(total_regrowth)
                    n = (~module.attention_mask.data).float().sum().item()
                    print(n)
                    expeced_growth_probability = (total_regrowth / n)
                    new_weights = torch.rand(module.attention_mask.data.shape).cuda() < expeced_growth_probability
                    module.attention_mask.data = module.attention_mask.data | new_weights
    '''
                    Collect score
    '''
    def score_collect(self, train_loader, model):
        assert self.death_mode == "avg_magni_var"

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

    def print_nonzero_counts(self):
        total_size = 0
        for name, weight in self.masks.items():
            total_size  += weight.numel()
        self.log.info('Total Model parameters: {}'.format(total_size))

        dense_size = 0
        for name, weight in self.masks.items():
            dense_size += weight.sum().int().item()

        self.log.info('Target density level is {}, current density level is {}'.format(
            self.density, (dense_size / total_size)))


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
            grads = torch.autograd.grad(loss, to_grads, only_inputs=True, retain_graph=True)
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

                avg_norm = (avg - avg.mean()) / torch.clamp(avg.std(), min=1e-7)
                var_norm = (var - var.mean()) / torch.clamp(var.std(), min=1e-7)
                score = self.avg_magni_var_alpha * avg_norm + (1 - self.avg_magni_var_alpha) * var_norm

                scores_dict[name] = score

                module.record_attn_mean_var = None
                del module.attention_probs

        return scores_dict