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


class MeanEstimator(object):
    def __init__(self, score_type='avg', log=None):
        self.score_type = score_type
        self.log = log

        assert score_type == 'avg'

    @ torch.no_grad()
    def mean_estimator(self, train_loader, model):
        for name, module in model.named_modules():
            if isinstance(module, Attention):
                module.record_attn_mean_var = Mat_Avg_Var_Cal()
                module.attn_replace = "none"

        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(train_loader):
                batch = tuple(t.cuda() for t in batch)
                x, y = batch
                loss = model(x, y)
                if step % 50 == 0:
                    self.log.info("collecting score {}/{}".format(step, len(train_loader)))

        for name, module in model.named_modules():
            if isinstance(module, Attention):
                # calculate score
                avg = module.record_attn_mean_var.avg
                # normalize
                score = avg

                module.record_attn_mean_var = None
                module.attn_replace = "parameter"
                module.A.data = score

        # set the attn map parameter value

        return
