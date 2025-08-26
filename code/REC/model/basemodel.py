# Copyright (c) 2024 westlake-repl
# SPDX-License-Identifier: MIT

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from REC.utils import set_color


def all_gather(data,
               group=None,
               sync_grads=False):
    group = group if group is not None else torch.distributed.group.WORLD
    if torch.distributed.get_world_size() > 1:
        from torch.distributed import nn
        if sync_grads:
            return torch.stack(nn.functional.all_gather(data, group=group), dim=0)
        with torch.no_grad():
            return torch.stack(nn.functional.all_gather(data, group=group), dim=0)
    else:
        return data.unsqueeze(0)


def l2_norm(x, eps=1e-6):
    x = x / torch.clamp(
        torch.linalg.norm(x, ord=2, dim=-1, keepdim=True),
        min=eps,
    )
    return x

def nce_loss(
        pred,
        ground_truth,
        logit_scale,
        nce_samples=None,
        nce_thres=0.99,
        extra_neg_samples=None
    ):
    with torch.no_grad():
        logit_scale.clamp_(0, np.log(100))
    logit_scale = logit_scale.exp()
    N, D = pred.size()
    pred = pred / pred.norm(dim=-1, keepdim=True)
    ground_truth = ground_truth / ground_truth.norm(dim=-1, keepdim=True)
    pos_logits = F.cosine_similarity(pred, ground_truth, dim=-1).unsqueeze(-1)

    if nce_samples is not None:
        nce_indices = torch.randperm(ground_truth.size(0))
        nce_indices = nce_indices.repeat((nce_samples // nce_indices.size(0)) + 1)[:nce_samples]
        neg_samples = ground_truth[nce_indices]
    else:
        neg_samples = ground_truth

    neg_samples = all_gather(neg_samples, sync_grads=True).reshape(-1, D)
    if extra_neg_samples is not None:
        extra_neg_samples = extra_neg_samples / extra_neg_samples.norm(dim=-1, keepdim=True)
        neg_samples = torch.cat([extra_neg_samples, neg_samples], dim=0) # (bs, dim) + (bss, dim) -> (bss + bs, dim)
        
    neg_logits = pred @ neg_samples.t()
    fix_logits = ground_truth @ neg_samples.t()
    neg_logits[fix_logits > nce_thres] = torch.finfo(neg_logits.dtype).min
    if extra_neg_samples is not None:
        row_indices = torch.arange(N, dtype=torch.long, device=neg_logits.device)
        col_indices = torch.arange(N, dtype=torch.long, device=neg_logits.device)
        neg_logits[row_indices, col_indices] = torch.finfo(neg_logits.dtype).min

    logits = torch.cat([pos_logits, neg_logits], dim=-1) * logit_scale
    labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.int64)
    return logits, labels

class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    def load_weights(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        pretrained_dicts = checkpoint['state_dict']
        self.load_state_dict({k.replace('item_embedding.rec_fc', 'visual_encoder.item_encoder.fc'): v for k, v in pretrained_dicts.items()}, strict=False)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + set_color('\nTrainable parameters', 'blue') + f': {params}'
