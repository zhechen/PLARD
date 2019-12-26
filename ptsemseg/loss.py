import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy2d(input, target, weight=None, reduction='elementwise_mean'):
    n, c, h, w = input.size()

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(input, target, 
                           weight=weight, 
                           reduction=reduction,
                           ignore_index=255)
    return loss

def bootstrapped_cross_entropy2d(input, target, K, weight=None, size_average=True):
    
    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, target, K, weight=None, size_average=True):
        n, c, h, w = input.size()
        log_p = F.log_softmax(input, dim=1)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
        log_p = log_p.view(-1, c)

        mask = target >= 0
        target = target[mask]
        loss = F.nll_loss(log_p, target, weight=weight, ignore_index=250,
                          reduce=False, size_average=False)
        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(input=torch.unsqueeze(input[i], 0),
                                           target=torch.unsqueeze(target[i], 0),
                                           K=K,
                                           weight=weight,
                                           size_average=size_average)
    return loss / float(batch_size)


def multi_scale_cross_entropy2d(input, target, weight=None, size_average=True, scale_weight=None, vis=False):
    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight == None: 
        n_inp = len(input)
        scale = 0.4
        scale_weight = [1.0, 0.4, 0.16]

    losses = [0.0 for _ in range(len(input))]
    loss = 0.0
    for i, inp in enumerate(input):
        if type(target) == list:
            cur_loss = scale_weight[i] * cross_entropy2d(input=inp, target=target[i], weight=weight) 
        else:
            cur_loss = scale_weight[i] * cross_entropy2d(input=inp, target=target, weight=weight) 

        loss = loss + cur_loss
        if vis:
            losses[i] = cur_loss.detach().cpu().numpy() 

    if vis and loss.get_device() == 0:
        print('cls %.3f; cls aux %.3f; lidar cls aux: %.3f; bdry: %.3f'%
            (float(losses[0]), float(losses[1]), float(losses[2]), float(losses[-1])))

    return loss

