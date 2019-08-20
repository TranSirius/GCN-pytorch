import torch
import torch.nn as nn
import numpy as np

cross_entro_criterion = nn.CrossEntropyLoss()

def torch_row_normalize(x):
    row_sum = x.sum(1).reshape(-1, 1)
    ret = x / row_sum
    ret[torch.isinf(ret)] = 0
    return ret

def torch_col_normalize(x):
    col_sum = x.sum(0).reshape(1, -1)
    ret = x / col_sum
    ret[torch.isinf(ret)] = 0
    return ret

def logits_masked_loss(logits, labels, masks):
    preds = logits[masks.tolist()]
    target = torch.argmax(labels[masks.tolist()], 1)

    loss = cross_entro_criterion(preds, target)
    return loss

def masked_acc(logits, labels, masks):
    preds = torch.argmax(logits[masks.tolist()], 1)
    target = torch.argmax(labels[masks.tolist()], 1)

    res = (preds == target).float()
    res = sum(res) / len(res)
    return res