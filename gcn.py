import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.optim as optim

from utils.data_utils import *
from utils.preprocess import *
from utils.torch_utils import *
from models.gcn import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')
laplacian = torch.tensor(sp_sym_normalize(adj + sp.eye(int(adj.shape[0]))).toarray(), dtype = torch.float32).to('cuda')
features = torch.tensor(sp_row_normalize(features).toarray(), dtype = torch.float32).to('cuda')
y_train, y_val, y_test = torch.tensor(y_train, dtype = torch.float32).to('cuda'), torch.tensor(y_val, dtype = torch.float32).to('cuda'), torch.Tensor(y_test).to('cuda')

layer_num = 2 
hid_dims = [16, 7]
heads = [1, 1]
early_stop = 10
weight_decay = 5e-4
learning_rate = 0.01

Model = GCN
model = Model(
    laplacian = laplacian,
    features_dim = int(features.shape[1]),
    class_num = int(y_train.shape[1]),
    layer_num = layer_num,
    heads = heads,
    hid_dims = hid_dims,
    drop_rate = 0.5
)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
model.train()

L2List = []
for i in model.named_parameters():
    if i[0].startswith("layers.0"):
        L2List.append(i)

vl_ls = []
vl_ac = []
tolerence = 0
for epk in range(1000):
    model.train()
    optimizer.zero_grad()
    logits = model(features)
    loss = logits_masked_loss(logits, y_train, train_mask)
    L2loss = sum([torch.norm(i[1], p = 2) * torch.norm(i[1], p = 2) for i in L2List])
    (loss + L2loss * weight_decay).backward()
    optimizer.step()
    train_ls = loss.item()
    train_ac = masked_acc(logits, y_train, train_mask).item()

    model.eval()
    logits = model(features)
    loss = logits_masked_loss(logits, y_val, val_mask)
    val_ls = loss.item()
    val_acc = masked_acc(logits, y_val, val_mask).item()
    vl_ls.append(val_ls)
    vl_ac.append(val_acc)

    if epk > early_stop:
        if val_ls < np.min(vl_ls[-early_stop:-1]) or val_acc > np.max(vl_ac[-early_stop:-1]):
            tolerence = 0
        else:
            tolerence += 1
            if tolerence > early_stop:
                break

    print(
        "%d: train: loss - %6.4f, acc - %.4f | Val: loss - %6.4f, acc - %.4f" % (
            epk,
            train_ls, 
            train_ac,
            val_ls,
            val_acc
        )
    )

model.eval()
logits = model(features)
loss = logits_masked_loss(logits, y_test, test_mask)
test_ls = loss.item()
test_acc = masked_acc(logits, y_test, test_mask).item()
print(
    "Test Loss = %6.4f | Test Acc = %.4f" % (
        test_ls,
        test_acc
    )
)
f = open("Res.txt", 'a+')
print(test_acc, file = f)
f.close()