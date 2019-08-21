import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.optim as optim

from utils.data_utils import *
from utils.preprocess import *
from utils.torch_utils import *
from models.gcn_model import *

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="GCN - pytorch.")
    parser.add_argument('-dataset', nargs = '?', default = 'cora', help = 'Select your dataset')
    parser.add_argument('-epoch', type = int, default = 1000, help = 'maximum epoch, default is 1000')
    parser.add_argument('-early', type = int, default = 100, help = 'early stop, default is 100')
    parser.add_argument('-device', nargs = '?', default = 'cpu', help = 'which devices used to train, default is cpu')
    parser.add_argument('-lr', type = float, default = 0.01, help = 'learning rate')
    parser.add_argument('-weight', type = float, default = 5e-4, help = 'weight decay')
    parser.add_argument('-hidden', nargs = '?', default = '16, 7', help = 'hidden layer size, split by comma')
    parser.add_argument('-layer', type = int, default = 2, help = 'layer number')
    return parser.parse_args()

args = parse_args()
device = 'cuda'
if args.device == 'cpu':
    device = 'cpu'

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset)

laplacian = sp_sym_normalize(adj + sp.eye(adj.shape[0])).toarray()
laplacian = torch.tensor(laplacian, dtype = torch.float32).to(device)

features = sp_row_normalize(features).toarray()
features = torch.tensor(features, dtype = torch.float32).to(device)

y_train = torch.tensor(y_train, dtype = torch.float32).to(device)
y_val   = torch.tensor(y_val,   dtype = torch.float32).to(device)
y_test  = torch.tensor(y_test,  dtype = torch.float32).to(device)


layer_num = args.layer
hid_dims_str = args.hidden.split(',')
if len(hid_dims_str) < layer_num - 1:
    print('hidden size error')
hid_dims = [int(i) for i in hid_dims_str]
heads = [8, 8]
early_stop = args.early
weight_decay = args.weight
learning_rate = args.lr

Model = GCN
model = Model(
    laplacian = laplacian,
    features_dim = int(features.shape[1]),
    class_num = int(y_train.shape[1]),
    layer_num = layer_num,
    heads = heads,
    hid_dims = hid_dims,
    drop_rate = 0.5,
    device = device
)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
model.train()

L2List = []
for i in model.named_parameters():
    if i[0].startswith("layers.0"):
        L2List.append(i)

vl_ls = []
vl_ac = []
acc_max = 0
loss_min = 100
tolerence = 0
save_file = 'pretrained/gcn'
for epk in range(args.epoch):
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
        if val_ls < loss_min or val_acc > acc_max:
            if val_ls < loss_min and val_acc > acc_max:
                torch.save(model.state_dict(), save_file)
            tolerence = 0
            loss_min = min(loss_min, val_ls)
            acc_max = max(acc_max, val_acc)
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

model.load_state_dict(torch.load(save_file))
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