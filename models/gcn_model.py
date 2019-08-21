import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp

from models.layers.gcn_head import *

class GCN(nn.Module):
    def __init__(self, laplacian, features_dim, class_num, layer_num, heads, hid_dims, non_linear = nn.LeakyReLU(), drop_rate = 0.5, device = 'cuda'):
        super(GCN, self).__init__()
        self.laplacian = laplacian

        self.layers = nn.ModuleList()
        for layer in range(layer_num):
            head_list = nn.ModuleList()
            for head in range(heads[layer]):
                if layer == 0:
                    in_dim = int(features_dim)
                else:
                    in_dim = hid_dims[layer - 1] * heads[layer - 1]

                if layer != layer_num - 1:
                    func = non_linear
                    out_dim = hid_dims[layer]
                else:
                    func = lambda x : x
                    out_dim = class_num

                gcn_head = GcnHead(
                    laplacian = self.laplacian,
                    in_dim = in_dim, 
                    out_dim = out_dim, 
                    non_linear = func, 
                    bias = False,
                    drop_rate = drop_rate,
                    device = device
                )
                head_list.append(gcn_head)
            
            self.layers.append(head_list)
            del head_list
    
    def forward(self, features):
        mid = features
        for index, layer in enumerate(self.layers):
            hidden = []
            for head in layer:
                hidden.append(head(mid))
            
            if index != len(self.layers):
                mid = torch.cat(hidden, 1)
            else:
                mid = sum(hidden)
        return mid

            
