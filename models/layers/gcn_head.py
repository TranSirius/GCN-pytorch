import torch
import torch.nn as nn

class GcnHead(nn.Module):
    def __init__(self, laplacian, in_dim, out_dim, non_linear = lambda x: x, bias = True, drop_rate = 0.5, device = 'cuda'):
        super(GcnHead, self).__init__()
        self.laplacian = torch.tensor(laplacian, dtype = torch.float32).to(device)

        self.affine = nn.Linear(in_dim, out_dim, bias = bias).to(device)
        nn.init.xavier_normal_(self.affine.weight)
        self.non_linear = non_linear
        self.dropout = nn.Dropout(drop_rate).to(device)

    def forward(self, features):
        features = self.dropout(features)
        ret = self.affine(features)

        laplacian = self.dropout(self.laplacian)
        ret = self.dropout(ret)

        ret = laplacian.mm(ret)
        return self.non_linear(ret)