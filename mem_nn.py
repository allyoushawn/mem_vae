import torch
import torch.nn as nn
import numpy as np




class MemoryNetwork(torch.nn.Module):
    def __init__(self, num_keys, hid_dim, device):
        super(MemoryNetwork, self).__init__()
        self.device = device
        self.hid_dim = hid_dim
        w1 = torch.empty(num_keys, hid_dim)
        nn.init.uniform_(w1, a=-0.8, b=0.8)
        w2 = torch.empty(num_keys, hid_dim)
        nn.init.uniform_(w2, a=-0.8, b=0.8)
        self.keys = torch.nn.Parameter(w1)
        self.values = torch.nn.Parameter(w2)

        self.keys.requires_grad=False
        self.values.requires_grad=False
        self.sigmoid = nn.Sigmoid()
        self.kl = torch.nn.KLDivLoss(reduction='none')


    def get_alpha(self, query):
        # query shape: N x hid_dim
        return self.sigmoid(torch.mm(query, self.keys.transpose(0, 1) ))

    def read(self, alpha):
        # alpha shape: N x num_keys
        alpha_tmp = torch.unsqueeze(alpha, axis=2).repeat(1,1, self.hid_dim)
        return (alpha_tmp * self.values).mean(axis=1)

    def kl_div(self, z, z_p):
        # z here means latent variable, not syntax
        # z shae: N x hid_dim
        Z = torch.cat((z, 1-z), axis=1)
        Z_p = torch.cat((z_p, 1-z_p), axis=1)
        target = torch.log(Z_p + 1e-10)
        res = self.kl(target, Z)

        res = res.sum(axis=1)
        return res.mean()

