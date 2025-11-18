
import os
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np

class DSHLoss(torch.nn.Module):
    def __init__(self, bit, ):
        super(DSHLoss, self).__init__()
        self.m = 2 * bit
        # self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        # self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])

    def forward(self, u, y,feat2=None ):
        # self.U[ind, :] = u.data
        # self.Y[ind, :] = y.float()
        # .to("cuda:1")
        y = y.float()
        if feat2 is not None:
            dist = ((u.unsqueeze(1) - feat2.unsqueeze(0)).pow(2).sum(dim=2))
        else:
            dist = ((u.unsqueeze(1) - u.unsqueeze(0)).pow(2).sum(dim=2))
        y = (y @ y.t() == 0).float()
        y = y.to(1)
        dist = dist.to(1)
        loss = (1 - y) / 2 * dist + y / 2 * (self.m - dist).clamp(min=0)
        loss1 = loss.mean()

        return loss1



class DTSHLoss(torch.nn.Module):
    def __init__(self,):
        super(DTSHLoss, self).__init__()

    def forward(self, u, y,feat2=None):

        inner_product = u @ u.t() if feat2==None else u @ feat2.t()
        s = y.float() @ y.float().t() > 0
        count = 0
        loss1 = 0
        for row in range(s.shape[0]):
            # if has positive pairs and negative pairs
            if s[row].sum() != 0 and (~s[row]).sum() != 0:
                count += 1
                theta_positive = inner_product[row][s[row] == 1]
                theta_negative = inner_product[row][s[row] == 0]
                triple = (theta_positive.unsqueeze(1) - theta_negative.unsqueeze(0) - 0.5).clamp(
                    min=-100,
                    max=50)
                loss1 += -(triple - torch.log(1 + torch.exp(triple))).mean()

        if count != 0:
            loss1 = loss1 / count
        else:
            loss1 = 0

        loss2 = 0.1 * (u - u.sign()).pow(2).mean()
        return loss1 + loss2


