import torch
import torch.nn.functional as F
from torch import nn


from miner import QuadrupletMarginMiner, get_all_quadruplets


def distance(x, y, normalize_input=True):
    if normalize_input:
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
    if y == None:
        return -(x @ x.T)
    else:
        return -(x @ y.T)

class QuadrupletMarginLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.margin = 0.3
        self.need_cnt = "need_cnt"
        self.type_of_quadruplets = "all"
        self.miner = (
            None
            if self.type_of_quadruplets == "no-miner"
            else QuadrupletMarginMiner()
        )



    def forward(self, logits, labels, feat2):
        if self.miner is None:
            quadruplets = get_all_quadruplets(labels)
        else:
            quadruplets = self.miner(logits, labels, feat2)

        if quadruplets is None:
            print("no quadruplets")
            return (0, 0) if self.need_cnt else 0

        mat = distance(logits, feat2)
        I, J, K, N = quadruplets[:, 0], quadruplets[:, 1], quadruplets[:, 2], quadruplets[:, 3]
        ij_dists = mat[I, J]
        ik_dists = mat[I, K]
        in_dists = mat[I, N]
        violation1 = ij_dists - in_dists + self.margin
        violation2 = ik_dists - in_dists + self.margin
        losses = F.relu(violation1) + F.relu(violation2)
        loss = torch.mean(losses)

        return (loss, quadruplets.size(0)) if self.need_cnt else loss



