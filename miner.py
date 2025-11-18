import torch
from torch import nn
import torch.nn.functional as F

def distance(x, y, normalize_input=True):
    if normalize_input:
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
    if y == None:
        return -(x @ x.T)
    else:
        return -(x @ y.T)


def get_all_quadruplets(labels, need_jk=False):
    """
    get quadruplets like (A, P1, P2, N)
    just fit AP1, AP2, AN & P1!=P2
    note P1P2 may negative, P1N, P2N may positive, just leave another quadruplet to constrain the margin
    Args:
        multi-hot labels
    Returns:
        quadruplets: shape is n_quadruplets x 4
        Sjk: see "C. Multi-Label Based Hashing" of paper
    """
    sames = (labels @ labels.T > 0).byte()
    diffs = sames ^ 1
    sames.fill_diagonal_(0)

    # mining anchor, positive1, positive2
    I, J, K = torch.where(
        sames.unsqueeze(2) * sames.unsqueeze(1) * torch.triu(1 - torch.eye(sames.shape[0], device=labels.device))
    )

    if I.numel() == 0:
        # print("I is None")
        return None

    # finding negatives & gen quadruplets
    N = diffs[I].nonzero()
    if N.numel() == 0:
        # print("N is None")
        return None
    idx = N[:, 0]
    quadruplets = torch.hstack((I[idx].unsqueeze(1), J[idx].unsqueeze(1), K[idx].unsqueeze(1), N[:, 1].unsqueeze(1)))
    # assert (sames[J[idx], K[idx]] == (labels @ labels.T > 0).byte()[J[idx], K[idx]]).all()
    return quadruplets if not need_jk else (quadruplets, sames[J[idx], K[idx]])


class   QuadrupletMarginMiner(nn.Module):
    """
    Returns quadruplets that violate the margin
    """

    def __init__(self, ):
        super().__init__()
        self.margin = 0.3
        self.type_of_quadruplets = "all"
        self.what_is_hard = "one"

    def forward(self, logits, labels, feat2):
        quadruplets = get_all_quadruplets(labels)
        if quadruplets is None:
            return None
        # mat = distance(logits, self.type_of_distance)
        mat = distance(logits.detach(), feat2)
        I, J, K, N = quadruplets[:, 0], quadruplets[:, 1], quadruplets[:, 2], quadruplets[:, 3]
        ij_dists = mat[I, J]
        ik_dists = mat[I, K]
        in_dists = mat[I, N]
        margin1 = in_dists - ij_dists
        margin2 = in_dists - ik_dists
        # print("margin1:", margin1 <= self.margin)
        # print("margin2:", margin2 <= self.margin)
        # violation1 = ij_dists - in_dists + self.margin
        # violation2 = ik_dists - in_dists + self.margin
        # print("violation:", nn.functional.relu(violation1)+nn.functional.relu(violation2))

        if self.what_is_hard == "one":
            opt = lambda x, y: x | y
        elif self.what_is_hard == "all":
            opt = lambda x, y: x & y
        else:
            raise NotImplementedError(f"not support: {self.what_is_hard}")

        if self.type_of_quadruplets == "easy":
            threshold_condition = opt(margin1 > self.margin, margin2 > self.margin)
        else:
            threshold_condition = opt(margin1 <= self.margin, margin2 <= self.margin)
            if self.type_of_quadruplets == "hard":
                threshold_condition &= opt(margin1 <= 0, margin2 <= 0)
            elif self.type_of_quadruplets == "semi-hard":
                threshold_condition &= opt(margin1 > 0, margin2 > 0)
            else:
                pass  # here is "all"
        if not threshold_condition.any():
            return None
        return quadruplets[threshold_condition]

