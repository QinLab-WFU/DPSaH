import torch
import torch.nn as nn
from torch.nn.functional import normalize
class MultiSimilarityLoss(nn.Module):
    def __init__(self,):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = 2.0
        self.scale_neg = 40.0
        self.PSD = 'True'
        self.OBDP = 'True'
        self.kd_tau = 1.0
        self.kd_lambda = 75.0
        self.n_epochs = 100
        self.diffusion_w = 0.99
        self.diffusion_tau = 1.0

    def softmax(self, x, tau):
        s = torch.nn.Softmax(dim=1)
        return s(x / tau)

    def logsoftmax(self, x, tau):
        ls = torch.nn.LogSoftmax(dim=1)
        return ls(x / tau)

    def kl_d(self, input, target):
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        return kl_loss(input, target)

    def forward(self, feats, labels ,dataset="MSLOSS",feat2=None):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        if feat2 == None:
            sim_mat = normalize(torch.matmul(feats, torch.t(feats)))
        else:
            sim_mat = normalize(torch.matmul(feats,torch.t(feat2)))

        if dataset == "cifar10-1":
            labels = torch.argmax(labels , dim=1)
        else:
            labels = labels @ labels.t() > 0

        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):
            pos_pair_ = sim_mat[i][labels == labels[i] if dataset =='cifar10-1' else labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            
            neg_pair_ = sim_mat[i][labels != labels[i] if dataset =='cifar10-1' else (labels[i] == False)]

            if torch.numel(pos_pair_) == 0 or torch.numel(neg_pair_) == 0:
                continue

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_) ]
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_) ]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue
                    
            # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / batch_size





        if self.PSD:
            similarity = feats.mm(feats.T)
            teacher_similarity = torch.mm(feats, feats.t())

            if self.OBDP:
                #### diffusion
                ##### standard symmetry normalization
                # masks = torch.eye(batch.size(0)).cuda()
                # affinity = copy.deepcopy(teacher_similarity)
                # affinity[range(batch.size(0)), range(batch.size(0))] = 0.
                # degree = torch.sum(affinity, dim=-1) + 1e-12
                # mat = (degree ** (-0.5)).repeat(batch.size(0), 1) * masks
                # S = mat @ affinity @ mat
                # W = (1 - self.diffusion_w) * torch.inverse(masks - self.diffusion_w * S)
                # target_cache_affinity = torch.matmul(W, teacher_similarity)
                # target_cache = self.softmax(target_cache_affinity, self.tau)

                ##### softmax-based normalization.
                ##### Here use softmax-based normalization is more convenient and achieve almost the same performance.
                masks = torch.eye(feats.size(0)).cuda()
                W = teacher_similarity - masks * 1e9
                W = self.softmax(W, self.diffusion_tau)
                W = (1 - self.diffusion_w) * torch.inverse(masks - self.diffusion_w * W)
                target_cache = torch.matmul(W, self.softmax(teacher_similarity, self.kd_tau))
            else:
                target_cache = self.softmax(teacher_similarity, self.tau)

            sim_cache = self.logsoftmax(similarity, self.kd_tau)
            loss_kd = self.kl_d(sim_cache, target_cache.detach())
            loss = torch.mean(loss) + (100 / self.n_epochs) * self.kd_lambda * (self.kd_tau ** 2) * loss_kd

        else:
            loss = torch.mean(loss)

        return loss












