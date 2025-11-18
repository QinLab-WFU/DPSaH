from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from sampler import ParametricSampler


class MarginLoss(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.margin = args.margin
        self.n_classes = args.n_classes
        self.beta_constant = args.beta_constant

        self.beta_val = args.beta
        self.beta = args.beta if args.beta_constant else nn.Parameter(torch.ones(self.n_classes) * args.beta)

        self.nu = args.nu

        self.sampler = ParametricSampler(args)

    def forward(self, batch, labels):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels: nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
        """
        sampled_triplets = self.sampler(batch, labels)

        n_triplets = len(sampled_triplets)
        if n_triplets == 0:
            return None, 0

        d_ap, d_an = [], []
        for triplet in sampled_triplets:
            train_triplet = {
                "Anchor": batch[triplet[0], :],
                "Positive": batch[triplet[1], :],
                "Negative": batch[triplet[2]],
            }

            pos_dist = ((train_triplet["Anchor"] - train_triplet["Positive"]).pow(2).sum() + 1e-8).pow(1 / 2)
            neg_dist = ((train_triplet["Anchor"] - train_triplet["Negative"]).pow(2).sum() + 1e-8).pow(1 / 2)

            d_ap.append(pos_dist)
            d_an.append(neg_dist)
        d_ap, d_an = torch.stack(d_ap), torch.stack(d_an)

        if self.beta_constant:
            beta = self.beta
        else:
            # beta = torch.stack([self.beta[labels[triplet[0]]] for triplet in sampled_triplets])
            anchor_labels = labels[np.array(sampled_triplets)[:, 0]]
            beta = torch.einsum("nc,c->n", anchor_labels, self.beta) / anchor_labels.sum(dim=1)

        pos_loss = F.relu(d_ap - beta + self.margin)
        neg_loss = F.relu(beta - d_an + self.margin)

        pair_count = torch.sum((pos_loss > 0.0) + (neg_loss > 0.0))

        if pair_count == 0.0:
            loss = torch.sum(pos_loss + neg_loss)
        else:
            loss = torch.sum(pos_loss + neg_loss) / pair_count

        # if self.nu: loss = loss + beta_regularisation_loss.type(torch.cuda.FloatTensor)

        return loss, n_triplets
