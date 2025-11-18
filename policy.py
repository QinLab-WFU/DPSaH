from argparse import Namespace

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()

        self.input_dim, self.output_dim, self.action_space = (
            args.policy_input_dim,
            args.n_bins - 1,
            nn.Parameter(torch.tensor(args.policy_action_space), requires_grad=False),
        )

        #####
        self.network = []
        input_dim = self.input_dim
        for n_hidden in args.policy_size:
            self.network.extend([nn.Linear(int(input_dim), int(n_hidden)), nn.ReLU()])
            input_dim = n_hidden
        self.network = nn.Sequential(*self.network)

        #####
        self.action_layer = nn.Linear(args.policy_size[-1], len(self.action_space) * self.output_dim)
        self.state_value_layer = nn.Linear(args.policy_size[-1], 1)

    def forward(self, x, a=None):
        x = self.network(x)
        action_weights = self.action_layer(x).view(x.size(0), len(self.action_space), self.output_dim)
        probs = F.softmax(action_weights, dim=-1)
        distr = Categorical(probs.permute(0, 2, 1))
        if a is None:
            actions = distr.sample()
        else:
            actions = a
        log_probs = distr.log_prob(actions)
        state_values = self.state_value_layer(x)
        return actions, log_probs, state_values

    def convert(self, actions):
        return self.action_space[actions]
