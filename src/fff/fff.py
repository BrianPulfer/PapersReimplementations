"""
Unofficial re-implementation of

            'Fast Feedforward Networks'

by Peter Belcak and Roger Wattenhofer

ArXiv: https://arxiv.org/abs/2308.14711
"""

import torch
import torch.nn as nn


class FFFMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        hidden_activation: nn.Module = nn.ReLU(),
        out_activation: nn.Module = nn.Sigmoid(),
        swap_prob: float = 0.1,
    ):
        super(FFFMLP, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.swap_prob = swap_prob
        self.use_swapping = True

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.hidden_activation = hidden_activation
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.activation = out_activation

    def forward(self, x: torch.Tensor):
        out = self.linear2(self.hidden_activation(self.linear1(x)))
        out = self.activation(out)

        if self.use_swapping and torch.rand(1) < self.swap_prob:
            out = 1 - out

        return out

    def train(self, mode: bool = True):
        self.use_swapping = True
        return super().train(mode)

    def eval(self):
        self.use_swapping = False
        return super().eval()


class FFFLayer(nn.Module):
    def __init__(
        self, depth: int, in_dim: int, node_hidden: int, leaf_hidden: int, out_dim: int
    ):
        super(FFFLayer, self).__init__()

        self.depth = depth
        self.node_hidden = node_hidden
        self.leaf_hidden = leaf_hidden

        nodes = [FFFMLP(in_dim, node_hidden, 1) for _ in range(2 ** (depth) - 1)]
        leaves = [
            FFFMLP(
                in_dim,
                leaf_hidden,
                out_dim,
                out_activation=nn.Identity(),
                swap_prob=0.0,
            )
            for _ in range(2**depth)
        ]
        self.tree = nn.ModuleList(nodes + leaves)

    def forward(self, x: torch.Tensor, idx: int = 1):
        c = self.tree[idx - 1](x)

        if 2 * idx + 1 <= len(self.tree):
            left = self.forward(x, 2 * idx)
            right = self.forward(x, 2 * idx + 1)

            return c * left + (1 - c) * right

        return c
