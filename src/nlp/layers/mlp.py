import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self, in_dim, hidden_dim=None, out_dim=None, activation=nn.GELU(), drop_p=0.1
    ) -> None:
        super(MLP, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else in_dim * 4
        self.out_dim = out_dim if out_dim is not None else in_dim

        self.linear1 = nn.Linear(self.in_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        out = self.dropout(out)
        return out
