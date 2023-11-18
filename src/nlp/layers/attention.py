import torch
import torch.nn as nn


class Attention(nn.Module):
    """Single Attention Head."""

    def __init__(self, dropout_p=0.1):
        super(Attention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, q, k, v, mask=None):
        # Comput the attention scores by computing the dot product of queries with keys
        b, t, d = q.shape
        attn = q @ k.transpose(-2, -1) / (d**0.5)  # b, nq, nk

        # Mask interactions that should not be captured
        if mask is not None:
            assert (
                mask.shape == attn.shape
            ), f"Mask has shape {mask.shape} != {attn.shape}"
            attn = attn.masked_fill(mask == 0, float("-inf"))

        # Computing final output by multiplying attention scores with values
        attn = self.softmax(attn)
        out = attn @ v

        # Dropping out as regularization during training
        out = self.dropout(out)
        return out


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention."""

    def __init__(self, n_heads, dropout_p=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.heads = nn.ModuleList([Attention(dropout_p) for _ in range(self.n_heads)])

    def forward(self, q, k, v, mask=None):
        # Check that dimensionalities are divisible by the number of heads
        b, nq, d = q.shape
        b, nk, dv = v.shape
        assert (
            d % self.n_heads == 0
        ), f"{d}-dimensional query cannot be broken into {self.n_heads} heads."
        assert (
            dv % self.n_heads == 0
        ), f"{dv}-dimensional value cannot be broken into {self.n_heads} heads."

        # Computing attention in all sub-vectors
        qk_dim_per_head = int(d / self.n_heads)
        v_dim_per_head = int(dv / self.n_heads)
        out = torch.cat(
            [
                head(
                    q[:, :, i * qk_dim_per_head : (i + 1) * qk_dim_per_head],
                    k[:, :, i * qk_dim_per_head : (i + 1) * qk_dim_per_head],
                    v[:, :, i * v_dim_per_head : (i + 1) * v_dim_per_head],
                    mask,
                )
                for i, head in enumerate(self.heads)
            ],
            dim=-1,
        )

        return out
