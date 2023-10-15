import torch
import torch.nn as nn

from src.nlp.layers.attention import MultiHeadAttention
from src.nlp.layers.mlp import MLP


class EncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        n_heads,
        mlp_hidden=None,
        mlp_out=None,
        mlp_activation=nn.GELU(),
        dropout_p=0.1,
    ):
        super(EncoderBlock, self).__init__()

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.to_qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.to_o = nn.Linear(hidden_dim, hidden_dim)

        self.mhsa = MultiHeadAttention(n_heads, dropout_p)
        self.mlp = MLP(hidden_dim, mlp_hidden, mlp_out, mlp_activation, dropout_p)

    def forward(self, x, attn_mask=None):
        # Attention and residual connection
        q, k, v = self.to_qkv(self.ln1(x)).chunk(3, -1)
        attn_out = self.to_o(self.mhsa(q, k, v, mask=attn_mask))
        x = x + attn_out

        # MLP and residual connection
        mlp_out = self.mlp(self.ln2(x))
        out = x + mlp_out

        return out


class EncoderTransformer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        n_heads,
        depth,
        mlp_hidden=None,
        mlp_out=None,
        mlp_activation=nn.GELU(),
        dropout_p=0.1,
    ):
        super(EncoderTransformer, self).__init__()

        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    hidden_dim, n_heads, mlp_hidden, mlp_out, mlp_activation, dropout_p
                )
                for _ in range(depth)
            ]
        )

    def forward(self, hidden, attn_mask=None):
        # Creating full attention mask if not provided
        if attn_mask is None:
            b, l, d = hidden.shape
            attn_mask = torch.ones((l, l), device=hidden.device).repeat(b, 1, 1)

        # Running blocks
        for block in self.blocks:
            hidden = block(hidden, attn_mask)
        return hidden
