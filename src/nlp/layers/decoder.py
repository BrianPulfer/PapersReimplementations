import torch
import torch.nn as nn

from src.nlp.attention import MultiHeadAttention
from src.nlp.mlp import MLP


class DecoderBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        n_heads,
        mlp_hidden=None,
        mlp_out=None,
        mlp_activation=nn.GELU(),
        dropout_p=0.1,
    ):
        super(DecoderBlock, self).__init__()

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

        self.sa_qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.sa_o = nn.Linear(hidden_dim, hidden_dim)

        self.xa_q = nn.Linear(hidden_dim, hidden_dim)
        self.xa_kv = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.xa_o = nn.Linear(hidden_dim, hidden_dim)

        self.mhsa = MultiHeadAttention(n_heads, dropout_p)
        self.mhxa = MultiHeadAttention(n_heads, dropout_p)
        self.mlp = MLP(hidden_dim, mlp_hidden, mlp_out, mlp_activation, dropout_p)

    def forward(self, x, kv=None, self_attn_mask=None, cross_attn_mask=None):
        # Self-attention and residual part
        q, k, v = self.sa_qkv(self.ln1(x)).chunk(3, -1)
        sa_out = self.sa_o(self.mhsa(q, k, v, self_attn_mask))
        x = x + sa_out

        # Cross-attention and residual part
        if kv is not None:
            q = self.xa_q(self.ln2(x))
            k, v = self.xa_kv(kv)
            xa_out = self.xa_o(self.mhxa(q, k, v, cross_attn_mask))
            x = x + xa_out

        # MLP and residual part
        out = x + self.mlp(self.ln3(x))
        return out


class DecoderTransformer(nn.Module):
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
        super(DecoderTransformer, self).__init__()

        self.embeddings = nn.Identity()
        self.pos_embeddings = nn.Identity()

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    hidden_dim, n_heads, mlp_hidden, mlp_out, mlp_activation, dropout_p
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x, kv=None, self_attn_mask=None, cross_attn_mask=None):
        hidden = self.embeddings(x) + self.pos_embeddings(x)

        for block in self.blocks:
            hidden = block(hidden, kv, self_attn_mask, cross_attn_mask)

        return hidden
