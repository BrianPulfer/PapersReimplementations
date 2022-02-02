import torch
import torch.nn as nn

torch.manual_seed(0)

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads  # The input vectors are broken into (heads) equally sized vectors

        assert self.head_dim * self.heads == self.embed_size, "Embed size has to be divisible by heads"

        # Linear layer
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)  # Maps embedding to query vectors
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)  # Maps embedding to key vectors
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)  # Maps embedding to value vectors

        # The output of the attention is a concatenation of all the results from all heads
        self.fc_out = nn.Linear(self.head_dim * self.heads, self.embed_size)

    def forward(self, values, keys, queries, mask):
        # Batch size
        N = queries.shape[0]

        # Source and target sentence lengths (key_len == query_len)
        key_len, query_len, value_len = keys.shape[1], queries.shape[1], values.shape[1]

        # Split embedding from (N, sentence_len, embedding_size) to (N, sentence_len, heads, head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)
        values = values.reshape(N, value_len, self.heads, self.head_dim)

        # Running embeddings through mappings
        keys = self.keys(keys)
        queries = self.queries(queries)
        values = self.values(values)

        # Carrying out the self-attention
        #  queries shape: (N, query_len, heads, heads_dim)
        #  keys shape:    (N, keys_len, heads, heads_dim)
        #  energy shape:  (N, heads, query_len, keys_len) -> Is the product between queries and keys
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Masking future tokens
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))  # Setting elements that need to be masked to '-inf'

        # Getting the attention values
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # Multiplying attention with values
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])

        # Concatenating all heads again
        out = out.reshape(N, query_len, self.heads * self.head_dim)

        # Running output attention values to a MLP
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)

        # LayerNorm normalizes the values of a layer ((sample - mean(sample)) / std(sample))
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # Feed forward block
        self.ff = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        # Dropout layer for regularizing training
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, keys, queries, mask):
        attention = self.attention(values, keys, queries, mask)
        x = self.dropout(self.norm1(attention + queries))
        forward = self.ff(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout, forward_expansion)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, sequence_length = x.shape

        # N stacked [0, 1, 2, ...]
        positions = torch.arange(0, sequence_length).expand(N, sequence_length).to(self.device)

        # Mapping sentences to word embeddings + the positional embedding
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        # Running through the layers
        for layer in self.layers:
            # Keys, queries and values are the same
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transfomer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, x, value, key, src_mask, trg_mask):
        # src_mask is optional, and it allows to avoid unnecessary computations
        # trg_mask is mandatory, and it tells onto which words attention should not be carried out
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))

        out = self.transfomer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
            self,
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, sequence_length = x.shape
        positions = torch.arange(0, sequence_length).expand(N, sequence_length).to(self.device)

        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            trg_pad_idx,
            embed_size=256,
            num_layers=6,
            forward_expansion=4,
            heads=8,
            dropout=0,
            device="cpu",
            max_length=100
    ):
        super(Transformer, self).__init__()

        # Encoder block
        self.encoder = Encoder(src_vocab_size,
                               embed_size,
                               num_layers,
                               heads,
                               device,
                               forward_expansion,
                               dropout,
                               max_length)

        # Decoder block
        self.decoder = Decoder(trg_vocab_size,
                               embed_size,
                               num_layers,
                               heads,
                               forward_expansion,
                               dropout,
                               device,
                               max_length)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # (N, 1, 1, src_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_length = trg.shape

        # Lower-triangular masking matrix
        trg_mask = torch.tril(torch.ones(trg_length, trg_length)).expand(
            N, 1, trg_length, trg_length
        )

        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)

        return out


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Creating 2 training sentences to be translated
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    # Creating the transformer model
    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)

    # Running the predictions
    out = model(x, trg[:, :-1])
    print(out.shape)
    print("The output of the transformer is (N_sentences, trg sentences lengths, trg vocabulary size)")


if __name__ == '__main__':
    main()
