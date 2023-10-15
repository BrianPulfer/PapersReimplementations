import numpy as np
import torch
import torch.nn as nn


def get_learnable_embedding(n, hidden_dim):
    return nn.Embedding(n, hidden_dim)


def get_sinusoidal_embedding(n, hidden_dim):
    emb = nn.Embedding(n, hidden_dim)

    def arg(pos, coord):
        return pos / np.power(10000, coord / hidden_dim)

    weight = [
        [
            np.sin(arg(pos, coord)) if coord % 2 == 0 else np.cos(arg(pos, coord))
            for coord in range(hidden_dim)
        ]
        for pos in range(n)
    ]

    emb.weight.data.copy_(torch.tensor(weight))
    emb.requires_grad_(False)
    return emb


def get_rope_embedding(n, hidden_dim):
    # TODO ...
    pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sin_emb = get_sinusoidal_embedding(256, 768)
    sin_emb = sin_emb.cpu().weight.data.numpy()
    plt.imshow(sin_emb)
    plt.xlabel("Hidden Dimension")
    plt.ylabel("Position")
    plt.title("Sinusoidal Embedding")
    plt.show()

    plt.imshow(sin_emb @ sin_emb.T)
    plt.xlabel("Position")
    plt.ylabel("Position")
    plt.title("Dot product of Sinusoidal Embeddings")
    plt.show()
