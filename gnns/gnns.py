"""Implementation of convolutional, attentional and message-passing GNNs, inspired by the paper
    Everything is Connected: Graph Neural Networks
(https://arxiv.org/pdf/2301.08210v1.pdf)
 
Useful links:
 - Petar Veličković PDF talk: https://petar-v.com/talks/GNN-EEML.pdf
"""

import warnings
from tqdm import tqdm
from argparse import ArgumentParser

import wandb

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda


# Definitions
NETWORK_TYPES = ["attn", "conv", "mp"]
AGGREGATION_FUNCTIONS = {
    "sum": lambda X: torch.sum(X, dim=1),
    "avg": lambda X: torch.mean(X, dim=1)
}


def parse_args():
    """Parses the program arguments"""
    parser = ArgumentParser()

    # Model arguments
    parser.add_argument(f"--type", type=str, help="Type of the network used.",
                        choices=NETWORK_TYPES, default="attn")
    parser.add_argument(f"--aggregation", type=str, help="Aggregation function",
                        choices=list(AGGREGATION_FUNCTIONS.keys()), default="sum")
    parser.add_argument(f"--n_layers", type=int,
                        help="Number of layers of the GNNs", default=8)

    # Training arguments
    parser.add_argument(f"--epochs", type=int,
                        help="Training epochs. Default is 100.", default=100)
    parser.add_argument(f"--lr", type=float,
                        help="Learning rate.", default=0.001)
    parser.add_argument(f"--batch_size", type=int,
                        help="Batch size used for training. Default is 32.", default=32)
    parser.add_argument(f"--checkpoint", type=str,
                        help="Path to model checkpoints", default=None)

    return vars(parser.parse_args())


def get_device():
    """Gets the CUDA device if available, warns that code will run on CPU only otherwise"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("\nFound GPU: ", torch.cuda.get_device_name(device))
        return device

    warnings.warn("WARNING: No GPU found - Training on CPU.")
    return torch.device("cpu")


class PsiNetwork(nn.Module):
    """
    Simple MLP network denoted as the 'psi' function in the paper.
    The role of this network is to extract relevant features to be passed to neighbouring edges.
    """

    def __init__(self, in_size, out_size):
        super(PsiNetwork, self).__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.relu = nn.ReLU()

    def forward(self, X):
        return self.relu(self.linear(X))


class GraphConvLayer(nn.Module):
    """
    Graph Convolutional layer.
    It computes the next hidden states of the edges as a convolution over neighbours.
    """

    def __init__(self, n, d, aggr):
        super(GraphConvLayer, self).__init__()

        self.psi = PsiNetwork(d, d)
        self.coefficients = nn.Parameter(torch.ones((n, n)) / n)
        self.aggr = AGGREGATION_FUNCTIONS[aggr]
        self.phi = nn.Sequential(
            nn.Linear(2*d, d),
            nn.ReLU(),
            nn.Linear(d, d)
        )

    def forward(self, H, A):
        weights = self.coefficients * A  # (N, N)
        features = self.psi(H)  # (B, N, D)
        messages = torch.einsum(
            "nm, bnd -> bnmd", weights, features)  # (B, N, N, D)
        messages = self.aggr(messages)  # (B, N, D)
        phi_input = torch.cat((messages, H), dim=-1)  #  (B, N, 2*D)
        return self.phi(phi_input)  # (B, N, D)

class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        
        self.dim = dim
        self.to_qk = nn.Linear(dim, 2*dim)
        
    def forward(self, x):
        q, k = self.to_qk(x).chunk(2, -1)
        attn_cues = ((q @ k.transpose(-2, -1)) / (self.dim**0.5 + 1e-5)).softmax(-1)
        return attn_cues
        

class GraphAttentionLayer(nn.Module):
    """
    Graph Attentional Layer.
    It computes the next hidden states of the edges using attention.
    """

    def __init__(self, n, d, aggr):
        super(GraphAttentionLayer, self).__init__()
        
        self.aggr = AGGREGATION_FUNCTIONS[aggr]
        
        self.psi = PsiNetwork(d, d)
        self.sa = Attention(d)
        self.phi = nn.Sequential(
            nn.Linear(2*d, d),
            nn.ReLU(),
            nn.Linear(d, d)
        )

    def forward(self, H, A):
        features = self.psi(H)  # (B, N, D)
        attn = self.sa(H) * A # (B, N, N)
        
        messages = torch.einsum("bnd, bnm -> bnmd", features, attn)  # (B, N, N, D)
        messages = self.aggr(messages) # (B, N, D)
        
        phi_input = torch.cat((messages, H), dim=-1)
        return self.phi(phi_input)


class GraphMPLayer(nn.Module):
    """
    Graph Message-Passing Layer.
    It computes the next hidden states of edges by learning affinity between neighboring edges.
    """

    def __init__(self, n, d, aggr):
        super(GraphMPLayer, self).__init__()
        # TODO
 
    def forward(self, H, A):
        # TODO
        pass


class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network class."""

    _NET_TYPE_TO_LAYER = {
        "attn": GraphAttentionLayer,
        "conv": GraphConvLayer,
        "mp": GraphMPLayer
    }

    def __init__(self, net_type, n_layers, n, d_in, d_hidden, d_out, aggr):
        super(GraphNeuralNetwork, self).__init__()

        assert net_type in NETWORK_TYPES, f"ERROR: GNN type {net_type} not supported. Pick one of {NETWORK_TYPES}"

        self.net_type = net_type
        self.n_layers = n_layers
        self.encoding = nn.Linear(d_in, d_hidden)
        self.layers = nn.ModuleList(
            [self._NET_TYPE_TO_LAYER[net_type](n, d_hidden, aggr)
             for _ in range(n_layers)])

        self.out_aggr = AGGREGATION_FUNCTIONS[aggr]
        self.out_mlp = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out)
        )

    def forward(self, X, A):
        # X has shape (N, D) and represents the edges.
        # A is binary with shape (N, N) and represents the adjacency matrix.
        H = self.encoding(X)
        for l in self.layers:
            H = l(H, A)
        return self.out_mlp(self.out_aggr(H))


def main():
    # Parsing arguments
    args = parse_args()
    print("Launched program with the following arguments:", args)

    # Loading data
    # We reshape the image such that each pixel is an edge with C features.
    transform = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # (C, H, W) -> (H*W, C).
        Lambda(lambda x: x.flatten(1).transpose(1, 0))
    ])
    train_set = CIFAR10("./../datasets", train=True,
                        transform=transform, download=True)
    test_set = CIFAR10("./../datasets", train=False,
                       transform=transform, download=True)

    train_loader = DataLoader(
        train_set, batch_size=args["batch_size"], shuffle=True)
    test_loader = DataLoader(
        test_set, batch_size=args["batch_size"], shuffle=False)

    # Building the Neighbourhood matrix (1024 x 1024) for all "graphs" (images of size 32x32)
    A = torch.zeros((32*32, 32*32))
    nums = torch.arange(32*32).reshape((32, 32))
    for i in range(32):
        for j in range(32):
            start_x, start_y = i-1 if i > 0 else 0, j-1 if j > 0 else 0
            neighbours = nums[start_x: i+2, start_y: j+2].flatten()

            for n in neighbours:
                A[i*32 + j, n] = A[n, i*32 + j] = 1

    # Creating model
    # Number of edges, edge dimensionality, hidden dimensionality and number of output classes
    n, d, h, o = 32*32, 3, 32, 10
    model = GraphNeuralNetwork(
        args["type"], args["n_layers"], n, d, h, o, aggr=args["aggregation"])
    print(model(torch.randn(7, 1024, 3), torch.randn(1024, 1024)).shape)

    # Training loop
    n_epochs = args["epochs"]
    checkpoint = args["checkpoint"] if args["checkpoint"] else f"({args['type']}_gnn).pt"
    optim = Adam(model.parameters(), args["lr"])
    criterion = nn.CrossEntropyLoss()

    device = get_device()
    model = model.to(device)
    model.train()

    min_loss = float("inf")
    progress_bar = tqdm(range(1, n_epochs+1))

    wandb.init(project="Papers Re-implementations",
               entity="peutlefaire",
               name=f"GNN ({args['type']})",
               config={
                   "type": args["type"],
                   "aggregation": args["aggregation"],
                   "layers": args["n_layers"],
                   "epochs": n_epochs,
                   "batch_size": args["batch_size"],
                   "lr": args["lr"]
               }
               )
    for epoch in progress_bar:
        epoch_loss = 0.0
        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            loss = criterion(model(x, A), y)
            epoch_loss += loss.item()
            optim.zero_grad()
            loss.backward()
            optim.step()

            wandb.log({
                "batch loss": loss.item()
            })

        description = f"Epoch {epoch}/{n_epochs} - Training loss: {epoch_loss:.3f}"
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            torch.save(model.state_dict(), checkpoint)
            description += "  -> ✅ Stored best model."

        progress_bar.set_description(description)
    wandb.finish()

    # Testing loop
    with torch.no_grad():
        correct, total = 0, 0
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model = model.eval()
        for batch in test_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            correct += (model(x, A).argmax(1) == y).sum().item()
            total += len(y)
    print(f"\n\nFinal test accuracy: {(correct / total):.2f}%")
    print("Program completed successfully!")


if __name__ == "__main__":
    main()
