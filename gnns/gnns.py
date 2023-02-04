"""Implementation of convolutional, attentional and message-passing GNNs, inspired by the paper
    Everything is Connected: Graph Neural Networks
(https://arxiv.org/pdf/2301.08210v1.pdf)
 
Useful links:
 - Petar Veličković PDF talk: https://petar-v.com/talks/GNN-EEML.pdf
"""

from torchvision.transforms import Compose, ToTensor, Resize, Lambda
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch
import wandb
from argparse import ArgumentParser
from tqdm import tqdm
import warnings


# Definitions
NETWORK_TYPES = ["attn", "conv"]
AGGREGATION_FUNCTIONS = {
    "sum": lambda X, dim=1: torch.sum(X, dim=dim),
    "avg": lambda X, dim=1: torch.mean(X, dim=dim)
}


def parse_args():
    """Parses the program arguments"""
    parser = ArgumentParser()

    # Data arguments
    parser.add_argument(f"--image_size", type=int,
                        help="Size to which reshape CIFAR images. Default is 14 (196 nodes).", default=14)

    # Model arguments
    parser.add_argument(f"--type", type=str, help="Type of the network used. Default is 'attn'",
                        choices=NETWORK_TYPES, default="attn")
    parser.add_argument(f"--aggregation", type=str, help="Aggregation function",
                        choices=list(AGGREGATION_FUNCTIONS.keys()), default="avg")
    parser.add_argument(f"--aggregation_out", type=str, help="Aggregation function for graph classification",
                        choices=list(AGGREGATION_FUNCTIONS.keys()), default="avg")
    parser.add_argument(f"--n_layers", type=int,
                        help="Number of layers of the GNNs", default=8)

    # Training arguments
    parser.add_argument(f"--epochs", type=int,
                        help="Training epochs. Default is 10.", default=10)
    parser.add_argument(f"--lr", type=float,
                        help="Learning rate. Default is 1e-3.", default=0.001)
    parser.add_argument(f"--batch_size", type=int,
                        help="Batch size used for training. Default is 64.", default=64)
    parser.add_argument(f"--checkpoint", type=str,
                        help="Path to model checkpoints. Default is 'gnn.pt'.", default="gnn.pt")

    return vars(parser.parse_args())


def get_device():
    """Gets the CUDA device if available, warns that code will run on CPU only otherwise"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("\nFound GPU: ", torch.cuda.get_device_name(device))
        return device
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("\nFound Apple MPS chip.")

    warnings.warn("\nWARNING: No GPU nor MPS found - Training on CPU.")
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

        self.coefficients = nn.Parameter(torch.ones((n, n)) / n)
        self.ln = nn.LayerNorm(d)
        self.psi = PsiNetwork(d, d)
        self.aggr = aggr

    def forward(self, H, A):
        weights = self.coefficients * A  # (N, N)
        messages = self.psi(self.ln(H))  # (B, N, D)
        messages = torch.einsum(
            "nm, bmd -> bndm", weights, messages)  # (B, N, D, N)
        messages = self.aggr(messages, dim=-1)  # (B, N, D)
        return messages


class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim

    def forward(self, x, mask=None):
        # x has shape (B, N, D)
        attn_cues = ((x @ x.transpose(-2, -1)) /
                     (self.dim**0.5 + 1e-5))  #  (B, N, N)

        if mask is not None:
            attn_cues = attn_cues.masked_fill(mask == 0, float("-inf"))

        attn_cues = attn_cues.softmax(-1)
        return attn_cues


class GraphAttentionLayer(nn.Module):
    """
    Graph Attentional Layer.
    It computes the next hidden states of the edges using attention.
    """

    def __init__(self, n, d, aggr):
        super(GraphAttentionLayer, self).__init__()

        self.aggr = aggr
        self.psi = PsiNetwork(d, d)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(2*d)
        self.sa = Attention(d)

    def forward(self, H, A):
        messages = self.psi(self.ln1(H))  #  (B, N, D)
        attn = self.sa(H, A)  # (B, N, N)

        messages = torch.einsum("bnm, bmd -> bndm", attn,
                                messages)  #  (B, N, D, N)
        messages = self.aggr(messages, dim=-1)  # (B, N, D)
        return messages


class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network class."""

    _NET_TYPE_TO_LAYER = {
        "attn": GraphAttentionLayer,
        "conv": GraphConvLayer
    }

    def _get_phi_net(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU(),
            nn.Linear(dim_out, dim_out)
        )

    def __init__(self, net_type, n_layers, n, d_in, d_hidden, d_out, aggr, aggr_out):
        super(GraphNeuralNetwork, self).__init__()

        assert net_type in NETWORK_TYPES, f"ERROR: GNN type {net_type} not supported. Pick one of {NETWORK_TYPES}"

        self.net_type = net_type
        self.n_layers = n_layers
        self.encoding = nn.Linear(d_in, d_hidden)
        self.layers = nn.ModuleList(
            [self._NET_TYPE_TO_LAYER[net_type](n, d_hidden, AGGREGATION_FUNCTIONS[aggr])
             for _ in range(n_layers)])

        self.phi_nets = nn.ModuleList(
            [self._get_phi_net(2*d_hidden, d_hidden) for _ in range(n_layers)])

        self.out_aggr = AGGREGATION_FUNCTIONS[aggr_out]
        self.out_mlp = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out)
        )

    def forward(self, X, A):
        # X has shape (B, N, D) and represents the edges.
        # A is binary with shape (N, N) and represents the adjacency matrix.
        H = self.encoding(X)
        for l, p in zip(self.layers, self.phi_nets):
            messages = l(H, A)
            H = H + p(torch.cat((H, messages), dim=-1))
        return self.out_mlp(self.out_aggr(H))


def main():
    # Parsing arguments
    args = parse_args()
    print("Launched program with the following arguments:", args)

    # Getting device
    device = get_device()

    # Loading data
    # We reshape the image such that each pixel is an edge with C features.
    img_size = args["image_size"]
    transform = Compose([
        ToTensor(),
        Resize((img_size, img_size)),
        # (C, H, W) -> (H*W, C).
        Lambda(lambda x: x.flatten().reshape(-1, 1))
    ])
    train_set = MNIST("./../datasets", train=True,
                      transform=transform, download=True)
    test_set = MNIST("./../datasets", train=False,
                     transform=transform, download=True)

    train_loader = DataLoader(
        train_set, batch_size=args["batch_size"], shuffle=True)
    test_loader = DataLoader(
        test_set, batch_size=args["batch_size"], shuffle=False)

    # Building the Neighbourhood matrix (1024 x 1024) for all "graphs" (images of size 32x32)
    A = torch.zeros((img_size**2, img_size**2)).to(device)
    nums = torch.arange(img_size**2).reshape((img_size, img_size))
    for i in range(img_size):
        for j in range(img_size):
            start_x, start_y = i-1 if i > 0 else 0, j-1 if j > 0 else 0
            neighbours = nums[start_x: i+2, start_y: j+2].flatten()

            for n in neighbours:
                A[i*img_size + j, n] = A[n, i*img_size + j] = 1

    # Creating model
    # Number of edges, edge dimensionality, hidden dimensionality and number of output classes
    n, d, h, o = img_size**2, 1, 16, 10
    model = GraphNeuralNetwork(args["type"], args["n_layers"], n, d,
                               h, o, aggr=args["aggregation"], aggr_out=args["aggregation_out"])

    # Training loop
    n_epochs = args["epochs"]
    checkpoint = args["checkpoint"] if args["checkpoint"] else f"({args['type']}_gnn).pt"
    optim = Adam(model.parameters(), args["lr"])
    criterion = nn.CrossEntropyLoss()
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
               })

    for epoch in progress_bar:
        epoch_loss = 0.0
        for batch in tqdm(train_loader, leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)

            loss = criterion(model(x, A), y)
            epoch_loss += loss.item() / len(train_loader)
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

        wandb.log({"epoch loss": epoch_loss})
        progress_bar.set_description(description)
    wandb.finish()

    # Testing loop
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for batch in test_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            correct += (model(x, A).argmax(1) == y).sum().item()
            total += len(y)
    print(f"\n\nFinal test accuracy: {(correct / total * 100):.2f}%")
    print("Program completed successfully!")


if __name__ == "__main__":
    main()
