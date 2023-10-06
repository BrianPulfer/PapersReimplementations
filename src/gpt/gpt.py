"""Implementation of a character-level GPT model from the paper:
    Attention is all you need
(https://arxiv.org/abs/1706.03762)

Useful links:
 - Karpathy's amazing youtube video: https://www.youtube.com/watch?v=kCc8FmEb1nY
'"""

import warnings
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

import wandb

# Definitions
NAME_TO_PARAMS = {
    "karpathy": {"depth": 6, "embed_dim": 384, "n_heads": 6},
    "tiny": {"depth": 12, "embed_dim": 192, "n_heads": 3},
    "small": {"depth": 12, "embed_dim": 384, "n_heads": 6},
    "base": {"depth": 12, "embed_dim": 768, "n_heads": 12},
    "large": {"depth": 24, "embed_dim": 1024, "n_heads": 16},
    "huge": {"depth": 32, "embed_dim": 1280, "n_heads": 16},
}


def parse_args():
    """Parses the arguments to train a GPT model and write down on a file the generated samples."""

    parser = ArgumentParser()

    # Data parameters
    parser.add_argument(
        f"--file",
        type=str,
        help="Path to the text file to train on.",
        default="1984_George_Orwell.txt",
    )
    parser.add_argument(
        f"--context_length",
        type=int,
        help="Size of the context. Default is 256.",
        default=256,
    )

    # Model parameters
    parser.add_argument(
        f"--model",
        type=str,
        help="Size of the model to use. Default is 'tiny'.",
        default=None,
    )
    parser.add_argument(
        f"--depth",
        type=int,
        help="If model is not specified -> number of transformer decoder blocks.",
        default=12,
    )
    parser.add_argument(
        f"--embed_dim",
        type=int,
        help="If model is not specified -> hidden transformer dimensionality.",
        default=192,
    )
    parser.add_argument(
        f"--n_heads",
        type=int,
        help="If model is not specified -> number of transformer heads.",
        default=3,
    )

    # Training parameters
    parser.add_argument(
        f"--experiment_name",
        type=str,
        help="Name of the experiment to be logged with W&B",
        default="GPT-Model",
    )
    parser.add_argument(
        f"--max_iters",
        type=int,
        help="Maximum training iterations. Default is 5000.",
        default=5000,
    )
    parser.add_argument(
        f"--batch_size", type=int, help="Batch size for training.", default=64
    )
    parser.add_argument(
        f"--lr", type=int, help="Learning rate for training.", default=3e-4
    )
    parser.add_argument(
        f"--vp",
        type=float,
        help="Validation percentage. Default is 0.2 (20%).",
        default=0.2,
    )
    parser.add_argument(
        f"--checkpoint",
        type=str,
        help="Path to the model checkpoint.",
        default="GPT_ckpt.pt",
    )

    # Generation parameters
    parser.add_argument(
        f"--n_gen_samples", type=int, help="Number of samples to generate.", default=50
    )
    parser.add_argument(
        f"--gen_samples_path",
        type=str,
        help="Path to the file where generated samples will be stored.",
        default="generated.txt",
    )

    return vars(parser.parse_args())


def get_device():
    """Gets the CUDA device if available, warns that code will run on CPU only otherwise"""

    if torch.cuda.is_available:
        device = torch.device("cuda")
        print("Found GPU: ", torch.cuda.get_device_name(device))
        return device

    warnings.warn("WARNING: No GPU found - Training on CPU.")
    return torch.device("cpu")


def read_text(path):
    """Reads all the content of the file."""

    file = open(path, "r", encoding="utf-8")
    text = file.read()
    file.close()
    return text


def get_batch(string, ctoi, batch_size, context_length, device="cpu"):
    """Returns a batch as a tuple of (x, y) where x is a (B,T) tensor and y is a (B, T, vocab_size)."""

    start_idxs = torch.randint(0, len(string) - context_length - 1, (batch_size,))

    x = []
    y = []
    for idx in start_idxs:
        all_chars = [ctoi[c] for c in string[idx : idx + context_length + 1]]
        x.append(all_chars[:-1])
        y.append(all_chars[1:])

    # Building torch tensors
    x, y = torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    # Converting labels to one-hot encodings (B, T) -> (B, T, vocab_size)
    # vocab_size = len(list(ctoi.keys()))
    # y = nn.functional.one_hot(y, num_classes=vocab_size)

    return x.to(device), y.to(device)


class SelfAttention(nn.Module):
    """Self Attention (SA) module"""

    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim

        self.to_qkv = nn.Linear(embed_dim, 3 * embed_dim)

    def forward(self, x):
        B, T, C = x.shape

        # Getting Queries, Keys and Values
        q, k, v = self.to_qkv(x).chunk(3, -1)

        # Computing the attention cues.
        attn = (q @ k.transpose(-2, -1)) / (
            self.embed_dim**0.5 + 1e-5
        )  # (B, T, C) @ (B, C, T) = (B, T, T)
        attn = attn.masked_fill(
            torch.tril(torch.ones(T, T)).to(x.device) == 0, float("-inf")
        )  # Masking future characters (right-triangular part of attn) with -inf
        attn = attn.softmax(
            -1
        )  # Normalizing the contributions of the attention cues (to sum to one)
        return attn @ v


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self Attention (MHSA) module"""

    def __init__(self, embed_dim, n_heads, dropout=0.2):
        super(MultiHeadSelfAttention, self).__init__()

        assert (
            embed_dim % n_heads == 0
        ), f"ERROR: Cannot entirely divide dimension {embed_dim} into {n_heads} heads."
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.self_attentions = nn.ModuleList(
            [SelfAttention(embed_dim // n_heads) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat(
            [
                sa(x[:, :, head * self.head_dim : (head + 1) * self.head_dim])
                for head, sa in enumerate(self.self_attentions)
            ],
            dim=-1,
        )  # Concatenating the results of all self attention heads
        return self.dropout(self.proj(out))


class DecoderBlock(nn.Module):
    """Decoder block of the transformer model"""

    def __init__(self, embed_dim, n_heads, dropout=0.2):
        super(DecoderBlock, self).__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

        self.mhsa = MultiHeadSelfAttention(self.embed_dim, self.n_heads)

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.mlp(self.ln1(x))
        x = x + self.mhsa(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, context_length, depth, embed_dim, n_heads):
        super(Transformer, self).__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.depth = depth
        self.embed_dim = embed_dim
        self.n_heads = n_heads

        # Text and Positional embeddings
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(context_length, embed_dim)

        # Decoder blocks
        self.blocks = nn.ModuleList(
            [DecoderBlock(embed_dim, n_heads) for _ in range(depth)]
        )

        # Final layer norm and linear layer to get final prediction
        self.ln = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, idxs):
        B, T = idxs.shape

        # Getting text and position embeddings and using them as input to decoder blocks
        te = self.text_embedding(idxs)
        pe = (
            self.pos_embedding(torch.tensor(list(range(T))).to(idxs.device))
            .unsqueeze(0)
            .repeat_interleave(B, 0)
        )

        # Running input through decoder blocks
        x = te + pe
        for block in self.blocks:
            x = block(x)

        # Producing final prediction for next character
        return self.linear(self.ln(x))


def training_loop(
    model,
    optimizer,
    criterion,
    batch_size,
    max_iterations,
    train_string,
    val_string,
    ctoi,
    checkpoint_path,
    name="GPT",
    log=True,
    device="cpu",
):
    """Trains the GPT model"""

    if log:
        # Starting a new Weights & Biases run
        wandb.init(
            project="Papers Re-implementations",
            name=name,
            config={
                "depth": model.depth,
                "embed_dim": model.embed_dim,
                "n_heads": model.n_heads,
                "max_iterations": max_iterations,
                "batch_size": batch_size,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    lowest_val_loss = float("inf")
    model = model.to(device)

    print("\nTraining started")
    progress_bar = tqdm(range(1, max_iterations + 1), desc="Training")
    for iteration in progress_bar:
        # Training step
        model.train()
        x, y = get_batch(train_string, ctoi, batch_size, model.context_length, device)
        y_hat = model(x)
        B, T, C = y_hat.shape
        train_loss = criterion(y_hat.view(B * T, C), y.view(B * T))
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Evaluation step
        with torch.no_grad():
            model.eval()
            x, y = get_batch(val_string, ctoi, batch_size, model.context_length, device)
            y_hat = model(x)
            B, T, C = y_hat.shape
            val_loss = criterion(model(x).view(B * T, C), y.view(B * T))

            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                torch.save(model.state_dict(), checkpoint_path)

        if log:
            # Logging information to W&B
            wandb.log(
                {"training loss": train_loss.item(), "validation loss": val_loss.item()}
            )

        progress_bar.set_description(
            f"Iteration {iteration}/{max_iterations}  ----- Train loss: {train_loss.item():.3f}   -----   Validation loss: {val_loss.item():.3f}-- "
        )

    if log:
        # Finishing W&B session
        wandb.finish()


@torch.no_grad()
def generate_text(
    model,
    n,
    context_length,
    itoc,
    checkpoint_path,
    file_path="generated.txt",
    device="cpu",
    write=True,
):
    """Generates and stores into a file text obtained with the trained model"""

    # Loading trained model
    state_dict = torch.load(checkpoint_path, map_location=device)
    model = model.to(device).eval()
    model.load_state_dict(state_dict)

    # Getting new text in batch
    vocab_size = len(list(itoc.keys()))
    output = torch.randint(0, vocab_size, (n, 1)).to(device)

    for _ in range(context_length - 1):
        probs = model(output)[:, -1].softmax(-1)
        next_chars = torch.multinomial(probs, 1)
        output = torch.cat((output, next_chars), dim=-1)

    # Converting the predictions into sentences
    generated_samples = [[itoc[c.item()] for c in o] for o in output.cpu()]

    if write:
        # Storing the sentences into the file
        file = open(file_path, "w")
        for i, sample in enumerate(generated_samples):
            file.write(f"################ SAMPLE {i+1} ################\n")
            file.write("".join(sample))
            file.write("\n\n\n")

    return generated_samples


def main():
    # Getting program arguments
    args = parse_args()

    # Creating training and validation data
    text = read_text(args["file"])
    chars = sorted(list(set(c for c in text)))
    ctoi = {c: i for i, c in enumerate(chars)}  # Char-to-index
    itoc = {i: c for i, c in enumerate(chars)}  # Index-to-char

    split_idx = int((1 - args["vp"]) * len(text))
    train_string, val_string = text[:split_idx], text[split_idx:]
    print(
        f"Text is composed of {len(text)} characters: {len(train_string)} for training and {len(val_string)} for validation."
    )

    # Getting the device
    device = get_device()

    # Creating the model
    model_args = (
        NAME_TO_PARAMS[args["model"]]
        if args["model"]
        else {k: args[k] for k in ["depth", "embed_dim", "n_heads"]}
    )
    model = Transformer(
        len(chars),
        args["context_length"],
        model_args["depth"],
        model_args["embed_dim"],
        model_args["n_heads"],
    )
    optimizer = Adam(model.parameters(), lr=args["lr"])
    print(
        f"\nCreated transformer model:\n\t{model_args}\n\tnumber of parameters: {sum([p.numel() for p in model.parameters()])}"
    )

    # Training loop
    training_loop(
        model,
        optimizer,
        nn.CrossEntropyLoss(),
        args["batch_size"],
        args["max_iters"],
        train_string,
        val_string,
        ctoi,
        args["checkpoint"],
        name="GPT",
        device=device,
    )

    # Text generation
    generate_text(
        model,
        args["n_gen_samples"],
        args["context_length"],
        itoc,
        args["checkpoint"],
        args["gen_samples_path"],
        device=device,
    )
    print("\n\n\nProgram completed successfully!")


if __name__ == "__main__":
    main()
