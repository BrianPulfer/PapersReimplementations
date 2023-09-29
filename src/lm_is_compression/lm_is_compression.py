"""Re-implementation of

            Language Modeling Is Compression
           (https://arxiv.org/abs/2309.10668)

by Delétang, Ruoss et. al.
"""

from argparse import ArgumentParser

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_reproducibility(seed=0):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ArithmeticEncoder:
    """The arithmetic encoder converts a sequence of tokens into a single
    number according to the probability distribution of the next token given
    by the language model."""

    def __init__(self, model: nn.Module, bos_token_id: int):
        self.bos_token_id = bos_token_id

        self.model = model.eval()
        self.softmax = nn.Softmax(dim=-1)

    def __call__(self, ids: torch.Tensor) -> bytes:
        return self.encode(ids)

    def highs_lows_to_lambdas(
        self, highs: torch.Tensor, lows: torch.Tensor
    ) -> torch.Tensor:
        # Now returning the midpoints
        # TODO: Return the numbers with the shortest binary representation
        return (highs + lows) / 2

    @torch.no_grad()
    def encode(self, ids: torch.Tensor) -> bytes:
        """Encode a sequence of tokens into a binary sequence of bits.
        The encoding is done by finding the scalar number in range [0, 1]
        using arithmetic encoding based on the probability distribution of the
        next token given by the language model.

        Args:
            ids (torch.Tensor): Two-dimensional tensor of token ids. Omit the BOS token.

        Returns:
            bytes: The encoded sequence of bits
        """
        # Appending the BOS token to the beginning of each sequence
        bos_tokens = torch.full(
            (ids.shape[0], 1), self.bos_token_id, dtype=torch.long, device=ids.device
        )
        ids = torch.cat([bos_tokens, ids], dim=1)
        N, T = ids.shape

        # Getting the probabilities of the next token
        logits = self.model(ids)["logits"]
        probs = self.softmax(logits)

        # Find the lambda number for each sequence
        lows, highs = torch.zeros(N, dtype=torch.double), torch.ones(
            N, dtype=torch.double
        )
        for i in range(T - 1):
            intervals = highs - lows

            # Getting cumulative probabilities
            # TODO: Parallelize this loop
            c_probs = torch.empty(N)
            n_probs = torch.empty(N)
            for j in range(N):
                c_probs[j] = probs[j, i, : ids[j, i + 1]].sum()
                n_probs[j] = probs[j, i, : ids[j, i + 1] + 1].sum()

            # Updating lows and highs
            highs = lows + intervals * n_probs
            lows = lows + intervals * c_probs

        # Return the lambda numbers
        return self.highs_lows_to_lambdas(highs, lows)

    @torch.no_grad()
    def decode(self, lambdas: torch.Tensor, atol=1e-30) -> torch.Tensor:
        """Undo the encoding and, given scalar lambdas, return the original input ids."""
        N, dev = lambdas.shape[0], lambdas.device
        ids = torch.full((N, 1), self.bos_token_id, dtype=torch.long, device=dev)

        # Recovering the ids
        lows, highs = torch.zeros(N, dtype=torch.double, device=dev), torch.ones(
            N, dtype=torch.double, device=dev
        )
        while not torch.allclose(
            self.highs_lows_to_lambdas(highs, lows), lambdas, atol=atol
        ):
            probs = self.softmax(self.model(ids)["logits"][:, -1])

            next_ids = torch.empty(N, dtype=torch.long, device=lambdas.device)
            for i in range(N):
                lamb = lambdas[i]
                low, high = lows[i], highs[i]
                for j in range(probs.shape[1]):
                    l = low + (high - low) * probs[i, :j].sum()
                    u = low + (high - low) * probs[i, : j + 1].sum()

                    if l <= lamb < u:
                        highs[i], lows[i] = u, l
                        next_ids[i] = j
                        break

            ids = torch.cat([ids, next_ids.unsqueeze(1)], dim=1)

        return ids

    def to(self, device):
        self.model.to(device)
        self.softmax.to(device)
        return self


def main(args):
    # Getting program parameters
    model_ckpt = args["model"]
    seed = args["seed"]

    # Setting reproducibility
    set_reproducibility(seed)

    # Preparing sentences to encode
    sentences = ["The quick brown fox jumps over the lazy dog."]

    # Loading model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_ckpt, torch_dtype=torch.float32, device_map="auto"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    # Encoding sentences
    ids = tokenizer(sentences, return_tensors="pt", padding=True)["input_ids"].cuda()
    encoder = ArithmeticEncoder(model, tokenizer.bos_token_id)
    encoded = encoder(ids)
    decoded = encoder.decode(encoded.cuda())

    # Printing results
    print("\n\nOriginal sentences:", sentences)
    print(
        "Decoded  sentences:", tokenizer.batch_decode(decoded, skip_special_tokens=True)
    )

    print("\n\nOriginal ids:", ids)
    print("Decoded  ids:", decoded[:, 1:])

    print("\n\nEncoded sentences (as scalars):", encoded)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-1.4b-v0")
    parser.add_argument("--seed", type=int, default=0)
    args = vars(parser.parse_args())
    print(args)
    main(args)
