import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams.update({"font.size": 22})
from argparse import ArgumentParser

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, set_seed

from src.nlp.lm_watermarking.watermarking import (
    detect_watermark,
    generate,
    get_perplexities,
)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--n_sentences", type=int, default=128, help="Number of sentences to generate"
    )
    parser.add_argument(
        "--seq_len", type=int, default=200, help="Length of the generated sentences"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for the generation"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.5, help="Green list proportion"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=2,
        help="Amount to add to the logits of the model when watermarking",
    )
    parser.add_argument(
        "--device", type=int, default=0, help="Device to use for generation"
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the generation")

    return vars(parser.parse_args())


class GPT2Wrapper(torch.nn.Module):
    """A wrapper around the GPT2 model to take ids as input and return logits as output."""

    def __init__(self):
        super(GPT2Wrapper, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

    def forward(self, input_ids):
        outputs = self.model(input_ids)
        return outputs.logits


def main():
    # Plotting parameters
    args = parse_args()
    n_sentences = args["n_sentences"]
    seq_len = args["seq_len"]
    batch_size = args["batch_size"]
    gamma = args["gamma"]
    delta = args["delta"]
    seed = args["seed"]

    # Setting seed
    set_seed(seed)

    # Device
    device = torch.device(
        "cuda:" + str(args["device"]) if torch.cuda.is_available() else "cpu"
    )

    # Model
    model = GPT2Wrapper().to(device)
    vocab_size = model.tokenizer.vocab_size

    # Prior text (BOS token)
    prior = (
        (model.tokenizer.bos_token_id * torch.ones((n_sentences, 1))).long().to(device)
    )

    # Collecting generations with and without watermark
    regular_ppls, regular_z_scores = [], []
    watermarked_ppls, watermarked_z_scores = [], []
    for i in tqdm(range(0, n_sentences, batch_size), desc="Generating sentences"):
        batch = prior[i : i + batch_size]

        # Regular sentences
        regular = generate(model, batch, max_length=seq_len, watermark=False)
        regular_ppls.extend(get_perplexities(model, regular).tolist())
        regular_z_scores.extend(detect_watermark(regular, vocab_size).tolist())

        # Watermarked sentences
        watermarked = generate(
            model, batch, max_length=seq_len, watermark=True, gamma=gamma, delta=delta
        )
        watermarked_ppls.extend(get_perplexities(model, watermarked).tolist())
        watermarked_z_scores.extend(detect_watermark(watermarked, vocab_size).tolist())

    # Scatter plot of perplexity vs z-score
    plt.figure(figsize=(10, 10))
    plt.scatter(regular_ppls, regular_z_scores, label="Regular")
    plt.scatter(watermarked_ppls, watermarked_z_scores, label="Watermarked")
    plt.legend()
    plt.title("Perplexity vs Z-score")
    plt.xlabel("Perplexity")
    plt.ylabel("Z-score")
    plt.savefig(
        f"perplexity_vs_zscore_(n={n_sentences}, seq_len={seq_len}, gamma={gamma}, delta={delta}, seed={seed}).png"
    )
    plt.show()
    print("Program completed successfully!")


if __name__ == "__main__":
    main()
