import torch
from accelerate import Accelerator
from transformers import AutoTokenizer, GPT2LMHeadModel, set_seed

from src.nlp.lm_watermarking.watermarking import (
    detect_watermark,
    generate,
    get_perplexities,
)


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
    """Plots the perplexity of the GPT2 model and the z-static for sentences generated with and without watermarking."""
    # Setting seed
    set_seed(0)

    # Device
    device = Accelerator().device

    # Language Model (GPT2)
    model = GPT2Wrapper().to(device)
    vocab_size = model.tokenizer.vocab_size

    # Prior text
    prior = model.tokenizer("Some text to be continued", return_tensors="pt")[
        "input_ids"
    ].to(device)

    # A sentence generated without watermarking
    normal_ids = generate(model, prior, max_length=200, watermark=False)
    n_ppl = get_perplexities(model, normal_ids).item()  # Perplexity
    n_z = detect_watermark(normal_ids, vocab_size).item()  # Z-statistic

    # A sentence generated with watermarking
    watermarked_ids = generate(model, prior, max_length=200, watermark=True)
    w_ppl = get_perplexities(model, watermarked_ids).item()  # Perplexity
    w_z = detect_watermark(watermarked_ids, vocab_size).item()  # Z-statistic

    # Showing non-watermarked text, PPL and probability of watermark
    print(
        f"\n\n\033[92mNormal text (PPL = {n_ppl:.2f}, Z-statistic = {n_z:.2f})\033[0m:\n"
    )
    print(model.tokenizer.decode(normal_ids[0]))

    # Showing watermarked text, PPL and probability of watermark
    print(f"\n\n\033[93mWM text (PPL = {w_ppl:.2f}, Z-statistic = {w_z:.2f})\033[0m:\n")
    print(model.tokenizer.decode(watermarked_ids[0]))


if __name__ == "__main__":
    main()
