"""
Re-implementation of

                      Language Models are Few-Shot Learners
                              Brown et al. (2020)
                        (https://arxiv.org/abs/2005.14165)

on the WikiPedia dataset.
"""

import os
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision("medium")

import pytorch_lightning as pl
import transformers
from datasets import load_dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import GPT2Tokenizer

transformers.logging.set_verbosity_error()

from src.nlp.gpt.model import GPT


def continue_sentences(gpt, tokenizer, max_len, file_path):
    """Uses the gpt model to continue sentences from a file. Prints the continued sentences."""
    file = open(file_path, "r")
    lines = file.readlines()
    lines = [line if not line.endswith("\n") else line[:-1] for line in lines]
    file.close()

    gpt = gpt.cuda()
    for i, line in enumerate(lines):
        # Preparing input
        input_ids = tokenizer(
            line,
            return_tensors="pt",
            max_length=max_len,
        )["input_ids"].cuda()

        # Generating sentence
        all_ids = gpt.generate(input_ids, max_len)

        # Decoding the sentence
        sentence = tokenizer.decode(all_ids.squeeze().tolist())
        print(f"\n\nSentence {i+1}:")
        print(f"\tOriginal: {line}\n\tCompleted: {sentence}")


def main(args):
    """
    Train a GPT model on Wikipedia.
    Use the model to continue sentences from a file.
    """
    # Unpacking parameters
    n_blocks = args["n_blocks"]
    n_heads = args["n_heads"]
    hidden_dim = args["hidden_dim"]
    dropout = args["dropout"]
    max_len = args["max_len"]
    batch_size = args["batch_size"]
    max_train_steps = args["max_train_steps"]
    lr = args["lr"]
    weight_decay = args["weight_decay"]
    warmup_steps = args["warmup_steps"]
    save_dir = args["save"]
    file_path = args["prompts"]
    seed = args["seed"]

    # Setting random seed
    pl.seed_everything(seed)

    # Load the dataset (wikipedia only has 'train', so we split it ourselves)
    train_set = load_dataset("wikipedia", "20220301.en", split="train[:80%]")
    val_set = load_dataset("wikipedia", "20220301.en", split="train[80%:90%]")
    test_set = load_dataset("wikipedia", "20220301.en", split="train[90%:]")

    # Setting format to torch (not striclty necessary)
    # train_set.set_format(type="torch", columns=["text"])
    # val_set.set_format(type="torch", columns=["text"])
    # test_set.set_format(type="torch", columns=["text"])

    # Loading the GPT2 tokenizer
    added_pad_token = False
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        added_pad_token = True

    # Tokenizing the whole dataset
    def collate(batch):
        return tokenizer(
            [sample["text"] for sample in batch],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )

    # Data loaders
    cpus = os.cpu_count()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cpus,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cpus,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cpus,
        collate_fn=collate,
    )

    # Initialize the model
    vocab_size = tokenizer.vocab_size + 1 if added_pad_token else tokenizer.vocab_size
    gpt = GPT(
        vocab_size,
        max_len,
        hidden_dim,
        n_heads,
        n_blocks,
        dropout=dropout,
        train_config={
            "lr": lr,
            "weight_decay": weight_decay,
            "max_train_steps": max_train_steps,
            "warmup_steps": warmup_steps,
        },
    )

    # Training
    wandb_logger = WandbLogger(project="Papers Re-implementations", name="GPT")
    wandb_logger.experiment.config.update(args)
    callbacks = [ModelCheckpoint(save_dir, monitor="val_loss", filename="best")]
    trainer = pl.Trainer(
        accelerator="auto",
        strategy="ddp",  # State dict not saved with fsdp for some reason
        max_steps=max_train_steps,
        logger=wandb_logger,
        callbacks=callbacks,
        profiler="simple",
    )
    trainer.fit(gpt, train_loader, val_loader)

    # Testing the best model
    gpt = GPT.load_from_checkpoint(os.path.join(save_dir, "best.ckpt"))
    trainer.test(gpt, test_loader)

    if file_path is not None and os.path.isfile(file_path):
        # Generating sentences from prompts
        continue_sentences(gpt, tokenizer, max_len, file_path)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Model hyper-parameters
    parser.add_argument("--n_blocks", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=128)

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_train_steps", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--save", type=str, default="checkpoints/gpt")

    # Testing parameters
    parser.add_argument("--prompts", type=str, default="data/nlp/gpt/prompts.txt")

    # Seed
    parser.add_argument("--seed", type=int, default=0)

    args = vars(parser.parse_args())
    print(args)
    main(args)
