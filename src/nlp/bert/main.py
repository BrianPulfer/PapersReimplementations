"""
Re-implementation of

    BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
                                    Devlin et al. (2018)
                        (https://arxiv.org/pdf/1810.04805.pdf)

on the WikiText-2 dataset.
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
from transformers import BertTokenizer

transformers.logging.set_verbosity_error()

from src.nlp.bert.data import BertDataset
from src.nlp.bert.model import Bert


def unmask_sentences(bert, test_file):
    pass


def main(args):
    """
    Train a BERT model on WikiText-2.
    Use the model to "unmask" sentences from a file.
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
    test_file = args["masked_sentences"]
    seed = args["seed"]

    # Setting random seed
    pl.seed_everything(seed)

    # Load the dataset
    wikitext = load_dataset("wikitext", "wikitext-2-v1")
    wikitext.set_format(type="torch", columns=["text"])
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Splitting into train, validation and test sets
    train_set, val_set, test_set = (
        BertDataset(wikitext["train"], tokenizer, max_len),
        BertDataset(wikitext["validation"], tokenizer, max_len),
        BertDataset(wikitext["test"], tokenizer, max_len),
    )

    # Data loaders
    cpus = os.cpu_count()
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=cpus
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=cpus
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=cpus
    )

    # Initialize the model
    vocab_size = tokenizer.vocab_size
    bert = Bert(
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
    wandb_logger = WandbLogger(project="Papers Re-implementations", name="BERT")
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
    trainer.fit(bert, train_loader, val_loader)

    # Testing the best model
    bert = Bert.load_from_checkpoint(os.path.join(save_dir, "best.ckpt"))
    trainer.test(bert, test_loader)

    if test_file is not None and os.path.isfile(test_file):
        # Unmasking sentences
        unmask_sentences(bert, test_file)


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
    parser.add_argument("--save", type=str, default="checkpoints/bert")

    # Testing parameters
    parser.add_argument(
        "--masked_sentences", type=str, default="data/nlp/bert/masked_sentences.txt"
    )

    # Seed
    parser.add_argument("--seed", type=int, default=0)

    args = vars(parser.parse_args())
    main(args)
