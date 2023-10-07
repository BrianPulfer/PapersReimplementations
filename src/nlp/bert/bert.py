"""
Re-implementation of

    BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
                                    Devlin et al. (2018)
                        (https://arxiv.org/pdf/1810.04805.pdf)

on the WikiText-2 dataset.
"""

import os
import random
from argparse import ArgumentParser
from typing import Any

import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

torch.set_float32_matmul_precision("medium")

import pytorch_lightning as pl
import transformers
from datasets import load_dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import BertTokenizer

transformers.logging.set_verbosity_error()

from src.nlp.layers.encoder import EncoderTransformer


class BertDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length, mlm_ratio=0.15, mask_ratio=0.8):
        super(BertDataset, self).__init__()

        # Dataset parameters
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_ratio = mlm_ratio
        self.mask_ratio = mask_ratio

        # MLM distribution
        nothing_prob = 1 - mlm_ratio
        mask_prob = (1 - nothing_prob) * mask_ratio
        different_word_prob = (1 - nothing_prob - mask_prob) / 2
        probs = torch.tensor(
            [nothing_prob, mask_prob, different_word_prob, different_word_prob]
        )
        self.mask_dist = torch.distributions.Multinomial(probs=probs)

        # Tokenizing dataset
        self.dataset = self.dataset.map(
            lambda samples: {"text": [txt for txt in samples["text"] if len(txt) > 0]},
            batched=True,
        )

    def __len__(self):
        return len(self.dataset) - 1

    def __getitem__(self, index):
        # First sentence
        s1 = self.dataset[index]["text"]

        # Next sentence and prediction label
        if torch.rand(1) > 0.5:
            # 50% of the time, pick the real next "sentence"
            s2 = self.dataset[index + 1]["text"]
            nsp_label = 1
        else:
            # The other 50% of the time, pick a random next "sentence"
            idx = random.randint(0, len(self.dataset) - 1)
            s2 = self.dataset[idx]["text"]
            nsp_label = 0

        # Preparing input ids
        tokenizer_out = self.tokenizer(
            s1,
            s2,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        input_ids, segment_idx, attn_mask = (
            tokenizer_out["input_ids"][0],
            tokenizer_out["token_type_ids"][0],
            tokenizer_out["attention_mask"][0],
        )

        # Getting mask indexes
        mask_idx = self.mask_dist.sample((len(input_ids),))

        # Not masking CLS and SEP
        sep_idx = -1
        for i in range(len(segment_idx)):
            if segment_idx[i] == 1:
                sep_idx = i - 1
                break
        mask_idx[0] = mask_idx[-1] = mask_idx[sep_idx] = torch.tensor([1, 0, 0, 0])
        mask_idx = mask_idx.argmax(dim=-1)

        # Getting labels for masked tokens
        mlm_idx = (mask_idx != 0).long()
        mlm_labels = input_ids.clone()

        # Masking input tokens according to strategy
        input_ids[mask_idx == 1] = self.tokenizer.mask_token_id
        input_ids[mask_idx == 2] = torch.randint(0, self.tokenizer.vocab_size, (1,))

        return {
            "input_ids": input_ids,
            "segment_ids": segment_idx,
            "attn_mask": attn_mask,
            "mlm_labels": mlm_labels,
            "mlm_idx": mlm_idx,
            "nsp_labels": nsp_label,
        }

    def tokenize_fn(self, examples):
        ids = [
            self.tokenizer(txt)["input_ids"] for txt in examples["text"] if len(txt) > 0
        ]
        ids = torch.tensor(ids)
        return ids


class Bert(pl.LightningModule):
    DEFAULT_BERT_CONFIG = {
        "lr": 1e-4,
        "betas": (0.9, 0.999),
        "weight_decay": 0.01,
        "max_train_steps": 10_000,
        "warmup_steps": 100,
    }

    def __init__(
        self,
        vocab_size,
        max_len,
        hidden_dim,
        n_heads,
        depth,
        dropout=0.1,
        train_config=None,
    ):
        super(Bert, self).__init__()

        # Saving hyper-parameters so that they are logged
        self.save_hyperparameters()

        # Local parameters
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.depth = depth
        self.dropout = dropout
        self.train_config = Bert.DEFAULT_BERT_CONFIG

        # Schedulers
        self.linear_scheduler = None
        self.warmup_scheduler = None

        # Training config
        if train_config is not None:
            self.train_config.update(train_config)

        # Special tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) / hidden_dim**0.5)
        self.sep_token = nn.Parameter(torch.randn(1, 1, hidden_dim) / hidden_dim**0.5)

        # Embeddings
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embeddings = nn.Parameter(
            torch.randn(max_len, hidden_dim) / hidden_dim**0.5
        )
        self.sentence_embeddings = nn.Parameter(
            torch.randn(2, hidden_dim) / hidden_dim**0.5
        )

        # Transformer and output layer
        self.transformer = EncoderTransformer(
            hidden_dim, n_heads, depth, dropout_p=dropout
        )

        # Next sentence classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 2)
        )

        # Masked language modeling head
        self.mask_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size, bias=True),
        )

    def forward(self, ids, segment_ids=None, attn_mask=None):
        # Embedding
        b, t = ids.shape
        hidden = self.embeddings(ids)
        hidden += self.pos_embeddings[:t].repeat(b, 1, 1)

        if segment_ids is not None:
            hidden += self.sentence_embeddings[segment_ids]

        # Transformer
        hidden = self.transformer(hidden, attn_mask=attn_mask)
        return hidden

    def scheduler_fn(self, step):
        warmup_steps = self.train_config["warmup_steps"]
        max_steps = self.train_config["max_train_steps"]

        if step < warmup_steps:
            return step / warmup_steps
        return 1 - (step - warmup_steps) / (max_steps - warmup_steps)

    def configure_optimizers(self):
        optim = AdamW(
            self.trainer.model.parameters(),
            lr=self.train_config["lr"],
            weight_decay=self.train_config["weight_decay"],
            betas=self.train_config["betas"],
        )

        self.linear_scheduler = LambdaLR(optim, self.scheduler_fn)

        return {"optimizer": optim}

    def get_losses(self, batch):
        # Unpacking batch
        ids = batch["input_ids"]
        segment_ids = batch["segment_ids"]
        attn_mask = batch["attn_mask"]
        mlm_labels = batch["mlm_labels"]
        mlm_idx = batch["mlm_idx"]
        nsp_labels = batch["nsp_labels"]

        # Forward pass
        hidden = self(ids, segment_ids, attn_mask)

        # Classification logits based on the CLS token
        class_preds = self.classifier(hidden[:, 0])
        class_loss = torch.nn.functional.cross_entropy(class_preds, nsp_labels)

        # Masked language modeling loss
        mlm_preds = self.mask_predictor(hidden[mlm_idx == 1])
        mlm_loss = torch.nn.functional.cross_entropy(
            mlm_preds, mlm_labels[mlm_idx == 1]
        )

        # Getting accuracies
        class_acc = (class_preds.argmax(dim=-1) == nsp_labels).float().mean()
        mlm_acc = (mlm_preds.argmax(dim=-1) == mlm_labels[mlm_idx == 1]).float().mean()

        return class_loss, mlm_loss, class_acc, mlm_acc

    def training_step(self, batch, batch_idx):
        # Getting losses
        class_loss, mlm_loss, c_acc, m_acc = self.get_losses(batch)

        # Total loss
        loss = class_loss + mlm_loss

        # Logging
        self.log_dict(
            {
                "lr": self.optimizers().param_groups[0]["lr"],
                "train_loss": loss,
                "train_class_loss": class_loss,
                "train_mlm_loss": mlm_loss,
                "train_class_acc": c_acc,
                "train_mlm_acc": m_acc,
            },
            sync_dist=True,
        )

        return loss

    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if self.warmup_scheduler is not None:
            self.warmup_scheduler.step()

        if self.linear_scheduler is not None:
            self.linear_scheduler.step()

        return super().on_train_batch_end(outputs, batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        # Getting losses
        class_loss, mlm_loss, c_acc, m_acc = self.get_losses(batch)

        # Total loss
        loss = class_loss + mlm_loss

        # Logging
        self.log_dict(
            {
                "val_loss": loss,
                "val_class_loss": class_loss,
                "val_mlm_loss": mlm_loss,
                "val_class_acc": c_acc,
                "val_mlm_acc": m_acc,
            },
            sync_dist=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        # Getting losses
        class_loss, mlm_loss, c_acc, m_acc = self.get_losses(batch)

        # Total loss
        loss = class_loss + mlm_loss

        # Logging
        self.log_dict(
            {
                "test_loss": loss,
                "test_class_loss": class_loss,
                "test_mlm_loss": mlm_loss,
                "test_class_acc": c_acc,
                "test_mlm_acc": m_acc,
            },
            sync_dist=True,
        )

        return loss


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
    callbacks = [ModelCheckpoint(save_dir, monitor="val_loss")]
    trainer = pl.Trainer(
        accelerator="auto",
        strategy="fsdp",
        max_steps=max_train_steps,
        logger=wandb_logger,
        callbacks=callbacks,
        profiler="advanced",
    )
    trainer.fit(bert, train_loader, val_loader)

    # Testing the best model
    bert = Bert.load_from_checkpoint(save_dir)
    trainer.test(bert, test_loader)

    # Unmasking sentences
    # TODO ...


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
    parser.add_argument("--save", type=str, default="models/bert.pt")

    # Other parameters
    parser.add_argument("--seed", type=int, default=0)

    args = vars(parser.parse_args())
    main(args)
