from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.nlp.layers.decoder import DecoderTransformer


class GPT(pl.LightningModule):
    DEFAULT_GPT_CONFIG = {
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
        super(GPT, self).__init__()

        # Saving hyper-parameters so that they are logged
        self.save_hyperparameters()

        # Local parameters
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.depth = depth
        self.dropout = dropout
        self.train_config = GPT.DEFAULT_GPT_CONFIG
        self.cross_entropy = nn.CrossEntropyLoss()

        # Schedulers
        self.linear_scheduler = None
        self.warmup_scheduler = None

        # Training config
        if train_config is not None:
            self.train_config.update(train_config)

        # Embeddings
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embeddings = nn.Parameter(
            torch.randn(max_len, hidden_dim) / hidden_dim**0.5
        )

        # Transformer and output layer
        self.transformer = DecoderTransformer(
            hidden_dim, n_heads, depth, dropout_p=dropout
        )

        # Next token classifier
        self.classifier = nn.LayerNorm(hidden_dim)

    def forward(self, ids, attn_mask=None):
        # Embedding
        b, t = ids.shape
        hidden = self.embeddings(ids)
        hidden += self.pos_embeddings[:t].repeat(b, 1, 1)

        # Transformer
        hidden = self.transformer(hidden, self_attn_mask=attn_mask)

        # Classification
        return self.classifier(hidden)

    def get_losses(self, batch):
        # Unpacking batch
        ids = batch["input_ids"]
        attn_mask = batch["attention_mask"]

        # Running forward
        out = self(ids, attn_mask)

        # Computing cross-entropy loss
        preds, labels = out[:, :-1], ids[:, 1:]
        ce_loss = self.cross_entropy(preds.view(-1, self.vocab_size), labels.view(-1))

        accuracy = (preds.argmax(dim=-1) == labels).float().mean()
        perplexity = torch.exp(ce_loss)

        return ce_loss, accuracy, perplexity

    def configure_optimizers(self):
        optim = AdamW(
            self.trainer.model.parameters(),
            lr=self.train_config["lr"],
            weight_decay=self.train_config["weight_decay"],
            betas=self.train_config["betas"],
        )

        self.linear_scheduler = LambdaLR(optim, self.scheduler_fn)
        return {"optimizer": optim}

    def scheduler_fn(self, step):
        warmup_steps = self.train_config["warmup_steps"]
        max_steps = self.train_config["max_train_steps"]

        if step < warmup_steps:
            return step / warmup_steps
        return 1 - (step - warmup_steps) / (max_steps - warmup_steps)

    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if self.linear_scheduler is not None:
            self.linear_scheduler.step()

        return super().on_train_batch_end(outputs, batch, batch_idx)

    def training_step(self, batch, batch_idx):
        # Getting losses & accuracies
        ce_loss, accuracy, perplexity = self.get_losses(batch)

        # Logging
        self.log_dict(
            {
                "lr": self.optimizers().param_groups[0]["lr"],
                "train_loss": ce_loss,
                "train_acc": accuracy,
                "train_ppl": perplexity,
            },
            sync_dist=True,
        )

        return ce_loss

    def validation_step(self, batch, batch_idx):
        # Getting losses & accuracies
        ce_loss, accuracy, perplexity = self.get_losses(batch)

        # Logging
        self.log_dict(
            {
                "lr": self.optimizers().param_groups[0]["lr"],
                "val_loss": ce_loss,
                "val_acc": accuracy,
                "val_ppl": perplexity,
            },
            sync_dist=True,
        )

        return ce_loss

    def test_step(self, batch, batch_idx):
        # Getting losses & accuracies
        ce_loss, accuracy, perplexity = self.get_losses(batch)

        # Logging
        self.log_dict(
            {
                "lr": self.optimizers().param_groups[0]["lr"],
                "test_loss": ce_loss,
                "test_acc": accuracy,
                "test_ppl": perplexity,
            },
            sync_dist=True,
        )

        return ce_loss
