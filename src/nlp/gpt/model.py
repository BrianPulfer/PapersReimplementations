from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.nlp.layers.decoder import DecoderTransformer
from src.nlp.layers.embeddings import get_learnable_embedding


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

        # Schedulers
        self.linear_scheduler = None
        self.warmup_scheduler = None

        # Training config
        if train_config is not None:
            self.train_config.update(train_config)

        # Embeddings
        self.embeddings = get_learnable_embedding(
            vocab_size, hidden_dim
        )  # nn.Embedding(vocab_size, hidden_dim)
        self.pos_embeddings = get_learnable_embedding(
            max_len, hidden_dim
        )  # nn.Embedding(max_len, hidden_dim)

        # Transformer and output layer
        self.transformer = DecoderTransformer(
            hidden_dim, n_heads, depth, dropout_p=dropout
        )

        # Next token classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, vocab_size)
        )

    def forward(self, ids, attn_mask=None):
        # Embedding
        b, t = ids.shape
        hidden = self.embeddings(ids)
        hidden += self.pos_embeddings(torch.arange(t, device=ids.device)).repeat(
            b, 1, 1
        )

        # Transformer
        hidden = self.transformer(hidden, self_attn_mask=attn_mask)

        # Classification
        return self.classifier(hidden), hidden

    def get_losses(self, batch):
        # Unpacking batch
        ids = batch["input_ids"]
        attn_mask = batch["attention_mask"]

        # Running forward
        b, t = ids.shape
        out, _ = self(ids, attn_mask.repeat(1, t).reshape(b, t, t).tril())

        # Computing cross-entropy loss
        preds, labels = out[:, :-1], ids[:, 1:]
        preds, labels = preds[attn_mask[:, :-1] == 1], labels[attn_mask[:, :-1] == 1]
        ce_loss = nn.functional.cross_entropy(
            preds.reshape(-1, self.vocab_size), labels.reshape(-1)
        )

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

    def generate(self, input_ids, max_len):
        # Predicting next token until max_len
        remaining = max_len - input_ids.shape[1]
        for _ in range(remaining):
            # Running GPT
            preds = self(input_ids)[0]

            # Getting probability of next token
            probs = preds[:, -1, :].softmax(dim=-1)

            # Sampling next token with multinomial sampling
            next_token = torch.multinomial(probs, num_samples=1)

            # Adding token to input_ids
            input_ids = torch.cat((input_ids, next_token), dim=-1)
        return input_ids
