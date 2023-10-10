from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.nlp.layers.encoder import EncoderTransformer


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

        # Embeddings
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embeddings = nn.Embedding(max_len, hidden_dim)
        self.sentence_embeddings = nn.Embedding(2, hidden_dim)

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
        hidden += self.pos_embeddings(torch.arange(t, device=ids.device)).repeat(
            b, 1, 1
        )

        if segment_ids is not None:
            hidden += self.sentence_embeddings(segment_ids)
        else:
            hidden += self.sentence_embeddings(
                torch.zeros(1, dtype=torch.long, device=ids.device)
            ).repeat(b, 1, 1)

        # Transformer
        hidden = self.transformer(hidden, attn_mask=attn_mask)

        # Classification head based on CLS token
        class_preds = self.classifier(hidden[:, 0])

        # Masked language modeling head
        mlm_preds = self.mask_predictor(hidden)

        return hidden, class_preds, mlm_preds

    def get_losses(self, batch):
        # Unpacking batch
        ids = batch["input_ids"]
        segment_ids = batch["segment_ids"]
        attn_mask = batch["attention_mask"]
        mlm_labels = batch["mlm_labels"]
        mlm_idx = batch["mlm_idx"]
        nsp_labels = batch["nsp_labels"]

        # Running forward
        _, class_preds, mlm_preds = self(ids, segment_ids, attn_mask)
        mlm_preds, mlm_labels = mlm_preds[mlm_idx == 1], mlm_labels[mlm_idx == 1]

        # Classification loss
        class_loss = torch.nn.functional.cross_entropy(class_preds, nsp_labels)

        # Masked language modeling loss
        mlm_loss = torch.nn.functional.cross_entropy(mlm_preds, mlm_labels)

        # Getting accuracies
        class_acc = (class_preds.argmax(dim=-1) == nsp_labels).float().mean()
        mlm_acc = (mlm_preds.argmax(dim=-1) == mlm_labels).float().mean()

        return class_loss, mlm_loss, class_acc, mlm_acc

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

    def validation_step(self, batch, batch_idx):
        # Getting losses & accuracies
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
        # Getting losses & accuracies
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
