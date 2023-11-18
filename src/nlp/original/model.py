from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from src.nlp.layers.decoder import DecoderTransformer
from src.nlp.layers.embeddings import get_learnable_embedding
from src.nlp.layers.encoder import EncoderTransformer


class EncoderDecoderModel(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        max_len,
        hidden_dim,
        enc_n_heads,
        enc_n_blocks,
        dec_n_heads,
        dec_n_blocks,
        dropout,
        lr,
        weight_decay,
        warmup_steps,
    ):
        super(EncoderDecoderModel, self).__init__()

        # Saving hyper-parameters so that they are logged
        self.save_hyperparameters()

        # Local parameters
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.enc_n_heads = enc_n_heads
        self.enc_n_blocks = enc_n_blocks
        self.dec_n_heads = dec_n_heads
        self.dec_n_blocks = dec_n_blocks
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.scheduler = None

        # Embeddings (note: we're learning embeddings for both languages in MT)
        self.embedding = get_learnable_embedding(vocab_size, hidden_dim)
        self.enc_pos_embedding = get_learnable_embedding(max_len, hidden_dim)
        self.dec_pos_embedding = get_learnable_embedding(max_len, hidden_dim)

        # Encoder and decoder models
        self.enc_transformer = EncoderTransformer(
            hidden_dim, enc_n_heads, enc_n_blocks, dropout_p=dropout
        )
        self.dec_transformer = DecoderTransformer(
            hidden_dim, enc_n_heads, enc_n_blocks, dropout_p=dropout, with_xa=True
        )

        # Decoding head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, vocab_size)
        )

    def scheduler_fn(self, step):
        step += 1
        return self.hidden_dim ** (-0.5) * min(
            step ** (-0.5), step * self.warmup_steps ** (-1.5)
        )

    def configure_optimizers(self):
        optim = Adam(
            self.trainer.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        self.scheduler = LambdaLR(optim, self.scheduler_fn)
        return optim

    def forward(
        self, ids_enc, ids_dec, enc_attn_mask, dec_attn_mask, enc_dec_attn_mask
    ):
        assert ids_enc.shape[0] == ids_dec.shape[0]

        enc_out = self.forward_enc(ids_enc, enc_attn_mask)
        dec_out = self.forward_dec(ids_dec, dec_attn_mask, enc_dec_attn_mask, enc_out)

        return self.head(dec_out), enc_out, dec_out

    def forward_enc(self, ids_enc, enc_attn_mask):
        b, t = ids_enc.shape
        x_enc = self.embedding(ids_enc) + self.enc_pos_embedding(
            torch.arange(t).cuda()
        ).repeat(b, 1, 1)
        enc_out = self.enc_transformer(x_enc, enc_attn_mask)
        return enc_out

    def forward_dec(self, ids_dec, dec_attn_mask, enc_dec_attn_mask, enc_out):
        b, t = ids_dec.shape
        x_dec = self.embedding(ids_dec) + self.dec_pos_embedding(
            torch.arange(t).cuda()
        ).repeat(b, 1, 1)
        dec_out = self.dec_transformer(
            x_dec,
            kv=enc_out,
            self_attn_mask=dec_attn_mask,
            cross_attn_mask=enc_dec_attn_mask,
        )
        return dec_out

    def compute_loss(self, batch):
        ids_enc = batch["x_enc"]
        ids_dec = batch["x_dec"]
        enc_attn_mask = batch["enc_attn_mask"]
        dec_attn_mask = batch["dec_attn_mask"]
        enc_dec_attn_mask = batch["enc_dec_attn_mask"]

        b, te = ids_enc.shape
        td = ids_dec.shape[1]

        y_pred, _, _ = self(
            ids_enc,
            ids_dec,
            enc_attn_mask.repeat(1, te).reshape(b, te, te),
            dec_attn_mask.repeat(1, td).reshape(b, td, td).tril(),
            enc_dec_attn_mask.repeat(1, td).reshape(b, td, td),
        )

        y_pred = y_pred[dec_attn_mask == 1][:-1]
        y = ids_dec[dec_attn_mask == 1][1:]

        loss = nn.functional.cross_entropy(
            y_pred.reshape(-1, self.vocab_size), y.reshape(-1)
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("train_loss", loss)
        return loss

    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if self.scheduler is not None:
            self.scheduler.step()

        return super().on_train_batch_end(outputs, batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("test_loss", loss)
        return loss

    def generate(self, x_enc, x_dec, max_len=None):
        if max_len is None:
            max_len = self.max_len

        attn_enc = attn_enc_dec = torch.ones(x_enc.shape[1]).cuda()
        enc_out = self.forward_enc(x_enc, attn_enc)

        while x_dec.shape[1] < max_len:
            t = x_dec.shape[-1]
            attn_dec = torch.tril(torch.ones(t, t)).cuda()
            dec_out = self.forward_dec(x_dec, attn_dec, attn_enc_dec, enc_out)
            probs = torch.softmax(self.head(dec_out)[:, -1, :], dim=-1)
            token = torch.multinomial(probs, num_samples=1)
            x_dec = torch.cat((x_dec, token), dim=-1)
        return x_dec
