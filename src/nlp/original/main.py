"""
Re-implementation of

                            Attention is all you need
                              Vaswani et al. (2017)
                        (https://arxiv.org/abs/1706.03762)

on the WMT14 dataset.
"""
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer

from src.nlp.original.data import WMT14Dataset
from src.nlp.original.model import EncoderDecoderModel


def translate_sentences(file_path, model, tokenizer, max_len=128):
    file = open(file_path, "r")
    sentences = file.readlines()
    file.close()

    model = model.cuda()
    dec_input = tokenizer.bos_token_id * torch.ones(1, 1)
    dec_input = dec_input.long().cuda()
    for i, sentence in enumerate(sentences):
        x = tokenizer(
            sentence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )["input_ids"].cuda()
        y = model.generate(x, dec_input, max_len)[0]
        translated = tokenizer.decode(y)

        print(f"\n\nSENTENCE {i+1}:")
        print(f"\tOriginal: {sentence}\n\tTranslated: {translated}")


def main(args):
    # Unpacking arguments
    enc_n_blocks = args["enc_n_blocks"]
    enc_n_heads = args["enc_n_heads"]
    dec_n_blocks = args["dec_n_blocks"]
    dec_n_heads = args["dec_n_heads"]
    hidden_dim = args["hidden_dim"]
    dropout = args["dropout"]
    max_len = args["max_len"]
    batch_size = args["batch_size"]
    max_train_steps = args["max_train_steps"]
    lr = args["lr"]
    weight_decay = args["weight_decay"]
    warmup_steps = args["warmup_steps"]
    save_dir = args["save"]
    file_path = args["file"]
    seed = args["seed"]

    # Setting seed
    pl.seed_everything(seed)

    # Loading dataset
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        vocab_size += 1

    dataset = WMT14Dataset(tokenizer, batch_size, max_len)

    # Loading model
    model = EncoderDecoderModel(
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
    )

    # Training
    callbacks = [ModelCheckpoint(save_dir, monitor="val_loss", filename="best")]
    logger = WandbLogger(project="Papers Re-implementations", name="ORIGINAL")
    logger.experiment.config.update(args)
    trainer = pl.Trainer(
        devices="auto",
        strategy="ddp",
        max_steps=max_train_steps,
        callbacks=callbacks,
        logger=logger,
    )
    trainer.fit(model, datamodule=dataset)
    trainer.save_checkpoint(os.path.join(save_dir, "best.ckpt"))  # TODO DELETE

    # Testing
    model = EncoderDecoderModel.load_from_checkpoint(
        os.path.join(save_dir, "best.ckpt")
    )
    # trainer.test(model, datamodule=dataset) # TODO ENABLE

    # Translating custom sentences
    translate_sentences(file_path, model, tokenizer, max_len)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Model hyper-parameters
    parser.add_argument("--enc_n_blocks", type=int, default=12)
    parser.add_argument("--enc_n_heads", type=int, default=12)
    parser.add_argument("--dec_n_blocks", type=int, default=12)
    parser.add_argument("--dec_n_heads", type=int, default=12)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=128)

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_train_steps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--save", type=str, default="checkpoints/original")

    # Testing parameters
    parser.add_argument(
        "--file", type=str, default="data/nlp/original/translate_sentences.txt"
    )

    # Seed
    parser.add_argument("--seed", type=int, default=0)

    args = vars(parser.parse_args())
    print(args)
    main(args)
