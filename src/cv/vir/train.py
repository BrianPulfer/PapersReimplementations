from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomRotation,
    Resize,
    ToTensor,
)

from src.cv.vir.vir import ViR, ViRModes


class ViRLightningModule(pl.LightningModule):
    def __init__(
        self,
        lr=1e-3,
        out_dim=10,
        patch_size=14,
        depth=12,
        heads=12,
        embed_dim=768,
        max_len=257,
        alpha=1.0,
        mode=ViRModes.PARALLEL,
        dropout=0.1,
    ):
        super(ViRLightningModule, self).__init__()
        self.lr = lr
        self.model = ViR(
            out_dim,
            patch_size,
            depth,
            heads,
            embed_dim,
            max_len,
            alpha,
            mode,
            dropout,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log_dict(
            {
                "train_loss": loss,
                "train_acc": acc,
            }
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log_dict(
            {
                "validation_loss": loss,
                "validation_acc": acc,
            }
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log_dict(
            {
                "test_loss": loss,
                "test_acc": acc,
            }
        )
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.trainer.model.parameters(), self.lr)
        return optim


def main(args):
    # Seed everything
    pl.seed_everything(args["seed"])

    # Data
    resize = Resize((args["image_size"], args["image_size"]))
    normalize = Normalize([0.5] * 3, [0.5] * 3)
    train_transform = Compose(
        [resize, ToTensor(), normalize, RandomHorizontalFlip(), RandomRotation(5)]
    )
    val_transform = Compose([resize, ToTensor(), normalize])

    def make_transform(fn):
        def transform(samples):
            samples["image"] = [fn(img.convert("RGB")) for img in samples["image"]]
            samples["label"] = torch.tensor(samples["label"])
            return samples

        return transform

    train_set = load_dataset("frgfm/imagenette", "320px", split="train")
    val_set = load_dataset("frgfm/imagenette", "320px", split="validation")

    train_set.set_transform(make_transform(train_transform))
    val_set.set_transform(make_transform(val_transform))

    train_loader = DataLoader(
        train_set,
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=args["num_workers"],
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args["num_workers"],
    )

    # Load model
    model = ViRLightningModule(
        args["lr"],
        out_dim=10,
        patch_size=args["patch_size"],
        depth=args["depth"],
        heads=args["heads"],
        embed_dim=args["embed_dim"],
        max_len=(args["image_size"] // args["patch_size"]) ** 2 + 1,
        alpha=args["alpha"],
        mode=ViRModes.PARALLEL,
        dropout=args["dropout"],
    )

    # Train model
    logger = pl.loggers.WandbLogger(project="Papers Re-implementations", name="ViR")
    logger.experiment.config.update(args)
    trainer = pl.Trainer(
        strategy="ddp",
        accelerator="auto",
        max_epochs=args["epochs"],
        logger=logger,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=args["checkpoint_dir"],
                filename="vir-model",
                save_top_k=3,
                monitor="train_loss",
                mode="min",
            )
        ],
    )
    trainer.fit(model, train_loader)

    # Evaluate model (setting to recurrent mode)
    model.model.set_compute_mode(ViRModes.RECURRENT)
    trainer.test(model, val_loader)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Training arguments
    parser.add_argument("--seed", help="Random seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint_dir", help="Checkpoint directory", type=str, default="checkpoints"
    )
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=40)
    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=64)
    parser.add_argument("--image_size", help="Image size", type=int, default=224)
    parser.add_argument("--num_workers", help="Number of workers", type=int, default=4)

    # Model arguments
    parser.add_argument("--patch_size", help="Patch size", type=int, default=14)
    parser.add_argument("--depth", help="Depth", type=int, default=12)
    parser.add_argument("--heads", help="Heads", type=int, default=3)
    parser.add_argument(
        "--embed_dim", help="Embedding dimension", type=int, default=192
    )
    parser.add_argument("--alpha", help="Alpha", type=float, default=0.99)
    parser.add_argument("--dropout", help="Dropout", type=float, default=0.1)

    args = vars(parser.parse_args())

    print("\n\nProgram arguments:\n\n", args, "\n\n")
    main(args)
