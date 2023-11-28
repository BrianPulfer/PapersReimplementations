import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.utils import save_image

from src.cv.ign.ign import IdempotentNetwork
from src.cv.ign.model import DCGANLikeModel


def main(args):
    # Set seed
    pl.seed_everything(args["seed"])

    # Load datas
    normalize = Lambda(lambda x: (x - 0.5) * 2)
    noise = Lambda(lambda x: (x + torch.randn_like(x) * 0.15).clamp(-1, 1))
    train_transform = Compose([ToTensor(), normalize, noise])
    val_transform = Compose([ToTensor(), normalize])

    train_set = MNIST(
        root="data/mnist", train=True, download=True, transform=train_transform
    )
    val_set = MNIST(
        root="data/mnist", train=False, download=True, transform=val_transform
    )

    def collate_fn(samples):
        return torch.stack([sample[0] for sample in samples])

    train_loader = DataLoader(
        train_set,
        batch_size=args["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args["num_workers"],
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args["num_workers"],
    )

    # Initialize model
    prior = torch.distributions.Normal(torch.zeros(1, 28, 28), torch.ones(1, 28, 28))
    net = DCGANLikeModel()
    model = IdempotentNetwork(prior, net, args["lr"])

    if not args["skip_train"]:
        # Train model
        logger = WandbLogger(name="IGN", project="Papers Re-implementations")
        callbacks = [
            ModelCheckpoint(
                monitor="val/loss",
                mode="min",
                dirpath="checkpoints/ign",
                filename="best",
            )
        ]
        trainer = pl.Trainer(
            strategy="ddp",
            accelerator="auto",
            max_epochs=args["epochs"],
            logger=logger,
            callbacks=callbacks,
        )
        trainer.fit(model, train_loader, val_loader)

    # Loading the best model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = (
        IdempotentNetwork.load_from_checkpoint(
            "checkpoints/ign/best.ckpt", prior=prior, model=net
        )
        .eval()
        .to(device)
    )

    # Generating images with the trained model
    os.makedirs("generated", exist_ok=True)

    images = model.generate_n(100, device=device)
    save_image(images, "generated.png", nrow=10, normalize=True)

    print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--skip_train", action="store_true")
    args = vars(parser.parse_args())

    print("\n\n", args, "\n\n")
    main(args)
