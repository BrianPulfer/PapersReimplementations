from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

from src.fff.fff import FFFLayer


class FlattenMNIST:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten()


class PLWrapper(pl.LightningModule):
    def __init__(self, model, total_iters=10, hardening_weight=3.0, lr=0.2):
        super(PLWrapper, self).__init__()
        self.model = model
        self.cross_entropy = nn.CrossEntropyLoss()
        self.accuracy = Accuracy("multiclass", num_classes=10)
        self.total_iters = total_iters
        self.hardening_weight = hardening_weight
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def entropy(self, x):
        entropies = x * torch.log(x + 1e-8) + (1 - x) * torch.log(1 - x + 1e-8)
        return -torch.mean(entropies)

    def configure_optimizers(self):
        optimizer = SGD(self.model.parameters(), lr=self.lr)
        scheduler = LinearLR(optimizer, 1, 0, total_iters=self.total_iters)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, activations = self.model(x)
        activations = torch.cat(activations, dim=0)
        le = self.hardening_weight * self.entropy(activations)
        lce = self.cross_entropy(y_hat, y)
        loss = lce + le
        acc = self.accuracy(y_hat, y)
        self.log_dict(
            {
                "train_loss": loss,
                "train_loss_ce": lce,
                "train_loss_entropy": le,
                "train_acc": acc,
            }
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model(x)
        loss = self.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_acc", acc)


def main(args):
    """
    Training and evaluating an FFF model on MNIST image classification.
    Over-engineering code with Lightning and W&B to keep the good habits.
    """
    # Program arguments
    batch_size = args["batch_size"]
    max_epochs = args["max_epochs"]
    lr = args["lr"]
    hardening_weight = args["hardening_weight"]
    seed = args["seed"]

    # Setting reproducibility
    pl.seed_everything(seed)

    # Getting data
    transform = Compose([ToTensor(), FlattenMNIST()])
    train = MNIST("./", train=True, download=True, transform=transform)
    test = MNIST("./", train=False, download=True, transform=transform)
    train_loader = DataLoader(train, batch_size=batch_size, num_workers=4, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, num_workers=4, shuffle=False)

    # Getting model
    fff_model = FFFLayer(
        depth=3, in_dim=28 * 28, node_hidden=32, leaf_hidden=32, out_dim=10
    )

    model = PLWrapper(
        fff_model,
        total_iters=max_epochs * len(train_loader),
        hardening_weight=hardening_weight,
        lr=lr,
    ).train()

    # Training
    trainer = pl.Trainer(
        logger=WandbLogger(
            name="FFF MNIST",
            project="Papers Re-implementations",
        ),
        accelerator="auto",
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(dirpath="./checkpoints", monitor="train_loss", mode="min"),
            TQDMProgressBar(),
        ],
    )
    trainer.fit(model, train_loader)

    # Loading best model
    model = PLWrapper.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path, model=fff_model
    )

    # Testing and logging
    model.eval()
    trainer.test(model, test_loader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help="Number of epochs to train the model.",
    )
    parser.add_argument(
        "--lr", type=float, default=0.2, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--hardening_weight",
        type=float,
        default=3.0,
        help="Hardening weight for the entropy loss.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for the random number generators."
    )
    args = vars(parser.parse_args())
    print(args)
    main(args)
