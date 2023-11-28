from copy import deepcopy

import pytorch_lightning as pl
from torch.nn import L1Loss
from torch.optim import Adam


class IdempotentNetwork(pl.LightningModule):
    def __init__(
        self,
        prior,
        model,
        lr=1e-4,
        criterion=L1Loss(),
        lrec_w=20.0,
        lidem_w=20.0,
        ltight_w=2.5,
    ):
        super(IdempotentNetwork, self).__init__()
        self.prior = prior
        self.model = model
        self.model_copy = deepcopy(model)
        self.lr = lr
        self.criterion = criterion
        self.lrec_w = lrec_w
        self.lidem_w = lidem_w
        self.ltight_w = ltight_w

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optim = Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return optim

    def get_losses(self, x):
        # Prior samples
        z = self.prior.sample_n(x.shape[0]).to(x.device)

        # Updating the copy
        self.model_copy.load_state_dict(self.model.state_dict())

        # Forward passes
        fx = self(x)
        fz = self(z)
        fzd = fz.detach()

        l_rec = self.lrec_w * self.criterion(fx, x)
        l_idem = self.lidem_w * self.criterion(self.model_copy(fz), fz)
        l_tight = -self.ltight_w * self.criterion(self(fzd), fzd)

        return l_rec, l_idem, l_tight

    def training_step(self, batch, batch_idx):
        l_rec, l_idem, l_tight = self.get_losses(batch)
        loss = l_rec + l_idem + l_tight

        self.log_dict(
            {
                "train/loss_rec": l_rec,
                "train/loss_idem": l_idem,
                "train/loss_tight": l_tight,
                "train/loss": l_rec + l_idem + l_tight,
            },
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        l_rec, l_idem, l_tight = self.get_losses(batch)
        loss = l_rec + l_idem + l_tight

        self.log_dict(
            {
                "val/loss_rec": l_rec,
                "val/loss_idem": l_idem,
                "val/loss_tight": l_tight,
                "val/loss": loss,
            },
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx):
        l_rec, l_idem, l_tight = self.get_losses(batch)
        loss = l_rec + l_idem + l_tight

        self.log_dict(
            {
                "test/loss_rec": l_rec,
                "test/loss_idem": l_idem,
                "test/loss_tight": l_tight,
                "test/loss": loss,
            },
            sync_dist=True,
        )

    def generate_n(self, n, device=None):
        z = self.prior.sample_n(n)

        if device is not None:
            z = z.to(device)

        return self(z)
