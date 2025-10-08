"""Linear regression code from Dive Into Deep Learning see linear_regression.ipynb for interactive code"""

import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.utils.data
from d2l import torch as d2l
from torch import nn


class Module(nn.Module, d2l.HyperParameters):
    """Base class of our models"""

    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = d2l.ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, "net"), "Neural network is not defined"
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, "trainer"), "Trainer is not inited"
        self.board.xlabel = "epoch"
        if train:
            x = self.trainer.train_batch_idx / self.trainer.num_train_batches
            n = self.trainer.num_train_batches / self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / self.plot_valid_per_epoch
        self.board.draw(
            x,
            value.to(d2l.cpu()).detach().numpy(),
            ("train_" if train else "val_") + key,
            every_n=int(n),
        )

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot("loss", l, train=True)

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot("loss", l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError


class DataModule(d2l.HyperParameters):
    """Base class of data"""

    def __init__(self, root="../data", num_workers=4):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)


class Trainer:
    """Base class for training models"""

    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.max_epochs = max_epochs
        self.gradient_clip_val = gradient_clip_val
        assert num_gpus == 0, "No gpu support yet"

    def prepare_data(self, data: DataModule):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (
            len(self.val_dataloader) if self.val_dataloader is not None else 0
        )

    def prepare_model(self, model: Module):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]

    def fit(self, model: Module, data: DataModule):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        raise NotImplementedError


@dataclass
class SyntheticRegressionData(DataModule):
    """Synthetic data for linear regression"""

    w: torch.Tensor
    b: float
    noise = 0.01
    num_train = 1000
    num_val = 1000
    batch_size = 32

    def __post_init__(self):
        n = self.num_train + self.num_val
        self.X = torch.randn(n, len(self.w))
        noise = torch.randn(n, 1) * self.noise
        self.y = torch.matmul(self.X, self.w.reshape((-1, 1))) + self.b + noise

    def get_tensorloader(self, tensors, train: bool, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=train)

    def get_dataloader(self, train: bool):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)
