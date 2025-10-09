"""Linear regression code from Dive Into Deep Learning see linear_regression.ipynb for interactive code"""

import random
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.utils.data
from d2l import torch as d2l
from torch import nn


@dataclass
class SGD:
    """Minibatch stochastic gradient descent"""

    params: List[torch.Tensor]
    learning_rate: float

    def step(self):
        for param in self.params:
            param -= self.learning_rate * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


@dataclass
class Module(nn.Module):
    """Base class of our models"""

    plot_train_per_epoch = 2
    plot_valid_per_epoch = 1

    def __post_init__(self):
        super().__init__()
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

    def training_step(self, batch) -> torch.Tensor:
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot("loss", l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot("loss", l, train=False)

    def configure_optimizers(self) -> SGD:
        raise NotImplementedError


@dataclass
class DataModule:
    """Base class of data"""

    root = "../data"
    num_workers = 4

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)


@dataclass
class Trainer:
    """Base class for training models"""

    max_epochs: int
    num_gpus = 0
    gradient_clip_val = 0

    def __post_init__(self):
        assert self.num_gpus == 0, "No gpu support yet"

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
        self.model = model

    def fit(self, model: Module, data: DataModule):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def prepare_batch(self, batch):
        return batch

    def clip_gradients(self, grad_clip_val, model):
        params = [p for p in model.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm

    def fit_epoch(self):
        self.model.train()
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()
            self.train_batch_idx += 1
        if self.val_dataloader is None:
            return
        self.model.eval()
        for batch in self.val_dataloader:
            with torch.no_grad():
                self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx += 1


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


@dataclass
class LinearRegressionScratch(Module):
    """A linear regression model implemented from scratch"""

    num_inputs: int
    learning_rate: float
    sigma = 0.01

    def __post_init__(self):
        super().__post_init__()
        # Initialize weights to normally distributed values
        self.w = torch.normal(0, self.sigma, (self.num_inputs, 1), requires_grad=True)
        # Initialize bias to 0
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward prop where X is a tensor whose rows are examples. For a single sample pass a row vector."""
        return torch.matmul(X, self.w) + self.b

    def loss(self, y_hat, y):
        l = ((y_hat - y) ** 2) / 2
        return l.mean()

    def configure_optimizers(self):
        return SGD([self.w, self.b], self.learning_rate)
