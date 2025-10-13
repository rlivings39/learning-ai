"""Linear regression code from Dive Into Deep Learning see linear_regression.ipynb for interactive code"""

import random
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.utils.data
import torchvision
from d2l import torch as d2l
from torch import nn
from torchvision import transforms


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

    plot_train_per_epoch: int = 2
    plot_valid_per_epoch: int = 1

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

    def configure_optimizers(self):
        raise NotImplementedError


@dataclass
class DataModule:
    """Base class of data"""

    root: str = "./data"
    num_workers: int = 0

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)


@dataclass
class FashionMNIST(DataModule):
    """Fashion-MNIST dataset"""

    batch_size: int = 64
    resize: Tuple[int, ...] = (28, 28)

    def __post_init__(self):
        xform = transforms.Compose(
            [transforms.Resize(self.resize), transforms.ToTensor()]
        )
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=xform, download=True
        )
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=xform, download=True
        )

    def text_labels(self, indices):
        labels = [
            "t-shirt",
            "trouser",
            "pullover",
            "dress",
            "coat",
            "sandal",
            "shirt",
            "sneaker",
            "bag",
            "ankle boot",
        ]
        return [labels[int(i)] for i in indices]

    def get_dataloader(self, train):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(
            data, self.batch_size, shuffle=train, num_workers=self.num_workers
        )

    def visualize(self, batch, nrows=1, ncols=8, labels=[]):
        X, y = batch
        if not labels:
            labels = self.text_labels(y)
        d2l.show_images(X.squeeze(1), nrows, ncols, titles=labels)


@dataclass
class Trainer:
    """Base class for training models"""

    max_epochs: int
    num_gpus: int = 0
    gradient_clip_val: int = 0

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

    w: torch.Tensor = torch.tensor(0)
    b: float = 0.0
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

    num_inputs: int = 0
    learning_rate: float = 0.1
    sigma: float = 0.01

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


@dataclass
class LinearRegression(Module):
    """Linear regression model with PyTorch"""

    learning_rate: float = 0.11

    def __post_init__(self):
        super().__post_init__()
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)
        self._loss_fn = nn.MSELoss()

    def __hash__(self):
        """Hacky hash since dataclass isn't hashable by default but torch relies on models being hashable"""
        return self.net.__hash__()

    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y):
        return self._loss_fn(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)

    def get_w_b(self):
        return (self.net.weight.data, self.net.bias.data)


@dataclass
class Classifier(Module):
    """Base class of classification models"""

    learning_rate: float = 0.1

    def __post_init__(self):
        super().__post_init__()

    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot("loss", self.loss(Y_hat, batch[-1]), train=False)
        self.plot("accuracy", self.accuracy(Y_hat, batch[-1]), train=False)

    def accuracy(self, Y_hat, Y, averaged=True):
        """Compute number of correct predictions"""
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        preds = Y_hat.argmax(axis=1).type(Y.dtype)
        compare = (preds == Y.reshape(-1)).type(torch.float32)
        return compare.mean() if averaged else compare

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)


def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


def cross_entropy(y_hat, y):
    l = -torch.log(y_hat[list(range(len(y_hat))), y]).mean()
    return l


@dataclass
class MultiLayerPerceptronScratch(Classifier):
    """A multi-layer perceptron (MLP) built from scratch with 1 hidden layer"""

    num_inputs: int = 0
    num_outputs: int = 0
    num_hiddens: int = 0
    learning_rate: float = 0.1
    sigma: float = 0.01

    def __post_init__(self):
        super().__post_init__()
        self.W1 = nn.Parameter(
            torch.randn(self.num_inputs, self.num_hiddens) * self.sigma
        )
        self.b1 = nn.Parameter(torch.zeros(self.num_hiddens))
        self.W2 = nn.Parameter(
            torch.randn(self.num_hiddens, self.num_outputs) * self.sigma
        )
        self.b2 = nn.Parameter(torch.zeros(self.num_outputs))

    def __hash__(self):
        return hash((self.W1, self.b1, self.W2, self.b2))

    def forward(self, X):
        X = X.reshape((-1, self.num_inputs))
        H = relu(torch.matmul(X, self.W1) + self.b1)
        Y = torch.matmul(H, self.W2) + self.b2
        return Y

    def loss(self, y_hat, y):
        return cross_entropy(y_hat, y)
