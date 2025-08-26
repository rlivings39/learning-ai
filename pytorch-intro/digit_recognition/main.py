import numpy as np
import torch
from torch import tensor
from torch.utils.data import DataLoader


def load_data():
    data_len = 1000
    X: np.ndarray = np.load("data/X.npy")[:data_len]
    y = np.load("data/y.npy")[:data_len]
    print(f"sum(y) = {y.sum()}")
    return torch.from_numpy(X), torch.from_numpy(y)


class DigitNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.stack = torch.nn.Sequential(
            torch.nn.Linear(400, 25, dtype=torch.double),
            torch.nn.Sigmoid(),
            torch.nn.Linear(25, 15, dtype=torch.double),
            torch.nn.Sigmoid(),
            torch.nn.Linear(15, 1, dtype=torch.double),
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.stack(x)
        return logits


def make_model():
    model = DigitNetwork()
    print(model)
    return model


def train(X: torch.Tensor, y: torch.Tensor, model: DigitNetwork, loss_fn, optim):
    size = y.size(0)
    model.train()
    for idx in range(0, size, 100):
        val = X[idx : idx + 100, :]
        pred = model(val)
        loss = loss_fn(pred, y[idx : idx + 100].to(torch.double))
        loss.backward()
        optim.step()
        optim.zero_grad()
        if idx % 100 == 0:
            print(f"loss: {loss:>7f}")


def test(X, y, model: DigitNetwork, loss_fn, optim):
    size = y.size(0)
    model.eval()
    test_loss, correct = 0, 0
    wrong_idx = -1
    with torch.no_grad():
        for idx in range(0, size, 100):
            val = X[idx : idx + 100, :]
            pred = model(val)
            pred = pred > 0.5
            exp = y[idx : idx + 100]
            print(f"sum(exp) = {exp.sum()}")
            matches = pred == exp
            num_correct = matches.type(torch.double).sum()
            if wrong_idx == -1 and num_correct < 100:
                mismatch = ~matches
                wrong_idx = np.flatnonzero(mismatch)[0]
            correct += num_correct
    print(f"Number correct: {correct}/{size}. First wrong index: {wrong_idx}\n")


def main():
    X, y = load_data()
    print(f"Shape(x) = {X.shape}")
    print(f"Shape(y) = {y.shape}")

    model = make_model()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}--------------------------------------------")
        train(X, y, model, loss_fn, optim)
        test(X, y, model, loss_fn, optim)


if __name__ == "__main__":
    main()
