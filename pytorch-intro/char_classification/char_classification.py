"""
Example on char recognition from https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

This relies on the data from https://download.pytorch.org/tutorial/data.zip being present in ./data
"""

import glob
import os
import random
import string
import time
import unicodedata
from io import open
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_device(device)
print(f"Using device = {torch.get_default_device()}")

# We can use "_" to represent an out-of-vocabulary character, that is, any character we are not handling in our model
allowed_characters = string.ascii_letters + " .,;'" + "_"
n_letters = len(allowed_characters)


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in allowed_characters
    )


# We use a one-hot vector to encode each letter
def letter_to_index(letter: str):
    "Return _ if we find a letter unknown to our model"
    if letter not in allowed_characters:
        letter = "_"
    return allowed_characters.find(letter)


# TODO why use a line_length x 1 x n_letters array rather than line_length x n_letters?
def line_to_tensor(line: str):
    tensor = torch.zeros(len(line), 1, n_letters)
    for idx, letter in enumerate(line):
        tensor[idx][0][letter_to_index(letter)] = 1
    return tensor


class NamesDataset(Dataset):
    def __init__(self, data_dir: str | None = None):
        if data_dir is None:
            data_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "data", "names"
            )

        # Provenance of data
        self.data_dir = data_dir
        self.load_time = time.localtime
        labels_set = set()
        self.data: List[str] = []
        self.data_tensors: List[torch.Tensor] = []
        self.labels: List[str] = []
        self.labels_tensors: List[torch.Tensor] = []

        # Read .txt files in given folder
        text_files = glob.glob(os.path.join(data_dir, "*.txt"))
        for filename in text_files:
            label, _ = os.path.splitext(os.path.basename(filename))
            labels_set.add(label)
            with open(filename, encoding="utf-8") as f:
                lines = f.read().strip().split("\n")
                for name in lines:
                    self.data.append(name)
                    self.data_tensors.append(line_to_tensor(name))
                    self.labels.append(label)
        # Cache tensors for labels
        self.labels_unique = list(labels_set)
        for label in self.labels:
            temp_tensor = torch.tensor(
                [self.labels_unique.index(label)], dtype=torch.long
            )
            self.labels_tensors.append(temp_tensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        data_item = self.data[idx]
        data_label = self.labels[idx]
        data_tensor = self.data_tensors[idx]
        label_tensor = self.labels_tensors[idx]
        return label_tensor, data_tensor, data_label, data_item

    def label_from_output(self, output):
        top_n, top_i = output.topk(1)
        label_i = top_i[0].item()
        return self.labels_unique[label_i], label_i


class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, line_tensor: torch.Tensor):
        rnn_out, hidden = self.rnn(line_tensor)
        output = self.hidden_to_output(hidden[0])
        output = self.softmax(output)
        return output


def train(
    rnn: CharRNN,
    training_data,
    n_epoch=10,
    n_batch_size=64,
    report_every=50,
    learning_rate=0.2,
    criterion=nn.NLLLoss(),
):
    """
    Train on a batch of training_data for a specified number of iterations and report periodically
    """
    # Log losses for plots
    current_loss = 0
    all_losses = []
    rnn.train()
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
    start = time.time()
    print(f"Training on data set with n = {len(training_data)}")
    for iter in range(1, n_epoch + 1):
        # Clear gradients
        rnn.zero_grad()
        # Create minibatches. DataLoaders don't work as each name is a different length???
        batches = list(range(len(training_data)))
        random.shuffle(batches)
        batches = np.array_split(batches, len(batches) // n_batch_size)

        for idx, batch in enumerate(batches):
            batch_loss = 0
            for i in batch:
                (label_tensor, text_tensor, label, text) = training_data[i]
                output = rnn.forward(text_tensor)
                loss = criterion(output, label_tensor)
                batch_loss += loss
            # Optimize parameters
            batch_loss.backward()
            nn.utils.clip_grad_norm_(rnn.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()

            current_loss += batch_loss.item() / len(batch)
        all_losses.append(current_loss / len(batches))
        if iter % report_every == 0:
            print(
                f"{iter} ({iter / n_epoch:.0%}): \t average batch loss = {all_losses[-1]}"
            )
        current_loss = 0
    return all_losses


def main():
    ds = NamesDataset()

    # Get our training a test data
    train_set, test_set = torch.utils.data.random_split(
        ds, [0.85, 0.15], generator=torch.Generator(device=device).manual_seed(2024)
    )
    n_hidden = 128
    rnn = CharRNN(n_letters, n_hidden, len(ds.labels_unique))
    print(rnn)
    train(rnn, train_set)


if __name__ == "__main__":
    main()
