"""
Example on char recognition from https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

This relies on the data from https://download.pytorch.org/tutorial/data.zip being present in ./data
"""

import glob
import os
import string
import time
import unicodedata
from io import open

import torch
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
    def __init__(self, data_dir: str):
        # Provenance of data
        self.data_dir = data_dir
        self.load_time = time.localtime
        labels_set = set()
        self.data = []
        self.data_tensors = []
        self.labels = []
        self.labels_tensors = []

        # Read .txt files in given folder
        text_files = glob.glob(os.path.join(data_dir, "*.txt"))
        for filename in text_files:
            label, _ = os.path.splitext(os.path.basename(filename))

        pass
