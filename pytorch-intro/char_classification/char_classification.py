"""
Example on char recognition from https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

This relies on the data from https://download.pytorch.org/tutorial/data.zip
being present in data
"""

import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_device(device)
print(f"Using device = {torch.get_default_device()}")
