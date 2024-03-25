r"""
Info on PyTorch Datasets and DataLoaders from https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
"""

# Dataset - Data structure storing data samples and corresponding labels
# DataLoader - Wraps an iterable around a `Dataset` for easy access

# Loading a data set
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",         # Path where train/test data is stored
    train=True,          # True for training data, False for test data
    download=True,       # Downloads data if not available in root
    transform=ToTensor() # Feature and label transformations
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = training_data.classes
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
