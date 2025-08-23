r"""
PyTorch transforms demo from https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html
"""

import torch
from torchvision import datasets
from torchvision.transforms import Lambda, ToTensor

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),  # Convert PIL image to a tensor
    # The following converts the label index, y into a 1-hot encoded tensor (aka tensor
    # of zeros with a 1 in the spot corresponding the index)
    target_transform=Lambda(
        lambda y: torch.zeros(10, dtype=torch.float).scatter_(
            0, torch.tensor(y), value=1
        )
    ),
)
print(ds)
