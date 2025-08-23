r"""
PyTorch demo showing how to build models/NNs: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
"""

import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Pick our device for training. Prefer high performance if available
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")


# Define our network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# Create an instance of the network and move it to the chosen device
model = NeuralNetwork().to(device)
print(model)

# Apply th emodel
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# Exploring the model layers

# Take a sample of 3 images
input_image = torch.rand(3, 28, 28)
print(input_image.size())

# nn.Flatten - Flatten ND arrays into a 1D array in row-major order
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# nn.Linear - Applies a linear transformation on the input
# using stored weights and biases
layer1 = nn.Linear(in_features=28 * 28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# nn.ReLU - Using nonlinear activations allows learning a
# variety of things. It's used between linear layers.
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# nn.Sequential - Ordered sequence of layers
seq_modules = nn.Sequential(flatten, layer1, nn.ReLU(), nn.Linear(20, 10))
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)

# nn.Softmax - Maps [-infinity, infinity] to [0,1]
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

# Model parameters - Many layers have parameters (weights and biases)
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
