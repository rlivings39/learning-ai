# PyTorch Intro

This is some [intro material](https://pytorch.org/tutorials/beginner/basics/intro.html) for using PyTorch from pytorch.org

## Notes

PyTorch has a number of pre-existing data sets in torchvision.datasets. PyTorch uses the concept of a DataLoader to load data in batches, normalize it, etc.

There are 3 classes of devices supported: `"cuda", "mps", "cpu"`. `"cuda"` runs on a CUDA GPU. `"mps"` uses the Metal programming framework on MacOS. `"cpu"` uses your CPU.

There is a general concept of moving something to the active device. That looks like `NeuralNetrowk().to(device)` or `data.to(device)`. The former creates and moves a NN to the device while the latter moves data to the device.

Tensors are the PyTorch analog of ND arrays. They are very similar to NumPy arrays with many similar operations. There's a bridge between NumPy and PyTorch.

Tensors are allocated on the CPU by default and must be moved using the `to` method to use them on a GPU or other device. Moving data can be expensive so keep that in mind.

By default, CPU tensors and numpy arrays share memory when one is created from the other.

## Usage

* Activate the `venv` `source .env/bin/activate` or `source .env/bin/activate.fish`
* Install any needed packages `pip install -r requirements.txt`
* Enjoy
