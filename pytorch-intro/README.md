# PyTorch Intro

This is some [intro material](https://pytorch.org/tutorials/beginner/basics/intro.html) for using PyTorch from pytorch.org

## Notes

PyTorch has a number of pre-existing data sets in torchvision.datasets. PyTorch uses the concept of a DataLoader to load data in batches, normalize it, etc.

There are 3 classes of devices supported: `"cuda", "mps", "cpu"`. `"cuda"` runs on a CUDA GPU. `"mps"` uses the Metal programming framework on MacOS. `"cpu"` uses your CPU.

There is a general concept of moving something to the active device. That looks like `NeuralNetrowk().to(device)` or `data.to(device)`. The former creates and moves a NN to the device while the latter moves data to the device.

Tensors are the PyTorch analog of ND arrays. They are very similar to NumPy arrays with many similar operations. There's a bridge between NumPy and PyTorch.

Tensors are allocated on the CPU by default and must be moved using the `to` method to use them on a GPU or other device. Moving data can be expensive so keep that in mind.

By default, CPU tensors and numpy arrays share memory when one is created from the other.

`Dataset` is a data structure storing data samples and corresponding labels. A `DataLoader` wraps an iterable around a `Dataset` for easy access.

The `Dataset` can specify options like the root location, if training or test data is wanted, whether to download the data if needed, and how to transform the data.

A `Dataset` can be indexed like a list `dataset[index]`

To create a custom `Dataset` you create a class inheriting from `Dataset` that implements `__init__, __len__, __getitem__`

`Dataset` allows you to deal with a single sample at a time. Often we want to work on samples in minibatches, shuffle data at every epoch to avoid overfitting, and use `multiprocessing` to accelerate data retrieval. `DataLoader` abstracts all of this.

Iterating through a `DataLoader` returns Tensors of the features `[batch_size, feature_size...]`.

Data often requires transformation to be ready for learning. Both the features and labels may require such transformations.

Neural networks are composed of multiple layers/modules that operate on data. `torch.nn` has all of the building blocks. All modules subclasses `nn.Module`. A neural network is a module that is composed of other modules (i.e. layers).

## Usage

* Activate the `venv`: `source .env/bin/activate` or `source .env/bin/activate.fish`
* Install any needed packages `pip install -r requirements.txt`
* Enjoy
