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

Add layers to the model simply by adding properties to `self`. They are treated like a dict and insertion order is preserved.

To use a model, call it with the input data.

Back propagation is the most frequently used algorithm. Parameters are adjusted according to the gradient of the loss function. `torch.autograd` is the automatic differentiation engine used for this.

## Automatic differentiation (aka autograd)

[Autograd reference page](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)

You want to be able to compute the gradients of the loss functions with respect to the parameters. So mark those with `requires_grad=True` or `x.requires_grad_(True)` like

```python
import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

```

Doing so causes computations with those tensors to build up a computational DAG as you operate on them. That DAG is then used to be able to compute gradients using the chain rule.

The above code produces `grad_fn` attributes on `z` and `loss`.

To compute gradients call `loss.backward()` and access gradients via `w.grad, b.grad`

The computational DAG is dynamically constructed from scratch after every call to `backward()` which allows things like control flow to work.

Gradient tracking can be disabled with `torch.no_grad()`:

```python
with torch.no_grad():
    z = torch.matmul(x, w)+b
```

for cases when you don't need gradients, e.g. forward prop, freezing parameters.

Jacobian products $v^T \cdot J$ can be computed by passing $v$ as an argument to backward `out.backward(input_vec, retain_graph=True)`.

## Optimizing model parameters

The goal is to optimize the model parameters.

There are some hyperparameters which can be tuned

* Number of epochs - How many passes are done over the data
* Batch size - the number of data samples propagated through the network before parameters are updated
* Learning rate - tunes the step size of the gradient descent algorithm. Small values result in slower convergence with higher precision. Larger values may result in faster convergence at the cost of possible unpredictability.

A loss function is also needed which serves as the objective function to optimize. Common loss functions include `nn.MSELoss` (Mean Square Error) for regression tasks, and `nn.NLLLoss` (Negative Log Likelihood) for classification. `nn.CrossEntropyLoss` combines `nn.LogSoftmax` and `nn.NLLLoss`.

An **optimizer** handles actually doing the optimization. Stochastic gradient descent (SGD), ADAM, and RMSProp are common optimizers but there are many available which work better for different models and data.

Setting up an optimizer involves providing model parameters and the learning rate

```python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

Inside the training loop we:

* Call `optimizer.zero_grad()` to reset gradients. Gradients accumulate.
* Backpropagate the prediction loss with `loss.backpropagate()`. Gradients are stored in the parameters.
* Call `optimizer.step()` to adjust the parameters with the gradients collected in the backward pass.

## Saving and loading models

Model parameters can be saved with `torch.save(model.state_dict(), 'model_weights.pth')` and loaded with `model.load_state_dict(torch.load('model_weights.pth', weights_only=True))`

## TensorRT

[TensorRT](https://developer.nvidia.com/tensorrt) is a set of runtime libraries, tunable optimized kernels, optimizers, and compilers used for optimizing the inference performance of deep learning inference.

TensorRT works on all major frameworks and supports targeting hyperscale data centers, workstations, laptops, and edge devices.

Optimizations include quantization, layer and tensor fusion, and kernel tuning techniques. It supports both post-training quantization and quantization-aware training. Small sizes like FP8, FP4, INT8, INT4 and advanced things like AWQ are available.

The flow typically looks like

```
Trained DNN -> ONNX conversion -> TensorRT optimizer -> TensorRT runtime
```

Some frameworks like PyTorch have framework online integrations

```
PyTorch trained DNN -> TensorRT optimizer (integration API) -> TensorRT runtime (integration API) -> PyTorch TorchScript in-framework inference
```

Features include

* TensorRT-LLM - An open source library to accelerate/optimize inference of LLMs
* TensorRT Cloud provides a cloud-based optimization service that optimizes based on given constraints and KPIs for LLMs.
* TensorRT model optimizer does quantization, pruning, speculation, sparsity, and distillation for downstream inference.
* TensorRT integrates with PyTorch and Hugging Face directly for claimed 6x inference speedups w/ a single line of code. An ONNX parser imports ONNX models from other frameworks. GPU Coder integrates MATLAB to generate engines for NVIDIA Jetson, DRIVE, and data center.
* Dynamo Triton is inference serving software
* Simplifies deployment and inference on RTX gpus

### Quantization

[A White Paper on Neural Network Optimization](https://arxiv.org/pdf/2106.08295) discusses 2 types of quantization **post training quantization (PTQ)** and **quantization aware training (QAT)**. The former is a push-button solution that can be applied to pre-trained networks. The latter requires labeled data and fine tuning but can enable lower bit quantization with comparable results. QAT includes quantization noise effects in the training process.

The authors say that PTQ can result in floating-point-like accuracy with 8-bit quantization. When quantizing from 32 bits to 8 bits NN storage size is reduced by a factor of 4 and matrix multiplication is reduced by a factor of 16 for a matrix-vector product (i.e. quadratic).

The paper describes decomposing a vector `x` into a float scalar and integer vector `s*x_int` for a single scalar for the whole vector. Doing this with weights and input vectors allows for most math to be done with integers/fixed point. Accumulators are kept as floating point and then flushed to smaller sizes before transmission as activations.

The paper uses **uniform affine quantization** or **asymmetric quantization**. This is defined by 3 parameters `s` - the scale factor, `z` - the zero point, and `b` the bit width.

Quantization maps a float to an integer value in $[0, 2^b-1]$ via
$$
x_{int} = clamp \left (round \left (\frac{x}{s} \right ) + z, 0, 2^b-1 \right)
$$

The zero point ensures that 0 maps to 0 to maintain zero padding and proper functionality for things like ReLU.

De-quantization is the inverse

$$
x \approx \hat{x} = q(s,z,b) = s(x_{int} - z)
$$

where $q()$ is the quantization function.

The range of this quantized value is $(q_{min}, q_{max}) = (-sz, s(2^b - 1 - z))$. The range inducing a **clipping error**. The range can be increased by increasing $s$ at the expense of increasing the **rounding error** which is $[-\frac{1}{2}s, \frac{1}{2}s]$.

**Symmetric uniform quantization** restricts `z` to 0 and requires choosing a signed or unsigned interpretation based on the necessary operations.

There is a question of how many different quantizations should be done. Per-tensor quantization is common where each tensor (i.e. weights and activations). Finer granularity can be used to improve accuracy at the expense of more bookkeeping and computation.

Quantized inference can be done on dedicated hardware or simulated on general-purpose hardware to test the effects.

Quantization is ideally applied after non-linearities to avoid extra work but hardware may not always support this.

Homogeneous bit-width is usually chosen. One can choose symmetric or asymmetric quantization and the appropriate granularity such as per-tensor or per-channel.

### Choosing quantization parameters

The quantization range needs to be chosen. That can be done with a few strategies

* **Min-max** Choose the min and max of the tensor. This can be sensitive to outliers.
* **Mean squared error (MSE)** Choose the min and max to minimize the Frobenius norm between the original value and quantized value. The optimization problem can be solved using some simple methods.
* **Cross-entropy** Minimize the cross-entropy function to avoid over correcting for the many small insignificant values present in classification tasks.
* **Batch-normalized range setting** Using batch normalized range setting can help with

### Techniques to improve performance with quantization

* **Cross layer equalization** is useful to help normalize when there are some layers with wildly differing ranges. **Absorbing biases** can be used to improve high bias and bias correction.

Bias correction can be done with experimental correction or analytical correction.

### Standard PTQ flow

The authors recommend the following flow

* **Cross-layer equalization** First apply cross-layer equalization (CLE), to help with depth-wise separable layers and more
* **Add quantizers** Choose quantizers and add quantization operations in the network depending on hardware. Use symmetric for weights and asymmetric for activations. Per-channel help if the hardware supports them.
* **Weight range setting** MSE based criteria is recommended. Min-max can be useful for per-channel.
* **AdaRound** If a calibration data set exists apply AdaRound to optimize rounding. Crucial for low bit (e.g. 4-bit) quantization
* **Bias correction** Without a calibration set but with batch normalization use analytical bias correction instead.
* **Activation range setting** Determine the quantization ranges of all data dependent tensors in the network (i.e., activations). Use the MSE based criteria for most of the layers, which requires a small calibration set to find the minimum MSE loss. BN range setting doesn't require data.

## Using this repo

* Activate the `venv`: `source .env/bin/activate` or `source .env/bin/activate.fish`
* Install any needed packages `pip install -r requirements.txt`
* Enjoy

## Actions

- [ ] Tensors and the stride vector. When are strides legal? I.e. what does contiguity mean?
  - [ ]https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107/16, https://stackoverflow.com/questions/48915810/what-does-contiguous-do-in-pytorch, https://stackoverflow.com/questions/48915810/what-does-contiguous-do-in-pytorch/67021086#67021086
- [ ] Basic network construction
- [ ] Invocation and tuning of backprop
- [ ] Invocation and tuning of forward prop
- [ ] DL compilers. Where does TensorRT fit in everything?
- [ ] TensorRT getting started
  * https://developer.nvidia.com/tensorrt-getting-started
  * https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html
  * https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorrt-updated/
  * https://developer.nvidia.com/blog/optimizing-and-serving-models-with-nvidia-tensorrt-and-nvidia-triton/
  * https://www.youtube.com/watch?v=SlUouzxBldU
- [ ] Types of layers: ReLU, softmax, linear, logistic, convolution
