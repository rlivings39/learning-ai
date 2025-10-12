# Dive into Deep Learning Notes

Working through the book https://d2l.ai/

## Setup

With Python 3.9 from the root of this repo. The d2l package [uses old dependencies](https://d2l.ai/chapter_installation/index.html) so we use Python 3.9

```bash
git submodule update --init --recursive
cd ai-courses-and-books/dive-into-deep-learning/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Some preparatory info

PyTorch provides the `Tensor` class which mimics the `NumPy` array but with more functionality. Use `Pandas` to import and manipulate tabular data.

Visualization libraries include `seaborn, Bokeh, matplotlib`.

https://d2l.ai/chapter_preliminaries/calculus.html shows some visualization examples and introduces the `%@save` comment.

The book presents some useful identities for gradients

* $\nabla_x Ax = A^T$ and $\nabla_x x^TA = A$
* For square matrices $\nabla_x x^TAx = (A+A^T)x$ so that $\nabla_x\|x\|^2 = \nabla_x x^Tx = 2x$
* For any matrix $\nabla_x\|X\|_F^2 = 2X$

The multi-variate chain rule applies frequently in deep learning. For $y = f(u)$ with $u\in\R^m = (g_1(x), g_2(x), ..., g_m(x))$ and $x\in\R^n$ or $u = g(x)$ we have

$$
\dfrac{\partial y}{\partial x_i} = \sum_{k=1}^m\dfrac{\partial y}{\partial u_k}\dfrac{\partial u_k}{\partial x_i}
$$

so that

$$
\nabla_x y = A\nabla_u y \text{ for } A \in \R ^{n \times m}
$$

where $A$ is the derivative of the vector $u$ wrt the vector $x$.

Automatic differentiation (**autograd**) is used by all DL frameworks to avoid the need to compute gradients by hand.

It generally works by producing a computational graph as you compute your objective function and then using the chain rule to compute derivatives.

In PyTorch you set `requires_grad=True` for the tensors for which you'd like to differentiate with respect to. Then you compute the target value, call `backward()` to do autograd, and access `x.grad`.

## Linear neural networks for regression

See [machine-learning/README.md](../machine-learning/README.md#linear-regression) for the Coursera treatment on this.

A neural network with a single fully connected linear layer can be used to perform traditional linear regression [linear_regression.py](./linear_regression.py) and [linear_regression.ipynb](./linear_regression.ipynb) demonstrate this.

## Classification in linear networks

See [machine-learning/README.md](ai-courses-and-books/machine-learning/README.md#logistic-regression) for the basics.



## Next

https://d2l.ai/chapter_multilayer-perceptrons/index.html

## TODO

- [ ] Work through multivariate chain rule and gradient identities with matrix formulations. [More info](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/multivariable-calculus.html)
- [ ] Review Jacobians and [non-scalar backprop](https://d2l.ai/chapter_preliminaries/autograd.html)
- [ ] Learn how autograd actually works in PyTorch
- [ ] Probability section

