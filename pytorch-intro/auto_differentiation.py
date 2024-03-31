r"""
Automatic differentiation example from https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
"""

# Back propagation is the most frequently used algorithm to train.
# Parameters are adjusted according to the gradient of the loss function.
# torch.autograd is the automatic differentiation engine used for this.

import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True) # Say that this parameter and b needs grad
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# Computing gradients
# loss.backward() runs the optimization
loss.backward()
print(w.grad)
print(b.grad)

# Disabling gradient tracking
z = torch.matmul(x, w)+b
print(z.requires_grad)

# Run without gradient tracking now.
# Useful when you only need forward computations or to freeze a parameter
with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

# Using the detach method has the same effect
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

# Tensor gradients and Jacobian products
inp = torch.eye(4, 5, requires_grad=True)
out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")
