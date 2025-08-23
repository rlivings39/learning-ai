r"""
Intro to PyTorch tensors from https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
"""

import numpy as np
import torch

# Initializing tensors

# From data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(f"Tensor: \n {x_data} \n")

# From NumPy array (there's a whole bridge between tensors and NumPy)
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# From another tensor. Attributes are maintained unless specifically modified
x_ones = torch.ones_like(x_data)  # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# We move our tensor to the GPU if available
if torch.cuda.is_available():
    x_data = x_data.to("cuda")

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:, 1] = 0
print(tensor)

# Concatenate along existing dimension
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# Concatenate along newly created dimension
t2 = torch.stack([tensor, tensor, tensor], dim=1)
print(t2)

# Typical matrix arithmetic operations exist

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

# Ryan: Can you allocate an output without filling it?
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# Single element tensors can be converted to a numeric value using `.item()`
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# In-place operations are named with a trailing _ and save memory
# but can cause issues with stateful operations like derivatives
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

# Tensors on CPU and NumPy arrays can share memory and do when converting
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# Modifying tensor update NumPy array
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# Similarly for NumPy to tensor
n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
