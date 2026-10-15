import torch

x = torch.tensor([[1, 2], [3, 4]])

print(x)
print("Shape:", x.shape)
print("Device:", x.device)

x_gpu = x.to("mps")

print("GPU Tensor:", x_gpu)
print("GPU Device:", x_gpu.device)

print("Original dtype:", x.dtype)

x_float = x.float()

print("Float dtype:", x_float.dtype)
print("Float Tensor:", x_float)

W = torch.tensor([[2.0, 0.0],
                  [0.0, 3.0]])

Y = x_float @ W

print("Matrix multiplication result:")
print(Y)
print("Output shape:", Y.shape)