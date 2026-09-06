import torch

print(f"torch 版本: {torch.__version__}")
print(f"MPS 是否可用: {torch.backends.mps.is_available()}")
print(f"MPS 是否已编译进这个版本: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    a = torch.rand(1000, 1000, device=device)
    b = torch.rand(1000, 1000, device=device)
    c = a @ b
    print(f"在 MPS 上做矩阵乘法成功，结果形状: {c.shape}, 设备: {c.device}")
else:
    print("MPS 不可用，需要用 CPU 跑（会慢很多）")
