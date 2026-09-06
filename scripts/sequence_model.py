import numpy as np
import torch
import torch.nn as nn

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备: {device}")

sequences = np.load("data_cleaned/nested_sequences.npy")
print(f"序列数据形状: {sequences.shape}, dtype: {sequences.dtype}")

class SetEncoder(nn.Module):
    def __init__(self, embed_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(3, embed_dim)  # 0=pad(-1平移后), 1=值0, 2=值1

    def forward(self, x):
        # x: (batch, window, inner_size), 取值 -1/0/1
        x_shifted = (x + 1).long()  # 变成 0/1/2
        embedded = self.embedding(x_shifted)  # (batch, window, inner_size, embed_dim)
        print(f"Embedding输出形状: {embedded.shape}")
        pooled = embedded.sum(dim=2)  # 对inner_size求和，(batch, window, embed_dim)
        print(f"集合求和输出形状: {pooled.shape}")
        return pooled

class SequenceModel(nn.Module):
    def __init__(self, embed_dim=8, hidden_dim=16):
        super().__init__()
        self.set_encoder = SetEncoder(embed_dim)
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        step_vectors = self.set_encoder(x)  # (batch, window, embed_dim)
        _, h_n = self.gru(step_vectors)  # h_n: (1, batch, hidden_dim)
        print(f"GRU隐藏状态形状: {h_n.shape}")
        h_last = h_n.squeeze(0)  # (batch, hidden_dim)
        print(f"GRU最后隐藏状态形状: {h_last.shape}")
        logit = self.output(h_last)  # (batch, 1)
        print(f"线性层输出形状: {logit.shape}")
        return logit

model = SequenceModel().to(device)

batch = sequences[:32]
batch_tensor = torch.tensor(batch, dtype=torch.float32, device=device)
print(f"输入batch形状: {batch_tensor.shape}, 设备: {batch_tensor.device}")

with torch.no_grad():
    output = model(batch_tensor)

print(f"模型输出形状: {output.shape}（应为 [32, 1]）")
print(f"输出所在设备: {output.device}（应为 mps，不是 cpu）")
print(f"输出示例（前5个，未过sigmoid）: {output[:5].squeeze().tolist()}")
