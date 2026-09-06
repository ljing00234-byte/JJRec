import numpy as np
import torch
import torch.nn as nn
import time

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备: {device}")

sequences = np.load("data_cleaned/nested_sequences.npy")
train_seq_idx = np.load("data_cleaned/train_seq_idx.npy")
train_labels = np.load("data_cleaned/train_labels.npy")

N_SAMPLE = 50000
sample_idx = train_seq_idx[:N_SAMPLE]
sample_labels = train_labels[:N_SAMPLE]

class SetEncoder(nn.Module):
    def __init__(self, embed_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(3, embed_dim)

    def forward(self, x):
        x_shifted = (x + 1).long()
        embedded = self.embedding(x_shifted)
        pooled = embedded.sum(dim=2)
        return pooled

class SequenceModel(nn.Module):
    def __init__(self, embed_dim=8, hidden_dim=16):
        super().__init__()
        self.set_encoder = SetEncoder(embed_dim)
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        step_vectors = self.set_encoder(x)
        _, h_n = self.gru(step_vectors)
        h_last = h_n.squeeze(0)
        logit = self.output(h_last)
        return logit

model = SequenceModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

BATCH_SIZE = 512
N_EPOCHS = 3
n_batches = (N_SAMPLE + BATCH_SIZE - 1) // BATCH_SIZE

start_time = time.time()
for epoch in range(N_EPOCHS):
    epoch_start_time = time.time()
    epoch_loss = 0.0
    perm = np.random.permutation(N_SAMPLE)
    for b in range(n_batches):
        batch_positions = perm[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        batch_seq_idx = sample_idx[batch_positions]
        batch_seqs = sequences[batch_seq_idx]
        batch_y = sample_labels[batch_positions]

        x = torch.tensor(batch_seqs, dtype=torch.float32, device=device)
        y = torch.tensor(batch_y, dtype=torch.float32, device=device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(batch_positions)

    epoch_loss /= N_SAMPLE
    epoch_elapsed = time.time() - epoch_start_time
    elapsed = time.time() - start_time
    print(f"epoch {epoch + 1}/{N_EPOCHS}: loss={epoch_loss:.4f}, 本轮耗时={epoch_elapsed:.1f}秒, 累计耗时={elapsed:.1f}秒")

total_time = time.time() - start_time
time_per_epoch = total_time / N_EPOCHS
print(f"\n平均每轮耗时: {time_per_epoch:.1f}秒")
full_train_estimate = time_per_epoch * (4612163 / N_SAMPLE)
print(f"按此速度推算，完整训练集（461万行）跑一轮预计耗时: {full_train_estimate:.1f}秒"
      f"（约{full_train_estimate / 60:.1f}分钟）")
