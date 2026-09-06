import numpy as np
import torch
import torch.nn as nn
import time
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, average_precision_score

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备: {device}")

sequences = np.load("data_cleaned/nested_sequences.npy")
train_seq_idx = np.load("data_cleaned/train_seq_idx.npy")
train_labels = np.load("data_cleaned/train_labels.npy")
test_seq_idx = np.load("data_cleaned/test_seq_idx.npy")
test_labels = np.load("data_cleaned/test_labels.npy")

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
N_EPOCHS = 5
N_TRAIN = len(train_seq_idx)
n_batches = (N_TRAIN + BATCH_SIZE - 1) // BATCH_SIZE

print(f"训练集行数: {N_TRAIN}, 每轮批次数: {n_batches}")

start_time = time.time()
for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss = 0.0
    perm = np.random.permutation(N_TRAIN)
    for b in range(n_batches):
        batch_positions = perm[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        batch_seq_idx = train_seq_idx[batch_positions]
        batch_seqs = sequences[batch_seq_idx]
        batch_y = train_labels[batch_positions]

        x = torch.tensor(batch_seqs, dtype=torch.float32, device=device)
        y = torch.tensor(batch_y, dtype=torch.float32, device=device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(batch_positions)

    epoch_loss /= N_TRAIN
    elapsed = time.time() - start_time
    print(f"epoch {epoch + 1}/{N_EPOCHS}: loss={epoch_loss:.4f}, 累计耗时={elapsed:.1f}秒")

print(f"\n训练总耗时: {time.time() - start_time:.1f}秒")

print("\n=== 在测试集上评估 ===")
model.eval()
N_TEST = len(test_seq_idx)
EVAL_BATCH = 4096
all_proba = []

with torch.no_grad():
    for b in range(0, N_TEST, EVAL_BATCH):
        batch_seq_idx = test_seq_idx[b:b + EVAL_BATCH]
        batch_seqs = sequences[batch_seq_idx]
        x = torch.tensor(batch_seqs, dtype=torch.float32, device=device)
        logits = model(x)
        proba = torch.sigmoid(logits).cpu().numpy().flatten()
        all_proba.append(proba)

proba = np.concatenate(all_proba)
y_test = test_labels

auc = roc_auc_score(y_test, proba)
ll = log_loss(y_test, proba)
brier = brier_score_loss(y_test, proba)
pr_auc = average_precision_score(y_test, proba)

print(f"序列模型: AUC={auc:.6f}, LogLoss={ll:.6f}, Brier={brier:.6f}, PR-AUC={pr_auc:.6f}")

model2_baseline = dict(auc=0.738182, log_loss=0.519266, brier=0.172884, pr_auc=0.499545)
seq_result = dict(auc=auc, log_loss=ll, brier=brier, pr_auc=pr_auc)

print("\n=== 对比（序列模型 - model2）===")
for metric in ['auc', 'log_loss', 'brier', 'pr_auc']:
    diff = seq_result[metric] - model2_baseline[metric]
    print(f"{metric}: model2={model2_baseline[metric]:.6f}, 序列模型={seq_result[metric]:.6f}, 差值={diff:+.6f}")
