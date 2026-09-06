import numpy as np
import torch
import torch.nn as nn
import time
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, average_precision_score

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备: {device}")

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
        return embedded.sum(dim=2)

class SequenceModel(nn.Module):
    def __init__(self, embed_dim=8, hidden_dim=16):
        super().__init__()
        self.set_encoder = SetEncoder(embed_dim)
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        step_vectors = self.set_encoder(x)
        _, h_n = self.gru(step_vectors)
        return self.output(h_n.squeeze(0))

def train_and_eval(sequences, window_name):
    model = SequenceModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    BATCH_SIZE = 512
    N_EPOCHS = 5
    N_TRAIN = len(train_seq_idx)
    n_batches = (N_TRAIN + BATCH_SIZE - 1) // BATCH_SIZE

    start_time = time.time()
    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0.0
        perm = np.random.permutation(N_TRAIN)
        for b in range(n_batches):
            pos = perm[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
            x = torch.tensor(sequences[train_seq_idx[pos]], dtype=torch.float32, device=device)
            y = torch.tensor(train_labels[pos], dtype=torch.float32, device=device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(pos)
        print(f"[{window_name}] epoch {epoch+1}/{N_EPOCHS}: loss={epoch_loss/N_TRAIN:.4f}")
    print(f"[{window_name}] 训练耗时: {time.time()-start_time:.1f}秒")

    model.eval()
    N_TEST = len(test_seq_idx)
    all_proba = []
    with torch.no_grad():
        for b in range(0, N_TEST, 4096):
            x = torch.tensor(sequences[test_seq_idx[b:b+4096]], dtype=torch.float32, device=device)
            all_proba.append(torch.sigmoid(model(x)).cpu().numpy().flatten())
    proba = np.concatenate(all_proba)

    return dict(
        auc=roc_auc_score(test_labels, proba),
        log_loss=log_loss(test_labels, proba),
        brier=brier_score_loss(test_labels, proba),
        pr_auc=average_precision_score(test_labels, proba),
    )

results = {}
results['model2'] = dict(auc=0.738182, log_loss=0.519266, brier=0.172884, pr_auc=0.499545)
results['窗口50(已有)'] = dict(auc=0.771101, log_loss=0.492565, brier=0.163774, pr_auc=0.547503)

seq20 = np.load("data_cleaned/nested_sequences_w20.npy")
results['窗口20'] = train_and_eval(seq20, "窗口20")
del seq20

seq100 = np.load("data_cleaned/nested_sequences_w100.npy")
results['窗口100'] = train_and_eval(seq100, "窗口100")
del seq100

print("\n=== 汇总对比 ===")
print(f"{'模型':<12} {'AUC':>10} {'LogLoss':>10} {'Brier':>10} {'PR-AUC':>10}")
for name, r in results.items():
    print(f"{name:<12} {r['auc']:>10.6f} {r['log_loss']:>10.6f} {r['brier']:>10.6f} {r['pr_auc']:>10.6f}")
