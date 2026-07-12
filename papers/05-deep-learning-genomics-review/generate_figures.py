import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Resolve path
script_dir = os.path.dirname(os.path.abspath(__file__))

BASES = "ACGT"
IDX = {b: i for i, b in enumerate(BASES)}

def one_hot(s):
    m = np.zeros((4, len(s)), np.float32)
    for j, b in enumerate(s):
        m[IDX[b], j] = 1
    return m

# Generate synthetic dataset
SEQ_LEN, MOTIF, N = 200, "CCGCGNGGNGGCAG", 2000
rand = lambda L: "".join(np.random.choice(list(BASES), L))
inst = lambda m: "".join(np.random.choice(list(BASES)) if b == "N" else b for b in m)

X, y = [], []
for _ in range(N):
    s = list(rand(SEQ_LEN))
    m = inst(MOTIF)
    p = np.random.randint(0, SEQ_LEN - len(m))
    s[p:p + len(m)] = list(m)
    X.append("".join(s))
    y.append(1)

for _ in range(N):
    X.append(rand(SEQ_LEN))
    y.append(0)

Xoh = np.stack([one_hot(s) for s in X])
y = np.array(y, np.float32)

# Shuffle and split
perm = np.random.permutation(len(y))
Xoh, ys = Xoh[perm], y[perm]
n = int(0.8 * len(ys))
Xtr, Xte, ytr, yte = Xoh[:n], Xoh[n:], ys[:n], ys[n:]

# CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv1d(4, 16, 15)
        self.f = nn.Linear(16, 1)
        
    def forward(self, x):
        return self.f(torch.relu(self.c(x)).max(2).values)

model = CNN()
opt = torch.optim.Adam(model.parameters(), 1e-3)
lf = nn.BCEWithLogitsLoss()

# Training loop
Xt, yt = torch.tensor(Xtr), torch.tensor(ytr).unsqueeze(1)
hist = []
for e in range(15):
    o = torch.randperm(len(Xt))
    tot = 0.0
    for i in range(0, len(Xt), 128):
        j = o[i:i + 128]
        opt.zero_grad()
        L = lf(model(Xt[j]), yt[j])
        L.backward()
        opt.step()
        tot += L.item() * len(j)
    hist.append(tot / len(Xt))
    print(f"epoch {e+1:2d}/15  loss {hist[-1]:.4f}")

# Plot 1: Training loss
plt.figure(figsize=(5, 3))
plt.plot(range(1, 16), hist, marker="o", color="#3498db")
plt.xlabel("epoch")
plt.ylabel("BCE loss")
plt.title("Training loss")
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "toy_training_loss.png"), dpi=150)
plt.close()

# Evaluate test set
with torch.no_grad():
    p = torch.sigmoid(model(torch.tensor(Xte))).numpy().ravel()

# Plot 2: ROC Curve
fpr, tpr, _ = roc_curve(yte, p)
auc_val = roc_auc_score(yte, p)
plt.figure(figsize=(4, 4))
plt.plot(fpr, tpr, label=f"AUC={auc_val:.3f}", color="#e74c3c")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "toy_roc_curve.png"), dpi=150)
plt.close()

# In-silico mutagenesis (ISM) on first positive sequence in test set
pos_idx = np.where(yte == 1)[0][0]
seq = Xte[pos_idx]
ref = "".join(BASES[i] for i in seq.argmax(0))

def prob(m):
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(m[None]))).item()

r0 = prob(one_hot(ref))
D = np.zeros((4, SEQ_LEN), np.float32)
for j in range(SEQ_LEN):
    for b in range(4):
        mut = one_hot(ref).copy()
        mut[:, j] = 0
        mut[b, j] = 1
        D[b, j] = prob(mut) - r0

# Plot 3: Heatmap of Saturation Mutagenesis
v = max(abs(D).max(), 1e-6)
plt.figure(figsize=(12, 2.4))
im = plt.imshow(D, aspect="auto", cmap="RdBu", vmin=-v, vmax=v)
plt.yticks(range(4), list(BASES))
plt.xlabel("position")
plt.ylabel("alt base")
plt.title("Saturation mutagenesis: change in P(bound) per single-nucleotide variant")
plt.colorbar(im, fraction=0.025, pad=0.02)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "toy_ism_heatmap.png"), dpi=150)
plt.close()

print("Figures successfully generated!")
