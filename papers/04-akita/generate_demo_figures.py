import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)

# 1. Synthetic Data Generation Logic (convergent CTCF motif architecture)
BASES = "ACGT"
IDX = {b: i for i, b in enumerate(BASES)}
COMP = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
MOTIF = "CCGCGNGGNGGCAG"

def revcomp(m):
    return "".join(COMP[b] for b in m[::-1])

def one_hot(s):
    x = np.zeros((4, len(s)), np.float32)
    for j, b in enumerate(s):
        x[IDX[b], j] = 1
    return x

SEQ_LEN = 512
NB = 16
BIN = SEQ_LEN // NB

def background(nb):
    # contact probability decays exponentially with genomic distance
    i = np.arange(nb)
    return 0.2 * np.exp(-np.abs(i[:, None] - i[None, :]) / 4.0)

def make_sample():
    seq = list("".join(np.random.choice(list(BASES), SEQ_LEN)))
    sites = []
    # Place 1 to 3 motifs randomly in bins
    for _ in range(np.random.randint(1, 4)):
        b = np.random.randint(0, NB)
        o = np.random.choice([1, -1])
        m = MOTIF if o == 1 else revcomp(MOTIF)
        m = "".join(np.random.choice(list(BASES)) if ch == "N" else ch for ch in m)
        p = b * BIN + (BIN - len(m)) // 2
        seq[p:p + len(m)] = list(m)
        sites.append((b, o))
    C = background(NB)
    # If we have a convergent pair (forward motif followed by reverse motif), we form a loop boundary/domain
    for a, oa in sites:
        for d, od in sites:
            if a < d and oa == 1 and od == -1:
                C[a, d] += 2.0
                C[d, a] += 2.0
                C[a:d + 1, a:d + 1] += 0.4
    C = np.clip(C, 0, 3.0).astype(np.float32)
    return one_hot("".join(seq)), C, sites

# Generate 2500 samples
Xs, Ys = [], []
for _ in range(2500):
    x, cmap, _ = make_sample()
    Xs.append(x)
    Ys.append(cmap)
Xs = np.stack(Xs)
Ys = np.stack(Ys)

ntr = 2000
Xtr, Xte = torch.tensor(Xs[:ntr]), torch.tensor(Xs[ntr:])
Ytr, Yte = torch.tensor(Ys[:ntr]), torch.tensor(Ys[ntr:])

# 2. MiniAkita PyTorch Model
class MiniAkita(nn.Module):
    def __init__(self, d=32, nb=NB):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(4, 48, 15, padding=7),
            nn.ReLU(),
            nn.Conv1d(48, d, 15, padding=7),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(nb)
        )
        self.head = nn.Sequential(
            nn.Linear(2 * d + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.register_buffer("pos", torch.linspace(0, 1, nb))
        
    def forward(self, x):
        E = self.enc(x).transpose(1, 2)  # [B, nb, d]
        B, nb, d = E.shape
        Ei = E[:, :, None, :].expand(B, nb, nb, d)
        Ej = E[:, None, :, :].expand(B, nb, nb, d)
        pi = self.pos[None, :, None, None].expand(B, nb, nb, 1)
        pj = self.pos[None, None, :, None].expand(B, nb, nb, 1)
        M = self.head(torch.cat([Ei, Ej, pi, pj], -1)).squeeze(-1)
        return 0.5 * (M + M.transpose(1, 2))

model = MiniAkita()
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
lf = nn.MSELoss()

# 3. Train
hist = []
batch_size = 64
for e in range(60):
    o = torch.randperm(len(Xtr))
    tot = 0.0
    for i in range(0, len(Xtr), batch_size):
        j = o[i:i + batch_size]
        opt.zero_grad()
        L = lf(model(Xtr[j]), Ytr[j])
        L.backward()
        opt.step()
        tot += L.item() * len(j)
    hist.append(tot / len(Xtr))
    if (e + 1) % 10 == 0:
        print(f"Epoch {e+1}/60 - MSE: {hist[-1]:.4f}")

# Save figures in the script's directory
output_dir = os.path.dirname(os.path.abspath(__file__))

# 4. Save Training Loss Figure
plt.figure(figsize=(5, 3))
plt.plot(range(1, 61), hist, 'r-', linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Toy Model Training Loss")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "toy_training_loss.png"), dpi=150)
plt.close()

# 5. Predict on Test
with torch.no_grad():
    Pte = model(Xte)

# Save Comparison Figure (Target vs Prediction)
# Find a sample with a clear loop/domain (max intensity in upper triangle)
best_k = 0
max_val = 0
for idx in range(len(Yte)):
    upper_tri = torch.triu(Yte[idx], diagonal=2)
    val = upper_tri.max().item()
    if val > max_val:
        max_val = val
        best_k = idx

k = best_k
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
im0 = ax[0].imshow(Yte[k].numpy(), cmap="Reds", vmin=0, vmax=2.5)
ax[0].set_title("Target Contact Map")
ax[0].set_xlabel("Bin")
ax[0].set_ylabel("Bin")

im1 = ax[1].imshow(Pte[k].numpy(), cmap="Reds", vmin=0, vmax=2.5)
ax[1].set_title("Predicted Contact Map")
ax[1].set_xlabel("Bin")
ax[1].set_ylabel("Bin")

fig.colorbar(im1, ax=ax, fraction=0.025, pad=0.04)
plt.suptitle("Akita 3D Genome Folding Target vs. Prediction")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "toy_prediction_comparison.png"), dpi=150)
plt.close()

# 6. Mutagenesis Comparison
x_wt = Xte[best_k:best_k+1] # shape [1, 4, 512]
x_mut = x_wt.clone()

# Replace the motif regions (middle of each bin) with random one-hot vectors to disrupt CTCF binding
for b in range(NB):
    p = b * BIN + (BIN - 14) // 2
    for pos in range(p, p + 14):
        x_mut[0, :, pos] = 0
        rand_idx = np.random.randint(0, 4)
        x_mut[0, rand_idx, pos] = 1.0

with torch.no_grad():
    p_wt = model(x_wt)[0]
    p_mut = model(x_mut)[0]

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
im0 = ax[0].imshow(p_wt.numpy(), cmap="Reds", vmin=0, vmax=2.5)
ax[0].set_title("Wild Type Prediction")
ax[0].set_xlabel("Bin")
ax[0].set_ylabel("Bin")

im1 = ax[1].imshow(p_mut.numpy(), cmap="Reds", vmin=0, vmax=2.5)
ax[1].set_title("CTCF Mutated Prediction")
ax[1].set_xlabel("Bin")
ax[1].set_ylabel("Bin")

fig.colorbar(im1, ax=ax, fraction=0.025, pad=0.04)
plt.suptitle("In Silico CTCF Mutagenesis Folding Disruption")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "toy_ctcf_mutagenesis.png"), dpi=150)
plt.close()

print("Figures successfully generated!")
