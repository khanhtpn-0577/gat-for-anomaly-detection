# -*- coding: utf-8 -*-
"""
Ablation study: effect of removing each loss component.
Architecture: identical to proposed_model_edge_weights_v2.py
  - GATv2Conv (3 layers, hidden=128, out=64) + learnable edge weights S
  - adj_target = sigmoid(S_param) matrix (dynamic, detached)
  - anomaly score = 5 * BCE(adj_logits, S_mat_infer) + Σ hinge(alpha, important_edges)

Variants (only loss terms change, model arch is fixed):
  FULL        — all losses enabled (baseline)
  NO_EDGE     — remove L_edge  (hinge attention penalty)
  NO_CONTRAST — remove L_contrast (NT-Xent)
  NO_COS      — remove L_cos (cosine anti-collapse)
  NO_PE       — remove positional encoding (temporal feature)

Single run: seed=42, no multi-seed averaging.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================
# CONFIG
# =====================================================================
DATA_PATH  = os.path.join(os.path.dirname(__file__),
                          "../related_data/dataset/base_dataset.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Tuned lambda values (same as proposed_model_edge_weights_v2.py)
lambda_recon    = 3.0
lambda_contrast = 0.5
lambda_edge     = 15.0
lambda_cos      = 0.3
important_edges = [(1, 2), (0, 1), (1, 3), (1, 4)]

SEED = 42

# =====================================================================
# LOSS VARIANTS
# =====================================================================
LOSS_CONFIGS = {
    "FULL":        {"use_recon": True,  "use_edge": True,  "use_contrast": True,  "use_cosine": True,  "use_pe": True},
    "NO_RECON":    {"use_recon": False, "use_edge": True,  "use_contrast": True,  "use_cosine": True,  "use_pe": True},
    "NO_EDGE":     {"use_recon": True,  "use_edge": False, "use_contrast": True,  "use_cosine": True,  "use_pe": True},
    "NO_CONTRAST": {"use_recon": True,  "use_edge": True,  "use_contrast": False, "use_cosine": True,  "use_pe": True},
    "NO_COS":      {"use_recon": True,  "use_edge": True,  "use_contrast": True,  "use_cosine": False, "use_pe": True},
    "NO_PE":       {"use_recon": True,  "use_edge": True,  "use_contrast": True,  "use_cosine": True,  "use_pe": False},
}

# 8 directed edges & lookup map
EDGE_LIST = [(0,1),(1,0),(1,2),(2,1),(1,3),(3,1),(1,4),(4,1)]
_edge_map  = torch.full((5, 5), -1, dtype=torch.long)
for _idx, (_s, _d) in enumerate(EDGE_LIST):
    _edge_map[_s, _d] = _idx

# =====================================================================
# DATA LOADING
# =====================================================================
df = pd.read_csv(DATA_PATH)
if "tcp?ActiveOpens" in df.columns:
    df["tcpActiveOpens"] = df["tcp?ActiveOpens"]

df_normal_all = df[df["class"] == "normal"].reset_index(drop=True)
df_attacks    = df[df["class"] != "normal"].reset_index(drop=True)
attack_types  = df_attacks["class"].unique()

df_normal_all   = df_normal_all.sample(frac=1, random_state=SEED).reset_index(drop=True)
df_normal_train = df_normal_all.iloc[:500].reset_index(drop=True)
df_normal_test  = df_normal_all.iloc[500:600].reset_index(drop=True)

print(f"Train normal: {len(df_normal_train)} | Test normal: {len(df_normal_test)}")
print(f"Attack types: {list(attack_types)}")

# =====================================================================
# FEATURE GROUPS
# =====================================================================
group_cols = {
    "Interface": ["ifInOctets11","ifOutOctets11","ifoutDiscards11",
                  "ifInUcastPkts11","ifInNUcastPkts11","ifInDiscards11",
                  "ifOutUcastPkts11","ifOutNUcastPkts11"],
    "IP":        ["ipInReceives","ipInDelivers","ipOutRequests",
                  "ipOutDiscards","ipInDiscards","ipForwDatagrams",
                  "ipOutNoRoutes","ipInAddrErrors"],
    "TCP":       ["tcpOutRsts","tcpInSegs","tcpOutSegs","tcpPassiveOpens",
                  "tcpRetransSegs","tcpCurrEstab","tcpEstabResets","tcpActiveOpens"],
    "UDP":       ["udpInDatagrams","udpOutDatagrams","udpInErrors","udpNoPorts"],
    "ICMP":      ["icmpInMsgs","icmpInDestUnreachs","icmpOutMsgs",
                  "icmpOutDestUnreachs","icmpInEchos","icmpOutEchoReps"],
}
group_cols = {k: [c for c in v if c in df.columns] for k, v in group_cols.items()}
node_order = ["Interface", "IP", "TCP", "UDP", "ICMP"]
max_dim    = max(len(v) for v in group_cols.values())

edge_index = torch.tensor(
    [[0,1],[1,0],[1,2],[2,1],[1,3],[3,1],[1,4],[4,1]]
).t().long()

# =====================================================================
# FEATURE CREATION
# =====================================================================
def create_features(df_set, N, scalers=None, fit=False):
    x_padded = torch.zeros(N, 5, max_dim)
    if fit:
        scalers = {}
    for i, group in enumerate(node_order):
        cols  = [c for c in group_cols[group] if c in df_set.columns]
        raw   = df_set[cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(float)
        if fit:
            sc = StandardScaler()
            scaled = sc.fit_transform(raw)
            scalers[group] = sc
        else:
            scaled = scalers[group].transform(raw)
        feats = torch.tensor(scaled[:N], dtype=torch.float)
        x_padded[:, i, :feats.shape[1]] = feats
    return x_padded, scalers


def add_time_emb(x_tensor):
    """(N, 5, F) → (N, 5, F+1) with sinusoidal PE."""
    N  = x_tensor.shape[0]
    te = torch.zeros(N, 5, 1)
    for t in range(N):
        te[t] = torch.sin(torch.tensor(t / N * 2 * np.pi).float())
    return torch.cat([x_tensor, te], dim=-1)


# Pre-build normal features WITH and WITHOUT PE
N_normal = len(df_normal_train)
x_normal_base, scalers_normal = create_features(df_normal_train, N_normal, fit=True)
x_normal_pe    = add_time_emb(x_normal_base)   # (N, 5, max_dim+1)
x_normal_no_pe = x_normal_base                 # (N, 5, max_dim)

in_ch_pe    = x_normal_pe.shape[-1]
in_ch_no_pe = x_normal_no_pe.shape[-1]
print(f"in_channels (with PE)={in_ch_pe} | (no PE)={in_ch_no_pe} | max_dim={max_dim}")

# =====================================================================
# HELPERS: S ↔ edge_attr / 5×5 matrix
# =====================================================================
def s_to_edge_attr(S_vals: torch.Tensor, ei: torch.Tensor) -> torch.Tensor:
    src_local = ei[0] % 5
    dst_local = ei[1] % 5
    idx = _edge_map.to(S_vals.device)[src_local, dst_local]
    return S_vals[idx].unsqueeze(-1)   # (|E|, 1)


def s_to_matrix(S_vals: torch.Tensor) -> torch.Tensor:
    mat = torch.zeros(5, 5, device=S_vals.device)
    for i, (s, d) in enumerate(EDGE_LIST):
        mat[s, d] = S_vals[i]
    return mat

# =====================================================================
# NT-Xent LOSS
# =====================================================================
def nt_xent_loss(emb1, emb2, temperature=0.1):
    emb1   = F.normalize(emb1, dim=-1)
    emb2   = F.normalize(emb2, dim=-1)
    sim    = torch.mm(emb1, emb2.t()) / temperature
    labels = torch.arange(emb1.size(0), device=emb1.device)
    return F.cross_entropy(sim, labels)

# =====================================================================
# MODEL — identical to proposed_model_edge_weights_v2.py
# =====================================================================
class ContrastiveGAT(nn.Module):
    def __init__(self, in_ch: int, hid: int, out_ch: int):
        super().__init__()
        self.gat1 = GATv2Conv(in_ch,   hid,    heads=4, concat=True,
                              dropout=0.2, edge_dim=1)
        self.gat2 = GATv2Conv(hid*4,   hid,    heads=4, concat=True,
                              dropout=0.2, edge_dim=1)
        self.gat3 = GATv2Conv(hid*4,   out_ch, heads=1, concat=False,
                              dropout=0.1, edge_dim=1)
        self.res  = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, data, edge_attr: torch.Tensor):
        x, ei = data.x, data.edge_index
        r   = self.res(x)
        x   = F.elu(self.gat1(x, ei, edge_attr=edge_attr))
        x   = F.elu(self.gat2(x, ei, edge_attr=edge_attr))
        emb = self.gat3(x, ei, edge_attr=edge_attr) + r
        B   = data.num_graphs
        emb        = emb.view(B, 5, -1)
        adj_logits = torch.bmm(emb, emb.transpose(1, 2))
        alpha      = F.softmax(adj_logits, dim=-1)
        return emb, adj_logits, alpha

# =====================================================================
# ANOMALY SCORE — same as proposed_model_edge_weights_v2.py
# =====================================================================
def anomaly_score(adj_logits, alpha_s, S_mat_tgt):
    recon = F.binary_cross_entropy_with_logits(adj_logits, S_mat_tgt).item()
    edge  = sum(max(0.0, 0.5 - alpha_s[i, j].item()) * lambda_edge
                for i, j in important_edges)
    return recon * 5 + edge

# =====================================================================
# TRAIN + EVAL
# =====================================================================
def train_and_eval(config_name: str, loss_config: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"  Config: {config_name}  |  {loss_config}")
    print(f"{'='*60}")

    use_pe = loss_config["use_pe"]
    in_ch  = in_ch_pe if use_pe else in_ch_no_pe
    x_train = x_normal_pe if use_pe else x_normal_no_pe

    data_list = [Data(x=x_train[t], edge_index=edge_index.clone())
                 for t in range(N_normal)]

    # Fresh S_param and model for each config
    torch.manual_seed(SEED)
    _s_init = torch.rand(8) * 0.6 + 0.2
    S_param = nn.Parameter(
        torch.log(_s_init / (1 - _s_init)).to(device)
    )

    model      = ContrastiveGAT(in_ch, hid=128, out_ch=64).to(device)
    optimizer  = torch.optim.Adam([
        {"params": model.parameters(), "lr": 1e-3, "weight_decay": 1e-5},
        {"params": [S_param],          "lr": 5e-3},
    ])
    scheduler  = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    recon_crit = nn.BCEWithLogitsLoss()
    loader     = DataLoader(data_list, batch_size=256, shuffle=True)

    best_loss, counter = float("inf"), 0
    S_epoch_log = []
    history = {k: [] for k in ["epoch", "recon", "contrast", "edge", "cosine", "total"]}

    print("\nTraining...\n")
    for epoch in range(400):
        model.train()
        tot = dict(loss=0., recon=0., contrast=0., edge=0., cos=0.)
        n_batch = 0

        for batch in loader:
            batch = batch.to(device)
            B     = batch.num_graphs
            optimizer.zero_grad()

            S  = torch.sigmoid(S_param)
            ea = s_to_edge_attr(S, batch.edge_index)

            emb1, adj_logits1, alpha1 = model(batch, ea)

            # Augmented view
            aug = batch.clone()
            aug.x = aug.x + 0.01 * torch.randn_like(aug.x)
            aug.edge_index, _ = dropout_edge(aug.edge_index, p=0.1)
            ea_aug = s_to_edge_attr(S, aug.edge_index)
            emb2, _, _ = model(aug, ea_aug)

            # adj_target from S (detached)
            S_mat        = s_to_matrix(S.detach())
            adj_target_b = S_mat.unsqueeze(0).expand(B, -1, -1)

            # Loss 1: reconstruction
            recon_loss = (recon_crit(adj_logits1, adj_target_b)
                          if loss_config["use_recon"]
                          else torch.tensor(0.0, device=device))

            # Loss 2: contrastive
            contrast_loss = (nt_xent_loss(emb1.mean(dim=1), emb2.mean(dim=1))
                             if loss_config["use_contrast"]
                             else torch.tensor(0.0, device=device))

            # Loss 3: cosine anti-collapse
            if loss_config["use_cosine"]:
                emb_n = F.normalize(emb1, dim=-1)
                cos   = torch.bmm(emb_n, emb_n.transpose(1, 2))
                cos   = cos.masked_fill(
                    torch.eye(5, device=device).bool().unsqueeze(0), 0.0)
                cosine_penalty = cos.mean()
            else:
                cosine_penalty = torch.tensor(0.0, device=device)

            # Loss 4: important-edge hinge
            if loss_config["use_edge"]:
                edge_loss = torch.tensor(0.0, device=device)
                for i, j in important_edges:
                    edge_loss = edge_loss + F.relu(0.5 - alpha1[:, i, j].mean()) * lambda_edge
            else:
                edge_loss = torch.tensor(0.0, device=device)

            loss = (lambda_recon    * recon_loss
                  + lambda_contrast * contrast_loss
                  + lambda_cos      * cosine_penalty
                  + edge_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_([S_param], 1.0)
            optimizer.step()

            tot["loss"]     += loss.item()
            tot["recon"]    += recon_loss.item()
            tot["contrast"] += contrast_loss.item()
            tot["edge"]     += edge_loss.item()
            tot["cos"]      += cosine_penalty.item()
            n_batch         += 1

        avg = {k: v / n_batch for k, v in tot.items()}
        scheduler.step(avg["loss"])
        S_epoch_log.append(torch.sigmoid(S_param.detach()).cpu())

        history["epoch"].append(epoch)
        history["recon"].append(avg["recon"])
        history["contrast"].append(avg["contrast"])
        history["edge"].append(avg["edge"])
        history["cosine"].append(avg["cos"])
        history["total"].append(avg["loss"])

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | recon={avg['recon']:.4f}  "
                  f"contrast={avg['contrast']:.4f}  edge={avg['edge']:.4f}  "
                  f"cosine={avg['cos']:.4f}  total={avg['loss']:.4f}")

        if avg["loss"] < best_loss - 5e-4:
            best_loss = avg["loss"]
            counter   = 0
        else:
            counter += 1
            if counter >= 50:
                print(f"  Early stopping at epoch {epoch}")
                break

    # ── S_infer ───────────────────────────────────────────────────────
    model.eval()
    S_infer     = torch.stack(S_epoch_log).mean(dim=0).to(device)
    S_mat_infer = s_to_matrix(S_infer)

    # ── threshold from normal TRAIN ───────────────────────────────────
    normal_scores = []
    with torch.no_grad():
        for db in DataLoader(data_list, batch_size=1, shuffle=False):
            db  = db.to(device)
            ea_ = s_to_edge_attr(S_infer, db.edge_index)
            _, al, alph = model(db, ea_)
            normal_scores.append(
                anomaly_score(al.squeeze(0), alph.squeeze(0), S_mat_infer))

    n_mean = np.mean(normal_scores)
    n_std  = np.std(normal_scores)
    thresh = n_mean + 2 * n_std
    print(f"\n  Threshold: mean={n_mean:.4f}  std={n_std:.4f}  thresh={thresh:.4f}")

    # ── eval helper ───────────────────────────────────────────────────
    def eval_df(df_set):
        N = len(df_set)
        if "tcp?ActiveOpens" in df_set.columns:
            df_set = df_set.copy()
            df_set["tcpActiveOpens"] = df_set["tcp?ActiveOpens"]
        x, _ = create_features(df_set, N, scalers=scalers_normal, fit=False)
        if use_pe:
            x = add_time_emb(x)
        dl = [Data(x=x[t], edge_index=edge_index.clone()) for t in range(N)]
        scores = []
        with torch.no_grad():
            for db in DataLoader(dl, batch_size=1, shuffle=False):
                db  = db.to(device)
                ea_ = s_to_edge_attr(S_infer, db.edge_index)
                _, al, alph = model(db, ea_)
                scores.append(
                    anomaly_score(al.squeeze(0), alph.squeeze(0), S_mat_infer))
        return np.array(scores)

    # ── normal TEST → FP/TN ───────────────────────────────────────────
    ntest_scores = eval_df(df_normal_test)
    FP = int((ntest_scores > thresh).sum())
    TN = len(ntest_scores) - FP

    # ── attack eval → TP/FN ───────────────────────────────────────────
    total_TP = total_attack = 0
    per_attack = {}
    for at in attack_types:
        df_at  = df_attacks[df_attacks["class"] == at].reset_index(drop=True)
        scores = eval_df(df_at)
        nd     = int((scores > thresh).sum())
        total_TP     += nd
        total_attack += len(scores)
        per_attack[at] = nd / len(scores)
        print(f"  {at:<14}  detected={nd}/{len(scores)}  recall={per_attack[at]:.4f}")

    FN        = total_attack - total_TP
    precision = total_TP / (total_TP + FP + 1e-8)
    recall    = total_TP / (total_attack + 1e-8)
    accuracy  = (total_TP + TN) / (total_attack + len(ntest_scores))
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    fpr       = FP / len(ntest_scores)

    print(f"\n  TP={total_TP}  FN={FN}  FP={FP}  TN={TN}")
    print(f"  Recall={recall:.4f}  Precision={precision:.4f}  "
          f"F1={f1:.4f}  Accuracy={accuracy:.4f}  FPR={fpr:.4f}")

    return {
        "recall": recall, "precision": precision,
        "accuracy": accuracy, "f1": f1, "fpr": fpr,
        "TP": total_TP, "FN": FN, "FP": FP, "TN": TN,
        "thresh": thresh, "n_mean": n_mean, "n_std": n_std,
        "per_attack": per_attack,
        "history": history,
    }

# =====================================================================
# RUN ALL CONFIGS
# =====================================================================
all_results = {}
for name, cfg in LOSS_CONFIGS.items():
    all_results[name] = train_and_eval(name, cfg)

# =====================================================================
# SUMMARY TABLE (console)
# =====================================================================
print("\n\n" + "=" * 95)
print("ABLATION SUMMARY — Loss Components (GATv2 + Learnable S)")
print("=" * 95)
print(f"{'Config':<15} {'Recall':>8} {'Precision':>10} {'F1':>8} {'Accuracy':>10} {'FPR':>8} {'FP':>5} {'FN':>5}")
print("-" * 95)
for name, m in all_results.items():
    print(f"{name:<15} {m['recall']:>8.4f} {m['precision']:>10.4f} "
          f"{m['f1']:>8.4f} {m['accuracy']:>10.4f} {m['fpr']:>8.4f} "
          f"{m['FP']:>5} {m['FN']:>5}")
print("=" * 95)

atk_list = list(attack_types)
print("\nPER-ATTACK RECALL")
print("-" * 95)
print(f"{'Config':<15}" + "".join(f"{at[:11]:>13}" for at in atk_list))
print("-" * 95)
for name, m in all_results.items():
    row = f"{name:<15}"
    for at in atk_list:
        row += f"{m['per_attack'].get(at, 0):>13.4f}"
    print(row)
print("=" * 95)

# =====================================================================
# SAVE — loss curves + markdown
# =====================================================================

# ── Loss curves (one subplot per config) ─────────────────────────────
fig, axes = plt.subplots(len(LOSS_CONFIGS), 1,
                         figsize=(12, 4 * len(LOSS_CONFIGS)), sharex=False)
for ax, (name, m) in zip(axes, all_results.items()):
    hist = m["history"]
    ax.plot(hist["epoch"], hist["total"],    label="Total",    linewidth=1.5)
    ax.plot(hist["epoch"], hist["recon"],    label="Recon",    linewidth=1.2, linestyle="--")
    ax.plot(hist["epoch"], hist["contrast"], label="Contrast", linewidth=1.2, linestyle="--")
    ax.plot(hist["epoch"], hist["edge"],     label="Edge",     linewidth=1.2, linestyle="--")
    ax.plot(hist["epoch"], hist["cosine"],   label="Cosine",   linewidth=1.2, linestyle="--")
    ax.set_title(f"Training Loss — {name}", fontsize=11)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "ablation_L_component_loss_curves.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nLoss curves saved → {plot_path}")

# ── Markdown ──────────────────────────────────────────────────────────
metric_rows = ["| Config | Recall | Precision | F1 | Accuracy | FPR | FP | FN |",
               "|---|---|---|---|---|---|---|---|"]
for name, m in all_results.items():
    metric_rows.append(
        f"| {name} | {m['recall']:.4f} | {m['precision']:.4f} | "
        f"{m['f1']:.4f} | {m['accuracy']:.4f} | {m['fpr']:.4f} | "
        f"{m['FP']} | {m['FN']} |")
metrics_table = "\n".join(metric_rows)

atk_rows = ["| Config | " + " | ".join(atk_list) + " |",
            "|---|" + "---|" * len(atk_list)]
for name, m in all_results.items():
    vals = " | ".join(f"{m['per_attack'].get(at, 0):.4f}" for at in atk_list)
    atk_rows.append(f"| {name} | {vals} |")
per_attack_table = "\n".join(atk_rows)

md_path = os.path.join(OUTPUT_DIR, "ablation_L_component.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write(f"""# Ablation Study — Loss Components

## Setup

| Parameter | Value |
|---|---|
| Model | GATv2Conv (3 layers, hidden=128, out=64) + learnable S |
| adj_target | sigmoid(S_param) matrix — dynamic, detached |
| lambda_recon | {lambda_recon} |
| lambda_contrast | {lambda_contrast} |
| lambda_edge | {lambda_edge} |
| lambda_cos | {lambda_cos} |
| max_epochs | 400 |
| early_stopping | patience=50, tol=5e-4 |
| seed | {SEED} |
| Train normal | 500 | Test normal | 100 |

| Config | use_recon | use_edge | use_contrast | use_cosine | use_pe |
|---|---|---|---|---|---|
| FULL | ✓ | ✓ | ✓ | ✓ | ✓ |
| NO_RECON | ✗ | ✓ | ✓ | ✓ | ✓ |
| NO_EDGE | ✓ | ✗ | ✓ | ✓ | ✓ |
| NO_CONTRAST | ✓ | ✓ | ✗ | ✓ | ✓ |
| NO_COS | ✓ | ✓ | ✓ | ✗ | ✓ |
| NO_PE | ✓ | ✓ | ✓ | ✓ | ✗ |

---

## Training Loss Curves

![Loss curves](ablation_L_component_loss_curves.png)

---

## Overall Metrics

{metrics_table}

---

## Per-Attack Recall

{per_attack_table}
""")

print(f"Markdown saved → {md_path}")
