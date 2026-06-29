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

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# CONFIG
# =====================================================================
DATA_PATH = os.path.join(os.path.dirname(__file__),
                         "../related_data/dataset/base_dataset.csv")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

lambda_recon    = 3.0
lambda_contrast = 0.5   # updated from 0.3 (grid search best)
lambda_edge     = 15.0
lambda_cos      = 0.3   # updated from 0.1 (grid search best)
important_edges = [(1, 2), (0, 1), (1, 3), (1, 4)]

SEEDS = [0, 1, 2]       # 3 independent runs with different normal subsets

# 8 directed edges & lookup map (global, built once)
EDGE_LIST = [(0,1),(1,0),(1,2),(2,1),(1,3),(3,1),(1,4),(4,1)]
_edge_map  = torch.full((5, 5), -1, dtype=torch.long)
for _idx, (_s, _d) in enumerate(EDGE_LIST):
    _edge_map[_s, _d] = _idx

# =====================================================================
# DATA LOADING (done once — full dataset)
# =====================================================================
df = pd.read_csv(DATA_PATH)
if "tcp?ActiveOpens" in df.columns:
    df["tcpActiveOpens"] = df["tcp?ActiveOpens"]

df_normal_all = df[df["class"] == "normal"].reset_index(drop=True)
df_attacks    = df[df["class"] != "normal"].reset_index(drop=True)
attack_types  = df_attacks["class"].unique()

print(f"Total normal: {len(df_normal_all)} | Attack types: {attack_types}")

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

in_channels = max_dim + 1   # +1 for sinusoidal positional encoding
print(f"in_channels={in_channels} | max_dim={max_dim}")

edge_index = torch.tensor(
    [[0,1],[1,0],[1,2],[2,1],[1,3],[3,1],[1,4],[4,1]]
).t().long() # khoi tao tensor (2,8) --> transpose thanh (8,2) va chuyen sang long

# =====================================================================
# FEATURE CREATION
# =====================================================================
def create_features(df_set, N, scalers=None, fit=False): #fit = true khi train, false khi test
    x_padded = torch.zeros(N, 5, max_dim)
    if fit:
        scalers = {}
    for i, group in enumerate(node_order):
        cols  = [c for c in group_cols[group] if c in df_set.columns]
        raw   = df_set[cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(float) # thay NaN bang 0, thay inf bang 0, chuyen sang float
        if fit:
            sc = StandardScaler()
            scaled = sc.fit_transform(raw) #ket qua scale
            scalers[group] = sc # scaler
        else:
            scaled = scalers[group].transform(raw)
        feats = torch.tensor(scaled[:N], dtype=torch.float) #chi lay N dong dau (vi hien tai scaled la tinh cho toan df_set) va chuyen sang tensor, shape (N, num_features)
        x_padded[:, i, :feats.shape[1]] = feats
    return x_padded, scalers


# =====================================================================
# HELPERS: S (8,) ↔ edge_attr / 5×5 matrix
# =====================================================================
def s_to_edge_attr(S_vals: torch.Tensor, ei: torch.Tensor) -> torch.Tensor: # dam bao viec lay S theo dung thu tu edge_index (8 edges) va tra ve edge_attr (|E|, 1). Vi khi xu ly theo batch thi thu tu edge_index co the bi tron, can dam bao lay S theo dung thu tu edge_index (8 edges) va tra ve edge_attr (|E|, 1)
    """
    Muc tieu: tu S_vals (8,) lay ra edge_attr (|E|, 1) theo dung thu tu edge_index (8 edges).
    s_vals la 8 gia tri dung theo thu tu index.
    ei la edge_index (2, |E|) chua index cua cac edge trong graph (co the bi tron do batch), can map sang edge_map de lay dung index, de lay s_vals cho dung.
    """
    src_local = ei[0] % 5 #do thi lon gom nhieu graph ghep lai, nen can lay mod 5 de lay index cua node trong graph dang xet
    dst_local = ei[1] % 5
    idx = _edge_map.to(S_vals.device)[src_local, dst_local]
    return S_vals[idx].unsqueeze(-1)   # (|E|, 1)


def s_to_matrix(S_vals: torch.Tensor) -> torch.Tensor: # tu S_vals (8,) tao ma tran 5x5, trong do S_vals[i] duoc dat vao vi tri (s,d) tu EDGE_LIST, cac vi tri khac la 0. Vi S_vals chi co gia tri cho 8 edge co trong EDGE_LIST, can tao ma tran 5x5 de hien thi va tinh toan sau nay.
    mat = torch.zeros(5, 5, device=S_vals.device)
    for i, (s, d) in enumerate(EDGE_LIST):
        mat[s, d] = S_vals[i]
    return mat


# =====================================================================
# NT-Xent loss
# =====================================================================
def nt_xent_loss(emb1: torch.Tensor, emb2: torch.Tensor,
                 temperature: float = 0.1) -> torch.Tensor:
    """
    emb1: (B, D) B: batch, D: embedding dim
    emb2: (B, D) cùng batch nhưng augmented view
    Mục tiêu: tính loss contrastive NT-Xent giữa emb1 và emb2, trong đó cặp (emb1[i], emb2[i]) là positive pair, còn lại là negative pairs.
    """
    emb1   = F.normalize(emb1, dim=-1)
    emb2   = F.normalize(emb2, dim=-1) # x/||x||, y/||y||
    sim    = torch.mm(emb1, emb2.t()) / temperature 
    labels = torch.arange(emb1.size(0), device=emb1.device) 
    return F.cross_entropy(sim, labels) 

# =====================================================================
# MODEL — ContrastiveGAT with GATv2Conv + edge_dim=1
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
        self.res  = (nn.Linear(in_ch, out_ch)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, data, edge_attr: torch.Tensor):
        x, ei = data.x, data.edge_index
        r   = self.res(x)
        x   = F.elu(self.gat1(x, ei, edge_attr=edge_attr))
        x   = F.elu(self.gat2(x, ei, edge_attr=edge_attr))
        emb = self.gat3(x, ei, edge_attr=edge_attr)
        B   = data.num_graphs
        emb        = emb.view(B, 5, -1)                         # (B,5,64)
        adj_logits = torch.bmm(emb, emb.transpose(1, 2))        # (B,5,5)
        alpha      = F.softmax(adj_logits, dim=-1)               # (B,5,5)
        return emb, adj_logits, alpha


# =====================================================================
# ANOMALY SCORE
# =====================================================================
def anomaly_score(adj_logits: torch.Tensor,
                  alpha_s:    torch.Tensor,
                  S_mat_tgt:  torch.Tensor) -> float:
    recon = F.binary_cross_entropy_with_logits(
        adj_logits, S_mat_tgt
    ).item()
    edge = sum(
        max(0.0, 0.5 - alpha_s[i, j].item()) * lambda_edge
        for i, j in important_edges
    )
    return recon * 5 + edge


# =====================================================================
# RUN ONE SEED
# =====================================================================
def run_one_seed(seed: int) -> dict:
    print(f"\n{'='*60}")
    print(f"  SEED = {seed}")
    print(f"{'='*60}")

    # ── data split ────────────────────────────────────────────────────
    df_norm_shuffled = df_normal_all.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_normal_train  = df_norm_shuffled.iloc[:500].reset_index(drop=True)
    df_normal_test   = df_norm_shuffled.iloc[500:600].reset_index(drop=True)
    print(f"Train normal: {len(df_normal_train)} | Test normal: {len(df_normal_test)}")

    # ── features ──────────────────────────────────────────────────────
    N_normal = len(df_normal_train)
    x_normal, scalers_normal = create_features(df_normal_train, N_normal, fit=True)

    time_emb = torch.zeros(N_normal, 5, 1)
    for t in range(N_normal):
        time_emb[t] = torch.sin(torch.tensor(t / N_normal * 2 * np.pi).float())
    x_normal = torch.cat([x_normal, time_emb], dim=-1)

    data_list_normal = [
        Data(x=x_normal[t], edge_index=edge_index.clone())
        for t in range(N_normal)
    ]

    # ── S_param (fresh each seed) ──────────────────────────────────────
    torch.manual_seed(seed)
    _s_init = torch.rand(8) * 0.6 + 0.2
    S_param = nn.Parameter(
        torch.log(_s_init / (1 - _s_init)).to(device)
    )

    # ── model & optimizer ─────────────────────────────────────────────
    model = ContrastiveGAT(in_channels, hid=128, out_ch=64).to(device)
    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": 1e-3, "weight_decay": 1e-5},
        {"params": [S_param],          "lr": 5e-3},
    ])
    scheduler  = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    recon_crit = nn.BCEWithLogitsLoss()
    loader     = DataLoader(data_list_normal, batch_size=256, shuffle=True)

    # ── training loop ─────────────────────────────────────────────────
    best_loss, counter = float("inf"), 0
    S_epoch_log: list[torch.Tensor] = []
    history = {k: [] for k in ["epoch", "recon", "contrast", "edge", "cosine", "total"]}

    print("\nTraining with learnable edge weights S...\n")

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

            # augmented view
            aug = batch.clone()
            aug.x = aug.x + 0.01 * torch.randn_like(aug.x)
            aug.edge_index, _ = dropout_edge(aug.edge_index, p=0.1)
            ea_aug = s_to_edge_attr(S, aug.edge_index)
            emb2, _, _ = model(aug, ea_aug)

            # adj_target (detached S)
            S_mat        = s_to_matrix(S.detach())
            adj_target_b = S_mat.unsqueeze(0).expand(B, -1, -1)

            # Loss 1: reconstruction
            recon_loss = recon_crit(adj_logits1, adj_target_b)

            # Loss 2: contrastive NT-Xent
            contrast_loss = nt_xent_loss(emb1.mean(dim=1), emb2.mean(dim=1))

            # Loss 3: cosine anti-collapse
            emb_n = F.normalize(emb1, dim=-1)
            cos   = torch.bmm(emb_n, emb_n.transpose(1, 2))
            cos   = cos.masked_fill(
                torch.eye(5, device=device).bool().unsqueeze(0), 0.0)
            cosine_penalty = cos.mean()

            # Loss 4: important-edge attention hinge
            edge_loss = torch.tensor(0.0, device=device)
            for i, j in important_edges:
                edge_loss = edge_loss + F.relu(0.5 - alpha1[:, i, j].mean()) * lambda_edge

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
        scheduler.step(avg["loss"]) #goi scheduler tren mean total_loss cua tung epoch
        S_epoch_log.append(torch.sigmoid(S_param.detach()).cpu())

        history["epoch"].append(epoch)
        history["recon"].append(avg["recon"])
        history["contrast"].append(avg["contrast"])
        history["edge"].append(avg["edge"])
        history["cosine"].append(avg["cos"])
        history["total"].append(avg["loss"])

        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | recon={avg['recon']:.4f}  "
                  f"contrast={avg['contrast']:.4f}  edge={avg['edge']:.4f}  "
                  f"cosine={avg['cos']:.4f}  total={avg['loss']:.4f}")

        if avg["loss"] < best_loss - 5e-4:
            best_loss = avg["loss"]
            counter   = 0
        else:
            counter += 1
            if counter >= 50:
                print(f"Early stopping at epoch {epoch}")
                break

    # ── learned S ─────────────────────────────────────────────────────
    model.eval()
    S_final = torch.sigmoid(S_param.detach())
    node_names = ["Interface", "IP", "TCP", "UDP", "ICMP"]

    print("\n" + "=" * 52)
    print(f"Learned edge weights S (final) — seed={seed}:")
    print(f"  {'Edge':<26}  S value")
    print("-" * 40)
    for (s, d), val in zip(EDGE_LIST, S_final.cpu().tolist()):
        print(f"  {node_names[s]:>9} → {node_names[d]:<9}  {val:.4f}")
    print("\nS as 5×5 matrix:")
    print(s_to_matrix(S_final.cpu()).numpy().round(4))
    print("=" * 52)

    # ── S_infer ───────────────────────────────────────────────────────
    S_infer     = torch.stack(S_epoch_log).mean(dim=0).to(device)
    S_mat_infer = s_to_matrix(S_infer)
    print("\nS_infer (mean across epochs):")
    print(s_to_matrix(S_infer.cpu()).numpy().round(4))

    # ── threshold ─────────────────────────────────────────────────────
    normal_scores = []
    with torch.no_grad():
        for db in DataLoader(data_list_normal, batch_size=1, shuffle=False):
            db  = db.to(device)
            ea_ = s_to_edge_attr(S_infer, db.edge_index)
            _, al, alph = model(db, ea_)
            normal_scores.append(
                anomaly_score(al.squeeze(0), alph.squeeze(0), S_mat_infer)
            )
    n_mean = np.mean(normal_scores)
    n_std  = np.std(normal_scores)
    thresh = n_mean + 2 * n_std
    print(f"\nThreshold: mean={n_mean:.4f}  std={n_std:.4f}  thresh={thresh:.4f}")

    # ── eval helper ───────────────────────────────────────────────────
    def eval_set(df_set: pd.DataFrame) -> np.ndarray:
        N = len(df_set)
        if "tcp?ActiveOpens" in df_set.columns:
            df_set = df_set.copy()
            df_set["tcpActiveOpens"] = df_set["tcp?ActiveOpens"]
        x, _ = create_features(df_set, N, scalers=scalers_normal, fit=False)
        te   = torch.zeros(N, 5, 1)
        for t in range(N):
            te[t] = torch.sin(torch.tensor(t / N * 2 * np.pi).float())
        x  = torch.cat([x, te], dim=-1)
        dl = [Data(x=x[t], edge_index=edge_index.clone()) for t in range(N)]
        scores = []
        with torch.no_grad():
            for db in DataLoader(dl, batch_size=1, shuffle=False):
                db  = db.to(device)
                ea_ = s_to_edge_attr(S_infer, db.edge_index)
                _, al, alph = model(db, ea_)
                scores.append(
                    anomaly_score(al.squeeze(0), alph.squeeze(0), S_mat_infer)
                )
        return np.array(scores)

    # ── per-attack eval ───────────────────────────────────────────────
    total_TP = total_attack = 0
    per_attack_recall = {}
    for at in attack_types:
        df_at  = df_attacks[df_attacks["class"] == at].reset_index(drop=True)
        scores = eval_set(df_at)
        nd     = int((scores > thresh).sum())
        total_TP     += nd
        total_attack += len(scores)
        recall_at     = nd / len(scores)
        per_attack_recall[at] = recall_at
        print(f"\n=== {at} ===  mean={scores.mean():.4f}  "
              f"detected={nd}/{len(scores)}  recall={recall_at:.4f}")

    # ── normal test ───────────────────────────────────────────────────
    ntest_scores = eval_set(df_normal_test)
    FP        = int((ntest_scores > thresh).sum())
    TN        = len(ntest_scores) - FP
    FN        = total_attack - total_TP
    precision = total_TP / (total_TP + FP + 1e-8)
    recall    = total_TP / (total_attack + 1e-8)
    accuracy  = (total_TP + TN) / (total_attack + len(ntest_scores))
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    fpr       = FP / len(ntest_scores)

    print(f"\n{'='*48}")
    print(f"TP={total_TP}  FN={FN}  FP={FP}  TN={TN}")
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"FPR:       {fpr:.4f}")
    print(f"{'='*48}")

    return {
        "recall":       recall,
        "precision":    precision,
        "accuracy":     accuracy,
        "f1":           f1,
        "fpr":          fpr,
        "per_attack":   per_attack_recall,
        "history":      history,
        "S_final":      S_final.cpu(),
        "S_infer":      S_infer.cpu(),
        "thresh":       thresh,
        "n_mean":       n_mean,
        "n_std":        n_std,
    }


# =====================================================================
# MAIN — 3 seeds, report mean ± std
# =====================================================================
all_results = []
for seed in SEEDS:
    res = run_one_seed(seed)
    all_results.append(res)

# ── aggregate global metrics ──────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  SUMMARY — {len(SEEDS)} seeds: {SEEDS}")
print(f"{'='*60}")

metrics = ["recall", "precision", "accuracy", "f1", "fpr"]
for m in metrics:
    vals = [r[m] for r in all_results]
    print(f"{m.capitalize():<12}: {np.mean(vals):.4f} ± {np.std(vals):.4f}   "
          f"(runs: {[f'{v:.4f}' for v in vals]})")

# ── aggregate per-attack recall ────────────────────────────────────────
print(f"\nPer-attack recall (mean ± std):")
for at in attack_types:
    vals = [r["per_attack"].get(at, float("nan")) for r in all_results]
    print(f"  {at:<14}: {np.mean(vals):.4f} ± {np.std(vals):.4f}   "
          f"(runs: {[f'{v:.4f}' for v in vals]})")

# =====================================================================
# SAVE OUTPUT — markdown + loss plots (seed 2 detail)
# =====================================================================
SEED2 = all_results[2]   # seed=2 is index 2
hist  = SEED2["history"]
node_names = ["Interface", "IP", "TCP", "UDP", "ICMP"]

# ── 1. Loss plots (2×2, one per component) ────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Training Loss — GATv2 (seed=2)", fontsize=14)

plot_cfg = [
    ("recon",    "Reconstruction Loss (L_recon)",   "#2563eb"),
    ("contrast", "Contrastive Loss (L_contrast)",   "#16a34a"),
    ("edge",     "Edge Attention Loss (L_edge)",    "#dc2626"),
    ("cosine",   "Cosine Penalty (L_cos)",          "#9333ea"),
]
for ax, (key, title, color) in zip(axes.flat, plot_cfg):
    ax.plot(hist["epoch"], hist[key], color=color, linewidth=1.5)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "loss_curves_gatv2_seed2.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nLoss plot saved → {plot_path}")

# ── 2. Epoch log string (all epochs, seed 2) ──────────────────────────
epoch_lines = ["| Epoch | Recon | Contrast | Edge | Cosine | Total |",
               "|------:|------:|---------:|-----:|-------:|------:|"]
for i, ep in enumerate(hist["epoch"]):
    epoch_lines.append(
        f"| {ep:5d} | {hist['recon'][i]:.4f} | {hist['contrast'][i]:.4f} | "
        f"{hist['edge'][i]:.4f} | {hist['cosine'][i]:.4f} | {hist['total'][i]:.4f} |"
    )
epoch_table = "\n".join(epoch_lines)

# ── 3. S matrix helper ─────────────────────────────────────────────────
def fmt_matrix(S_vals):
    mat = s_to_matrix(S_vals)
    rows = []
    for i, rname in enumerate(node_names):
        row = "  ".join(f"{mat[i,j].item():6.4f}" for j in range(5))
        rows.append(f"  {rname:<12} [ {row} ]")
    return "\n".join(rows)

def fmt_edge_table(S_vals, label):
    lines = [f"**{label}**\n",
             "| Edge | S value |",
             "|---|---|"]
    for (s, d), val in zip(EDGE_LIST, S_vals.tolist()):
        lines.append(f"| {node_names[s]} → {node_names[d]} | {val:.4f} |")
    return "\n".join(lines)

# ── 4. Per-attack table ───────────────────────────────────────────────
atk_header = ["| Attack | " + " | ".join(f"Seed {s}" for s in SEEDS) + " | Mean | Std |",
              "|---|" + "---|" * (len(SEEDS) + 2)]
for at in attack_types:
    vals = [r["per_attack"].get(at, float("nan")) for r in all_results]
    atk_header.append(
        f"| {at} | " + " | ".join(f"{v:.4f}" for v in vals) +
        f" | {np.mean(vals):.4f} | {np.std(vals):.4f} |"
    )
per_attack_table = "\n".join(atk_header)

# ── 5. Global metrics table ───────────────────────────────────────────
metric_rows = ["| Metric | " + " | ".join(f"Seed {s}" for s in SEEDS) + " | Mean | Std |",
               "|---|" + "---|" * (len(SEEDS) + 2)]
for m in ["recall", "precision", "accuracy", "f1", "fpr"]:
    vals = [r[m] for r in all_results]
    metric_rows.append(
        f"| {m.capitalize()} | " + " | ".join(f"{v:.4f}" for v in vals) +
        f" | {np.mean(vals):.4f} | {np.std(vals):.4f} |"
    )
metrics_table = "\n".join(metric_rows)

# ── 6. Write markdown ─────────────────────────────────────────────────
md_path = os.path.join(OUTPUT_DIR, "proposed_model_edge_weights_gatv2.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write(f"""# Proposed Model — GATv2 + Learnable Edge Weights S

## Config

| Parameter | Value |
|---|---|
| Model | GATv2Conv (3 layers, hidden=128, out=64) |
| lambda_recon | {lambda_recon} |
| lambda_contrast | {lambda_contrast} |
| lambda_edge | {lambda_edge} |
| lambda_cos | {lambda_cos} |
| tau (hinge) | 0.5 |
| lr_model | 1e-3 (wd=1e-5) |
| lr_S | 5e-3 |
| batch_size | 256 |
| max_epochs | 400 |
| early_stopping | patience=50, tol=5e-4 |
| Seeds | {SEEDS} |

---

## Training Loss Curves (Seed = 2)

![Loss curves](loss_curves_gatv2_seed2.png)

---

## Epoch Log — Seed 2 (all epochs)

{epoch_table}

---

## Learned Edge Weights — Seed 2

### S_final (epoch cuối)

{fmt_edge_table(SEED2['S_final'], 'S_final')}

```
{fmt_matrix(SEED2['S_final'])}
```

### S_infer (mean across epochs)

{fmt_edge_table(SEED2['S_infer'], 'S_infer')}

```
{fmt_matrix(SEED2['S_infer'])}
```

### Threshold (từ normal train set)

| mean | std | threshold (μ + 2σ) |
|---|---|---|
| {SEED2['n_mean']:.4f} | {SEED2['n_std']:.4f} | {SEED2['thresh']:.4f} |

---

## Overall Metrics (3 seeds)

{metrics_table}

---

## Per-Attack Recall (3 seeds)

{per_attack_table}
""")

print(f"Markdown saved → {md_path}")
