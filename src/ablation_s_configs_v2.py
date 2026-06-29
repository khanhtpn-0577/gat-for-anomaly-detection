# -*- coding: utf-8 -*-
"""
Ablation study (GATv2): effect of different fixed S configurations.
Same as ablation_s_configs.py but uses GATv2Conv instead of GATConv.
Architecture is identical to proposed_model_edge_weights.py, but S is FIXED
(not learned) for each config. Only model weights θ are trained.

Configs tested:
  base_original  — raw hardcoded values from base_proposed_model.py (> 1, BCE invalid range)
  base_normalized— base_original / 2.0  → fits [0,1]
  uniform_0.3    — all edges = 0.3
  uniform_0.5    — all edges = 0.5
  uniform_0.8    — all edges = 0.8
  all_ones       — all edges = 1.0 (binary)
  learned_final  — final sigmoid(S_param) from proposed_model_edge_weights.py run 1
  learned_mean   — S_infer (mean across epochs) from run 1
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

EDGE_LIST = [(0,1),(1,0),(1,2),(2,1),(1,3),(3,1),(1,4),(4,1)]
_edge_map  = torch.full((5, 5), -1, dtype=torch.long)
for _idx, (_s, _d) in enumerate(EDGE_LIST):
    _edge_map[_s, _d] = _idx

# =====================================================================
# S CONFIGURATIONS
# edge order: (0→1),(1→0),(1→2),(2→1),(1→3),(3→1),(1→4),(4→1)
# =====================================================================
CONFIGS = {
    "base_original":   [1.2, 1.2, 2.0, 2.0, 1.2, 1.2, 1.2, 1.2],  # base_proposed_model.py (> 1)
    "base_normalized": [0.6, 0.6, 1.0, 1.0, 0.6, 0.6, 0.6, 0.6],  # base_original / 2.0
    "uniform_0.3":     [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
    "uniform_0.5":     [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    "uniform_0.8":     [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
    "all_ones":        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "learned_final":   [0.2412, 0.7944, 0.4680, 0.3629, 0.8111, 0.5356, 0.5341, 0.8471],
    "learned_mean":    [0.3494, 0.7107, 0.3138, 0.3200, 0.6288, 0.5882, 0.5365, 0.7928],
}

# =====================================================================
# DATA LOADING
# =====================================================================
df = pd.read_csv(DATA_PATH)
if "tcp?ActiveOpens" in df.columns:
    df["tcpActiveOpens"] = df["tcp?ActiveOpens"]

df_normal_all = df[df["class"] == "normal"].reset_index(drop=True)
df_attacks    = df[df["class"] != "normal"].reset_index(drop=True)
attack_types  = df_attacks["class"].unique()

df_normal_all   = df_normal_all.sample(frac=1, random_state=42).reset_index(drop=True)
df_normal_train = df_normal_all.iloc[:500].reset_index(drop=True)
df_normal_test  = df_normal_all.iloc[500:600].reset_index(drop=True)

print(f"Train: {len(df_normal_train)} | Test: {len(df_normal_test)} | Attacks: {list(attack_types)}")

# =====================================================================
# FEATURES
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


def create_features(df_set, N, scalers=None, fit=False):
    x_padded = torch.zeros(N, 5, max_dim)
    if fit:
        scalers = {}
    for i, group in enumerate(node_order):
        cols  = [c for c in group_cols[group] if c in df_set.columns]
        raw   = df_set[cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(float)
        if fit:
            sc = StandardScaler(); scaled = sc.fit_transform(raw); scalers[group] = sc
        else:
            scaled = scalers[group].transform(raw)
        feats = torch.tensor(scaled[:N], dtype=torch.float)
        x_padded[:, i, :feats.shape[1]] = feats
    return x_padded, scalers


N_normal = len(df_normal_train)
x_normal, scalers_normal = create_features(df_normal_train, N_normal, fit=True)

time_emb = torch.zeros(N_normal, 5, 1)
for t in range(N_normal):
    time_emb[t] = torch.sin(torch.tensor(t / N_normal * 2 * np.pi).float())
x_normal    = torch.cat([x_normal, time_emb], dim=-1)
in_channels = x_normal.shape[-1]

edge_index = torch.tensor(
    [[0,1],[1,0],[1,2],[2,1],[1,3],[3,1],[1,4],[4,1]]
).t().long()

data_list_normal = [
    Data(x=x_normal[t], edge_index=edge_index.clone())
    for t in range(N_normal)
]

# =====================================================================
# HELPERS
# =====================================================================
def s_to_edge_attr(S_vals: torch.Tensor, ei: torch.Tensor) -> torch.Tensor:
    src_local = ei[0] % 5
    dst_local = ei[1] % 5
    idx = _edge_map.to(S_vals.device)[src_local, dst_local]
    return S_vals[idx].unsqueeze(-1)


def s_to_matrix(S_vals: torch.Tensor) -> torch.Tensor:
    mat = torch.zeros(5, 5, device=S_vals.device)
    for i, (s, d) in enumerate(EDGE_LIST):
        mat[s, d] = S_vals[i]
    return mat


# =====================================================================
# MODEL
# =====================================================================
def nt_xent_loss(emb1, emb2, temperature=0.1):
    emb1   = F.normalize(emb1, dim=-1)
    emb2   = F.normalize(emb2, dim=-1)
    sim    = torch.mm(emb1, emb2.t()) / temperature
    labels = torch.arange(emb1.size(0), device=emb1.device)
    return F.cross_entropy(sim, labels)


class ContrastiveGAT(nn.Module):
    def __init__(self, in_ch, hid, out_ch):
        super().__init__()
        self.gat1 = GATv2Conv(in_ch,   hid,    heads=4, concat=True,  dropout=0.2, edge_dim=1)
        self.gat2 = GATv2Conv(hid*4,   hid,    heads=4, concat=True,  dropout=0.2, edge_dim=1)
        self.gat3 = GATv2Conv(hid*4,   out_ch, heads=1, concat=False, dropout=0.1, edge_dim=1)
        self.res  = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, data, edge_attr):
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
# TRAIN + EVAL WITH FIXED S
# =====================================================================
def train_and_eval(config_name: str, s_vals: list) -> dict:
    print(f"\n{'='*60}")
    print(f"  Config: {config_name}")
    print(f"  S = {[round(v,4) for v in s_vals]}")
    print(f"{'='*60}")

    torch.manual_seed(42)

    S_fixed   = torch.tensor(s_vals, dtype=torch.float).to(device)  # fixed, no grad
    S_mat     = s_to_matrix(S_fixed)

    model     = ContrastiveGAT(in_channels, hid=128, out_ch=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    recon_crit= nn.BCEWithLogitsLoss()
    loader    = DataLoader(data_list_normal, batch_size=256, shuffle=True)

    best_loss, counter = float("inf"), 0

    for epoch in range(400):
        model.train()
        tot = dict(loss=0., recon=0., contrast=0., edge=0., cos=0.)
        n_batch = 0

        for batch in loader:
            batch = batch.to(device)
            B     = batch.num_graphs
            optimizer.zero_grad()

            ea = s_to_edge_attr(S_fixed, batch.edge_index)
            emb1, adj_logits1, alpha1 = model(batch, ea)

            aug = batch.clone()
            aug.x = aug.x + 0.01 * torch.randn_like(aug.x)
            aug.edge_index, _ = dropout_edge(aug.edge_index, p=0.1)
            ea_aug = s_to_edge_attr(S_fixed, aug.edge_index)
            emb2, _, _ = model(aug, ea_aug)

            adj_target_b  = S_mat.unsqueeze(0).expand(B, -1, -1)
            recon_loss    = recon_crit(adj_logits1, adj_target_b)
            contrast_loss = nt_xent_loss(emb1.mean(dim=1), emb2.mean(dim=1))

            emb_n = F.normalize(emb1, dim=-1)
            cos   = torch.bmm(emb_n, emb_n.transpose(1, 2))
            cos   = cos.masked_fill(
                torch.eye(5, device=device).bool().unsqueeze(0), 0.0)
            cosine_penalty = cos.mean()

            edge_loss = torch.tensor(0.0, device=device)
            for i, j in important_edges:
                edge_loss = edge_loss + F.relu(0.5 - alpha1[:, i, j].mean()) * lambda_edge

            loss = (lambda_recon    * recon_loss
                  + lambda_contrast * contrast_loss
                  + lambda_cos      * cosine_penalty
                  + edge_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            tot["loss"]     += loss.item()
            tot["recon"]    += recon_loss.item()
            tot["contrast"] += contrast_loss.item()
            tot["edge"]     += edge_loss.item()
            tot["cos"]      += cosine_penalty.item()
            n_batch         += 1

        avg_loss = tot["loss"] / n_batch
        scheduler.step(avg_loss)

        if epoch % 50 == 0:
            print(f"  Epoch {epoch:3d} | total={avg_loss:.4f}  "
                  f"recon={tot['recon']/n_batch:.4f}  "
                  f"edge={tot['edge']/n_batch:.4f}")

        if avg_loss < best_loss - 5e-4:
            best_loss = avg_loss; counter = 0
        else:
            counter += 1
            if counter >= 50:
                print(f"  Early stopping at epoch {epoch}")
                break

    # ── threshold from normal train ───────────────────────────────────
    model.eval()
    normal_scores = []
    with torch.no_grad():
        for db in DataLoader(data_list_normal, batch_size=1, shuffle=False):
            db  = db.to(device)
            ea_ = s_to_edge_attr(S_fixed, db.edge_index)
            _, al, alph = model(db, ea_)
            recon = F.binary_cross_entropy_with_logits(
                al.squeeze(0), S_mat).item()
            edge  = sum(max(0., 0.5 - alph.squeeze(0)[i,j].item()) * lambda_edge
                        for i, j in important_edges)
            normal_scores.append(recon * 5 + edge)

    n_mean = np.mean(normal_scores)
    n_std  = np.std(normal_scores)
    thresh = n_mean + 2 * n_std

    # ── attack evaluation ─────────────────────────────────────────────
    def eval_df(df_set):
        N = len(df_set)
        if "tcp?ActiveOpens" in df_set.columns:
            df_set = df_set.copy(); df_set["tcpActiveOpens"] = df_set["tcp?ActiveOpens"]
        x, _ = create_features(df_set, N, scalers=scalers_normal, fit=False)
        te = torch.zeros(N, 5, 1)
        for t in range(N):
            te[t] = torch.sin(torch.tensor(t / N * 2 * np.pi).float())
        x  = torch.cat([x, te], dim=-1)
        dl = [Data(x=x[t], edge_index=edge_index.clone()) for t in range(N)]
        scores = []
        with torch.no_grad():
            for db in DataLoader(dl, batch_size=1, shuffle=False):
                db  = db.to(device)
                ea_ = s_to_edge_attr(S_fixed, db.edge_index)
                _, al, alph = model(db, ea_)
                recon = F.binary_cross_entropy_with_logits(
                    al.squeeze(0), S_mat).item()
                edge  = sum(max(0., 0.5 - alph.squeeze(0)[i,j].item()) * lambda_edge
                            for i, j in important_edges)
                scores.append(recon * 5 + edge)
        return np.array(scores)

    total_TP = total_attack = 0
    per_attack = {}
    for at in attack_types:
        df_at  = df_attacks[df_attacks["class"] == at].reset_index(drop=True)
        scores = eval_df(df_at)
        nd     = int((scores > thresh).sum())
        total_TP     += nd
        total_attack += len(scores)
        per_attack[at] = {"recall": nd / len(scores), "detected": nd, "total": len(scores)}

    ntest  = eval_df(df_normal_test)
    FP     = int((ntest > thresh).sum())
    TN     = len(ntest) - FP
    FN     = total_attack - total_TP
    recall    = total_TP / (total_attack + 1e-8)
    precision = total_TP / (total_TP + FP + 1e-8)
    accuracy  = (total_TP + TN) / (total_attack + len(ntest))
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    fpr       = FP / len(ntest)

    metrics = {
        "TP": total_TP, "FN": FN, "FP": FP, "TN": TN,
        "recall": recall, "precision": precision,
        "accuracy": accuracy, "f1": f1, "fpr": fpr,
        "thresh": thresh, "per_attack": per_attack,
    }

    # ── print results ─────────────────────────────────────────────────
    print(f"\n  Threshold: {thresh:.4f}  (mean={n_mean:.4f}, std={n_std:.4f})")
    for at, res in per_attack.items():
        print(f"  {at:<12}  detected={res['detected']}/{res['total']}  recall={res['recall']:.4f}")
    print(f"\n  TP={total_TP}  FN={FN}  FP={FP}  TN={TN}")
    print(f"  Recall={recall:.4f}  Precision={precision:.4f}  "
          f"F1={f1:.4f}  FPR={fpr:.4f}")

    return metrics


# =====================================================================
# RUN ALL CONFIGS
# =====================================================================
all_results = {}
for name, s_vals in CONFIGS.items():
    all_results[name] = train_and_eval(name, s_vals)

# =====================================================================
# SUMMARY TABLE
# =====================================================================
print("\n\n" + "="*90)
print("ABLATION SUMMARY")
print("="*90)
header = f"{'Config':<20} {'Recall':>8} {'Precision':>10} {'F1':>8} {'FPR':>8} {'FP':>5} {'FN':>5} {'Thresh':>10}"
print(header)
print("-"*90)
for name, m in all_results.items():
    print(f"{name:<20} {m['recall']:>8.4f} {m['precision']:>10.4f} "
          f"{m['f1']:>8.4f} {m['fpr']:>8.4f} "
          f"{m['FP']:>5} {m['FN']:>5} {m['thresh']:>10.4f}")
print("="*90)

# Per-attack recall breakdown
print("\nPER-ATTACK RECALL BREAKDOWN")
print("-"*90)
attack_list = list(attack_types)
header2 = f"{'Config':<20}" + "".join(f"{at[:10]:>12}" for at in attack_list)
print(header2)
print("-"*90)
for name, m in all_results.items():
    row = f"{name:<20}"
    for at in attack_list:
        row += f"{m['per_attack'][at]['recall']:>12.4f}"
    print(row)
print("="*90)
