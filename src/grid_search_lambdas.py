# -*- coding: utf-8 -*-
"""
Grid search for lambda hyperparameters in GATv2 model.
Tunes: lambda_recon, lambda_contrast, lambda_cos, lambda_edge
Architecture and data pipeline identical to proposed_model_edge_weights_v2.py.

Results saved to: ../output/grid_search_results.csv
"""

import os
import csv
import itertools
import time
import sys
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
# GRID SEARCH SPACE
# =====================================================================
GRID = {
    "lambda_recon":    [1.0, 3.0, 5.0],
    "lambda_contrast": [0.1, 0.3, 0.5],
    "lambda_cos":      [0.1, 0.3, 0.5],
    "lambda_edge":     [5.0, 8.0, 15.0],
}

# Fixed hyperparameters
MAX_EPOCHS      = 150       # reduced from 400 for speed; early stopping compensates
EARLY_PATIENCE  = 25
EARLY_TOL       = 5e-4
BATCH_SIZE      = 256
LR_MODEL        = 1e-3
LR_S            = 5e-3
WEIGHT_DECAY    = 1e-5
HIDDEN_DIM      = 128
OUT_DIM         = 64
IMPORTANT_EDGES = [(1, 2), (0, 1), (1, 3), (1, 4)]
EDGE_LIST       = [(0,1),(1,0),(1,2),(2,1),(1,3),(3,1),(1,4),(4,1)]

OUTPUT_PATH = os.path.join(os.path.dirname(__file__),
                           "../output/grid_search_results.csv")

# =====================================================================
# DEVICE
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =====================================================================
# EDGE MAP (built once)
# =====================================================================
_edge_map = torch.full((5, 5), -1, dtype=torch.long)
for _idx, (_s, _d) in enumerate(EDGE_LIST):
    _edge_map[_s, _d] = _idx

# =====================================================================
# DATA LOADING & PREPROCESSING (done once, shared across all runs)
# =====================================================================
DATA_PATH = os.path.join(os.path.dirname(__file__),
                         "../related_data/dataset/base_dataset.csv")

df = pd.read_csv(DATA_PATH)
if "tcp?ActiveOpens" in df.columns:
    df["tcpActiveOpens"] = df["tcp?ActiveOpens"]

df_normal_all = df[df["class"] == "normal"].reset_index(drop=True)
df_attacks    = df[df["class"] != "normal"].reset_index(drop=True)
attack_types  = df_attacks["class"].unique()

df_normal_all   = df_normal_all.sample(frac=1, random_state=42).reset_index(drop=True)
df_normal_train = df_normal_all.iloc[:500].reset_index(drop=True)
df_normal_test  = df_normal_all.iloc[500:600].reset_index(drop=True)

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
            sc = StandardScaler()
            scaled = sc.fit_transform(raw)
            scalers[group] = sc
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

edge_index = torch.tensor(EDGE_LIST).t().long()

data_list_normal = [
    Data(x=x_normal[t], edge_index=edge_index.clone())
    for t in range(N_normal)
]

print(f"Train: {N_normal} | Test normal: {len(df_normal_test)} | "
      f"Attacks: {list(attack_types)}")
print(f"in_channels={in_channels} | max_dim={max_dim}\n")

# =====================================================================
# HELPERS
# =====================================================================
def s_to_edge_attr(S_vals, ei):
    src_local = ei[0] % 5
    dst_local = ei[1] % 5
    idx = _edge_map.to(S_vals.device)[src_local, dst_local]
    return S_vals[idx].unsqueeze(-1)


def s_to_matrix(S_vals):
    mat = torch.zeros(5, 5, device=S_vals.device)
    for i, (s, d) in enumerate(EDGE_LIST):
        mat[s, d] = S_vals[i]
    return mat


def nt_xent_loss(emb1, emb2, temperature=0.1):
    emb1   = F.normalize(emb1, dim=-1)
    emb2   = F.normalize(emb2, dim=-1)
    sim    = torch.mm(emb1, emb2.t()) / temperature
    labels = torch.arange(emb1.size(0), device=emb1.device)
    return F.cross_entropy(sim, labels)


# =====================================================================
# MODEL
# =====================================================================
class ContrastiveGATv2(nn.Module):
    def __init__(self, in_ch, hid, out_ch):
        super().__init__()
        self.gat1 = GATv2Conv(in_ch,   hid,    heads=4, concat=True,
                              dropout=0.2, edge_dim=1)
        self.gat2 = GATv2Conv(hid*4,   hid,    heads=4, concat=True,
                              dropout=0.2, edge_dim=1)
        self.gat3 = GATv2Conv(hid*4,   out_ch, heads=1, concat=False,
                              dropout=0.1, edge_dim=1)
        self.res  = (nn.Linear(in_ch, out_ch)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, data, edge_attr):
        x, ei = data.x, data.edge_index
        r   = self.res(x)
        x   = F.elu(self.gat1(x, ei, edge_attr=edge_attr))
        x   = F.elu(self.gat2(x, ei, edge_attr=edge_attr))
        emb = self.gat3(x, ei, edge_attr=edge_attr) + r
        B   = data.num_graphs
        emb        = emb.view(B, 5, -1)                          # (B,5,64)
        adj_logits = torch.bmm(emb, emb.transpose(1, 2))         # (B,5,5)
        alpha      = F.softmax(adj_logits, dim=-1)                # (B,5,5)
        return emb, adj_logits, alpha


# =====================================================================
# SINGLE TRAINING RUN
# =====================================================================
def train_and_eval(cfg: dict, run_id: int, total_runs: int) -> dict:
    lam_recon    = cfg["lambda_recon"]
    lam_contrast = cfg["lambda_contrast"]
    lam_cos      = cfg["lambda_cos"]
    lam_edge     = cfg["lambda_edge"]

    torch.manual_seed(0)

    # ── S_param ──────────────────────────────────────────────────────
    _s_init = torch.rand(8) * 0.6 + 0.2
    S_param = nn.Parameter(
        torch.log(_s_init / (1 - _s_init)).to(device)
    )

    # ── Model & optimizer ────────────────────────────────────────────
    model = ContrastiveGATv2(in_channels, hid=HIDDEN_DIM, out_ch=OUT_DIM).to(device)
    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": LR_MODEL, "weight_decay": WEIGHT_DECAY},
        {"params": [S_param],          "lr": LR_S},
    ])
    scheduler  = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    recon_crit = nn.BCEWithLogitsLoss()
    loader     = DataLoader(data_list_normal, batch_size=BATCH_SIZE, shuffle=True)

    best_loss, counter = float("inf"), 0
    S_epoch_log = []

    # ── Training loop ────────────────────────────────────────────────
    for epoch in range(MAX_EPOCHS):
        model.train()
        tot_loss = 0.0
        n_batch  = 0

        for batch in loader:
            batch = batch.to(device)
            B     = batch.num_graphs
            optimizer.zero_grad()

            S  = torch.sigmoid(S_param)
            ea = s_to_edge_attr(S, batch.edge_index)

            emb1, adj_logits1, alpha1 = model(batch, ea)

            aug = batch.clone()
            aug.x = aug.x + 0.01 * torch.randn_like(aug.x)
            aug.edge_index, _ = dropout_edge(aug.edge_index, p=0.1)
            ea_aug = s_to_edge_attr(S, aug.edge_index)
            emb2, _, _ = model(aug, ea_aug)

            S_mat        = s_to_matrix(S.detach())
            adj_target_b = S_mat.unsqueeze(0).expand(B, -1, -1)

            recon_loss    = recon_crit(adj_logits1, adj_target_b)
            contrast_loss = nt_xent_loss(emb1.mean(dim=1), emb2.mean(dim=1))

            emb_n = F.normalize(emb1, dim=-1)
            cos   = torch.bmm(emb_n, emb_n.transpose(1, 2))
            cos   = cos.masked_fill(
                torch.eye(5, device=device).bool().unsqueeze(0), 0.0)
            cosine_penalty = cos.mean()

            edge_loss = torch.tensor(0.0, device=device)
            for i, j in IMPORTANT_EDGES:
                edge_loss = edge_loss + F.relu(0.5 - alpha1[:, i, j].mean()) * lam_edge

            loss = (lam_recon    * recon_loss
                  + lam_contrast * contrast_loss
                  + lam_cos      * cosine_penalty
                  + edge_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_([S_param], 1.0)
            optimizer.step()

            tot_loss += loss.item()
            n_batch  += 1

        avg_loss = tot_loss / n_batch
        scheduler.step(avg_loss)
        S_epoch_log.append(torch.sigmoid(S_param.detach()).cpu())

        if avg_loss < best_loss - EARLY_TOL:
            best_loss = avg_loss
            counter   = 0
        else:
            counter += 1
            if counter >= EARLY_PATIENCE:
                break

        # ── Epoch progress bar ───────────────────────────────────────
        pct      = (epoch + 1) / MAX_EPOCHS
        bar_len  = 20
        filled   = int(bar_len * pct)
        bar      = "█" * filled + "░" * (bar_len - filled)
        sys.stdout.write(
            f"\r  epoch [{bar}] {epoch+1:>3}/{MAX_EPOCHS} "
            f"loss={avg_loss:.4f}  best={best_loss:.4f}  patience={counter}/{EARLY_PATIENCE}"
        )
        sys.stdout.flush()

    # ── S_infer ──────────────────────────────────────────────────────
    S_infer     = torch.stack(S_epoch_log).mean(dim=0).to(device)
    S_mat_infer = s_to_matrix(S_infer)

    # ── Threshold from normal train set ──────────────────────────────
    model.eval()
    normal_scores = []
    with torch.no_grad():
        for db in DataLoader(data_list_normal, batch_size=1, shuffle=False):
            db  = db.to(device)
            ea_ = s_to_edge_attr(S_infer, db.edge_index)
            _, al, alph = model(db, ea_)
            recon = F.binary_cross_entropy_with_logits(
                al.squeeze(0), S_mat_infer).item()
            edge  = sum(
                max(0., 0.5 - alph.squeeze(0)[i, j].item()) * lam_edge
                for i, j in IMPORTANT_EDGES
            )
            normal_scores.append(recon * 5 + edge)

    n_mean = np.mean(normal_scores)
    n_std  = np.std(normal_scores)
    thresh = n_mean + 2 * n_std

    # ── Eval helper ──────────────────────────────────────────────────
    def eval_df(df_set):
        N = len(df_set)
        if "tcp?ActiveOpens" in df_set.columns:
            df_set = df_set.copy()
            df_set["tcpActiveOpens"] = df_set["tcp?ActiveOpens"]
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
                ea_ = s_to_edge_attr(S_infer, db.edge_index)
                _, al, alph = model(db, ea_)
                recon = F.binary_cross_entropy_with_logits(
                    al.squeeze(0), S_mat_infer).item()
                edge  = sum(
                    max(0., 0.5 - alph.squeeze(0)[i, j].item()) * lam_edge
                    for i, j in IMPORTANT_EDGES
                )
                scores.append(recon * 5 + edge)
        return np.array(scores)

    # ── Attack evaluation ────────────────────────────────────────────
    total_TP = total_attack = 0
    per_attack_recall = {}
    for at in attack_types:
        df_at  = df_attacks[df_attacks["class"] == at].reset_index(drop=True)
        scores = eval_df(df_at)
        nd     = int((scores > thresh).sum())
        total_TP     += nd
        total_attack += len(scores)
        per_attack_recall[at] = nd / len(scores)

    ntest_scores = eval_df(df_normal_test)
    FP        = int((ntest_scores > thresh).sum())
    TN        = len(ntest_scores) - FP
    FN        = total_attack - total_TP
    recall    = total_TP / (total_attack + 1e-8)
    precision = total_TP / (total_TP + FP + 1e-8)
    accuracy  = (total_TP + TN) / (total_attack + len(ntest_scores))
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    fpr       = FP / len(ntest_scores)

    sys.stdout.write("\r" + " " * 80 + "\r")   # clear epoch bar
    sys.stdout.flush()

    result = {
        **cfg,
        "recall":    round(recall, 4),
        "precision": round(precision, 4),
        "f1":        round(f1, 4),
        "accuracy":  round(accuracy, 4),
        "fpr":       round(fpr, 4),
        "thresh":    round(thresh, 4),
        "epochs_run": epoch + 1,
        **{f"recall_{at}": round(v, 4) for at, v in per_attack_recall.items()},
    }

    print(f"[{run_id:>3}/{total_runs}] "
          f"recon={lam_recon} contrast={lam_contrast} cos={lam_cos} edge={lam_edge} | "
          f"Recall={recall:.4f}  F1={f1:.4f}  FPR={fpr:.4f}  epochs={epoch+1}")

    return result


# =====================================================================
# GRID SEARCH
# =====================================================================
keys   = list(GRID.keys())
combos = list(itertools.product(*[GRID[k] for k in keys]))
total  = len(combos)
print(f"Grid search: {total} combinations\n"
      f"  lambda_recon    : {GRID['lambda_recon']}\n"
      f"  lambda_contrast : {GRID['lambda_contrast']}\n"
      f"  lambda_cos      : {GRID['lambda_cos']}\n"
      f"  lambda_edge     : {GRID['lambda_edge']}\n")

results   = []
t_start   = time.time()

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

for run_id, combo in enumerate(combos, start=1):
    cfg = dict(zip(keys, combo))

    # ── Overall grid progress bar ─────────────────────────────────────
    g_pct    = (run_id - 1) / total
    g_filled = int(30 * g_pct)
    g_bar    = "█" * g_filled + "░" * (30 - g_filled)
    elapsed_so_far = time.time() - t_start
    eta = (elapsed_so_far / max(run_id - 1, 1)) * (total - run_id + 1)
    print(f"\n[Grid {g_bar}] {run_id}/{total}  "
          f"elapsed={elapsed_so_far/60:.1f}m  ETA={eta/60:.1f}m")
    print(f"  → recon={cfg['lambda_recon']}  contrast={cfg['lambda_contrast']}  "
          f"cos={cfg['lambda_cos']}  edge={cfg['lambda_edge']}")

    res = train_and_eval(cfg, run_id, total)
    results.append(res)

    # ── Save incrementally after each run ────────────────────────────
    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_PATH, index=False)

elapsed = time.time() - t_start
print(f"\nTotal time: {elapsed/60:.1f} min")

# =====================================================================
# SUMMARY — TOP 10 by F1
# =====================================================================
df_res = pd.DataFrame(results).sort_values("f1", ascending=False)

print("\n" + "="*90)
print("TOP 10 CONFIGS BY F1")
print("="*90)
cols_show = ["lambda_recon","lambda_contrast","lambda_cos","lambda_edge",
             "recall","precision","f1","accuracy","fpr"]
print(df_res[cols_show].head(10).to_string(index=False))
print("="*90)

best = df_res.iloc[0]
print(f"\nBest config:")
print(f"  lambda_recon    = {best['lambda_recon']}")
print(f"  lambda_contrast = {best['lambda_contrast']}")
print(f"  lambda_cos      = {best['lambda_cos']}")
print(f"  lambda_edge     = {best['lambda_edge']}")
print(f"  Recall={best['recall']}  Precision={best['precision']}  "
      f"F1={best['f1']}  FPR={best['fpr']}")
print(f"\nResults saved to: {OUTPUT_PATH}")
