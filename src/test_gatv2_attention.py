"""
test_gatv2_attention.py
=======================
Test công thức GAT vs GATv2: chứng minh tính Static vs Dynamic attention.

Setup đơn giản:
  - 1 graph, 5 nodes, features ngẫu nhiên (khác nhau mỗi lần chạy)
  - 1 GATConv layer, 1 head
  - 1 GATv2Conv layer, 1 head
  - Vẽ attention matrix A[i,j] = node i chú ý đến node j

Kỳ vọng:
  GAT  : mọi hàng (query node i) có cùng argmax column j → STATIC
  GATv2: các hàng có argmax khác nhau → DYNAMIC

Chạy:  python test_gatv2_attention.py
Output: output/test_gatv2/
"""

import os, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.data import Data

# ── PATHS ────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(os.path.dirname(SCRIPT_DIR), "output", "test_gatv2")
os.makedirs(OUT_DIR, exist_ok=True)

NODE_NAMES = ['Interface', 'IP', 'TCP', 'UDP', 'ICMP']
N_NODES    = 5
IN_CH      = 8   # feature dim

# ── 1. Graph ngẫu nhiên (khác nhau mỗi lần chạy) ─────────────
x = torch.randn(N_NODES, IN_CH)
edge_index = torch.tensor(
    [[i, j] for i in range(N_NODES) for j in range(N_NODES) if i != j],
    dtype=torch.long
).t()  # FC, 20 edges

graph = Data(x=x, edge_index=edge_index)
print(f"Graph: {N_NODES} nodes, {edge_index.shape[1]} edges, in_ch={IN_CH}")
print(f"Node features (random seed=time):\n{x.numpy().round(3)}\n")

# ── 2. Khởi tạo 1 layer GAT và 1 layer GATv2 (1 head) ────────
torch.manual_seed(0)
gat_layer   = GATConv(IN_CH, 16, heads=1, concat=False, add_self_loops=False)
torch.manual_seed(0)
gatv2_layer = GATv2Conv(IN_CH, 16, heads=1, concat=False, add_self_loops=False)

# ── 3. Extract attention matrix ───────────────────────────────
def get_attention_matrix(layer, graph, n=5):
    """
    Trả về A (5×5): A[i, j] = attention node i gán cho node j.
    (i = query, j = key/neighbor)
    """
    with torch.no_grad():
        _, (ei, alpha) = layer(graph.x, graph.edge_index,
                               return_attention_weights=True)
    # alpha: (|E|, 1) với heads=1
    alpha_flat = alpha.squeeze(-1)  # (20,)

    A = torch.zeros(n, n)
    for e in range(ei.shape[1]):
        dst = ei[1, e].item()   # node nhận (query)
        src = ei[0, e].item()   # node gửi (key)
        A[dst, src] = alpha_flat[e].item()
    return A.numpy()

A_gat   = get_attention_matrix(gat_layer,   graph)
A_gatv2 = get_attention_matrix(gatv2_layer, graph)

# ── 4. In kết quả ─────────────────────────────────────────────
def print_attention(A, title):
    print(f"── {title} ──")
    header = f"{'':12}" + "".join(f"{n:>10}" for n in NODE_NAMES)
    print(header)
    argmaxes = []
    for i in range(5):
        row_str = f"{NODE_NAMES[i]:12}"
        best_j, best_v = -1, -1.0
        for j in range(5):
            v = A[i, j]
            marker = " *" if j == np.argmax([A[i,k] for k in range(5) if k!=i]+[-1]) + (1 if np.argmax([A[i,k] for k in range(5) if k!=i]) >= i else 0) else "  "
            row_str += f"{v:8.4f}"
            if j != i and v > best_v:
                best_v, best_j = v, j
        argmaxes.append(best_j)
        print(f"{row_str}  → max: {NODE_NAMES[best_j]}")
    unique = len(set(argmaxes))
    verdict = "STATIC ✅ (tất cả query cùng argmax)" if unique==1 else f"DYNAMIC ({'✅' if unique>1 else '❌'}) ({unique} argmax khác nhau)"
    print(f"Argmax per query: {[NODE_NAMES[j] for j in argmaxes]}")
    print(f"→ {verdict}\n")
    return argmaxes

print("="*60)
print("Công thức GAT  : e(hi,hj) = LReLU( aᵀ[Whi ‖ Whj] )")
print("                          = LReLU( a1·Whi + a2·Whj )")
print("  → key score a2·Whj độc lập với query i → STATIC")
print()
print("Công thức GATv2: e(hi,hj) = aᵀ · LReLU( W·[hi ‖ hj] )")
print("  → hi và hj interact qua W trước nonlinearity → DYNAMIC")
print("="*60 + "\n")

am_gat   = print_attention(A_gat,   "GAT   (1 head) — Static attention (Theorem 1, ICLR 2022)")
am_gatv2 = print_attention(A_gatv2, "GATv2 (1 head) — Dynamic attention (Theorem 2, ICLR 2022)")

# ── 5. Vẽ heatmap so sánh ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, A, title, cmap, argmaxes in [
    (axes[0], A_gat,   'GAT — Static Attention (1 head)', 'Blues', am_gat),
    (axes[1], A_gatv2, 'GATv2 — Dynamic Attention (1 head)', 'Greens', am_gatv2),
]:
    masked = np.ma.masked_where(np.eye(5, dtype=bool), A)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=masked.max()+0.01, aspect='auto')

    ax.set_xticks(range(5)); ax.set_xticklabels(NODE_NAMES, rotation=30, fontsize=10)
    ax.set_yticks(range(5)); ax.set_yticklabels(NODE_NAMES, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('', fontsize=10)
    ax.set_ylabel('Query node i', fontsize=10)

    # Giá trị trong ô
    for i in range(5):
        for j in range(5):
            if i == j:
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, color='#dddddd'))
                ax.text(j, i, 'self', ha='center', va='center', fontsize=8, color='gray')
            else:
                color = 'white' if A[i,j] > masked.max()*0.6 else 'black'
                ax.text(j, i, f'{A[i,j]:.3f}', ha='center', va='center',
                        fontsize=9, color=color)

    # Đánh dấu argmax mỗi hàng bằng hình chữ nhật đỏ
    for i, best_j in enumerate(argmaxes):
        rect = plt.Rectangle((best_j-0.5, i-0.5), 1, 1,
                               fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

    plt.colorbar(im, ax=ax, shrink=0.85)

    # Chú thích kết luận
    unique = len(set(argmaxes))
    verdict = "" if unique > 1 else "STATIC: tất cả argmax = cùng 1 cột"
    color = '#1565c0'
    if verdict:
        ax.text(0.5, -0.18, verdict, transform=ax.transAxes,
                ha='center', fontsize=10, fontweight='bold', color=color)

plt.suptitle('GAT vs GATv2: Static vs Dynamic Attention\n'
             '(ô đỏ = argmax của từng query node, 1 head, 5-node FC graph)',
             fontsize=13, y=1.02)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'gat_vs_gatv2_1head.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")
