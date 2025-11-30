"""
Train a Graph Neural Network (GraphSAGE) to predict betweenness in a flight network.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import SAGEConv
from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = "./GNN/data"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading data...")
edge_index = torch.load(f"{DATA_DIR}/edge_index.pt").to(device)
x = torch.load(f"{DATA_DIR}/features.pt").to(device)
y = torch.load(f"{DATA_DIR}/labels.pt").to(device)

# Save feature normalization statistics for inference-time consistency
import os
os.makedirs("./GNN/model", exist_ok=True)
torch.save(x.mean(dim=0).cpu(), "./GNN/model/feature_mean.pt")
torch.save(x.std(dim=0).cpu(), "./GNN/model/feature_std.pt")

x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)

y_log = torch.log(y + 1e-9) 

if y_log.dim() == 1:
    y_log = y_log.unsqueeze(1)

num_nodes = x.shape[0]
perm = torch.randperm(num_nodes)
split = int(0.8 * num_nodes)
train_idx = perm[:split]
test_idx = perm[split:]

print(f"Data ready. Nodes: {num_nodes}, Features: {x.shape[1]}")


def compute_spearman_and_topk(pred_log_tensor, y_tensor, y_log_tensor, idxs=None, K=50):
    """Compute Spearman rho between pred_log and y_log on idxs (or full) and Top-K overlap.
    pred_log_tensor: torch tensor shape [N,1] (log-scale predictions)
    y_tensor: original loaded y tensor (same as saved labels, e.g., log1p(bet))
    y_log_tensor: torch tensor representing the target used for training (y transformed again in script)
    idxs: torch.Tensor or list of indices to evaluate on (if None, evaluate on all)
    Returns: (rho, topk_frac) where topk_frac in [0,1]
    """
    # Select indices
    if idxs is None:
        sel_pred_log = pred_log_tensor.squeeze().cpu().numpy()
        sel_y_log = y_log_tensor.squeeze().cpu().numpy()
        sel_pred_final = np.exp(sel_pred_log)
        sel_y_true = y_tensor.squeeze().cpu().numpy()
    else:
        sel_idx = np.array(idxs.cpu()) if isinstance(idxs, torch.Tensor) else np.array(idxs)
        sel_pred_log = pred_log_tensor.squeeze().cpu().numpy()[sel_idx]
        sel_y_log = y_log_tensor.squeeze().cpu().numpy()[sel_idx]
        sel_pred_final = np.exp(sel_pred_log)
        sel_y_true = y_tensor.squeeze().cpu().numpy()[sel_idx]

    # Spearman on log-scale (pred_log vs y_log)
    try:
        rho, _ = spearmanr(sel_pred_log, sel_y_log)
        if np.isnan(rho):
            rho = 0.0
    except Exception:
        rho = 0.0

    # Top-K overlap (operate on 'final' scale which here corresponds to y_tensor and exp(pred_log))
    K_eff = min(K, len(sel_y_true))
    if K_eff <= 0:
        topk_frac = 0.0
    else:
        topk_true = set(np.argsort(sel_y_true)[-K_eff:])
        topk_pred = set(np.argsort(sel_pred_final)[-K_eff:])
        overlap = len(topk_true.intersection(topk_pred))
        topk_frac = overlap / float(K_eff)

    return float(rho), float(topk_frac)


def plot_and_save_history(loss_epochs, losses, rho_epochs, rhos, topks, out_path_png, out_path_npz=None):
    # use seaborn style if available, otherwise fall back to a safe default
    try:
        plt.style.use('seaborn-darkgrid')
    except Exception:
        try:
            plt.style.use('ggplot')
        except Exception:
            pass
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(loss_epochs, losses, color='tab:blue', label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    # plot rhos and topks at their own sampled epochs
    ax2.plot(rho_epochs, rhos, color='tab:green', marker='o', label='Val Spearman Rho')
    ax2.plot(rho_epochs, topks, color='tab:orange', marker='x', label='Val Top-K Overlap')
    ax2.set_ylabel('Metric (rho / top-k)', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    # fix metric y-range to [0.5, 1.0]
    try:
        ax2.set_ylim(0.5, 1.0)
    except Exception:
        pass

    # combined legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.title('Training history')
    fig.tight_layout()
    plt.savefig(out_path_png)
    plt.close(fig)

    if out_path_npz is not None:
        np.savez(out_path_npz, loss_epochs=np.array(loss_epochs), losses=np.array(losses), rho_epochs=np.array(rho_epochs), rhos=np.array(rhos), topks=np.array(topks))


class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x

model = GraphSAGE(x.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

print("\nStarting training...")
# History containers
history_epochs = []
history_losses = []
history_rhos = []
history_topks = []

LOG_EVERY = 100

# We'll record loss every epoch (starting from epoch 1), and record rho/topk at intervals LOG_EVERY.
loss_epochs = []
losses = []
rho_epochs = []
history_rhos = []
history_topks = []

# total epochs
TOTAL_EPOCHS = 10000

for epoch in range(1, TOTAL_EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    
    out = model(x, edge_index)
    loss = loss_fn(out[train_idx], y_log[train_idx])
    
    loss.backward()
    optimizer.step()
    # record loss for this epoch (exclude epoch 0)
    loss_epochs.append(epoch)
    losses.append(float(loss.item()))

    if epoch % LOG_EVERY == 0:
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(out[test_idx], y_log[test_idx]).item()
            # compute both spearman and top-k using helper
            rho, topk_frac = compute_spearman_and_topk(out.detach(), y, y_log, idxs=test_idx, K=50)
            print(f"Epoch {epoch:05d} | Loss: {loss.item():.6f} | Val Rho: {rho:.4f} | Val TopK: {topk_frac*100:.1f}%")

            # record history for metrics (sampled)
            rho_epochs.append(epoch)
            history_rhos.append(rho)
            history_topks.append(topk_frac)

model.eval()
with torch.no_grad():
    pred_log = model(x, edge_index).cpu()
    pred_final = torch.exp(pred_log).numpy().flatten()
    y_true = y.cpu().numpy().flatten()

pred_test = pred_final[test_idx.cpu()]
y_test = y_true[test_idx.cpu()]

rho, _ = spearmanr(pred_test, y_test)
print(f"\n=== Final Evaluation on Test Set ===")
print(f"Spearman Correlation: {rho:.4f}")

K = 50
topk_true = set(np.argsort(y_true)[-K:])
topk_pred = set(np.argsort(pred_final)[-K:])
overlap = len(topk_true.intersection(topk_pred))
print(f"Top-{K} Overlap: {overlap}/{K} ({overlap/K*100:.1f}%)")

import os
os.makedirs("./GNN/model", exist_ok=True)

torch.save(model.state_dict(), "./GNN/model/graphsage.pt")

model.eval()
with torch.no_grad():
    all_pred_log = model(x, edge_index).cpu()
    torch.save(all_pred_log, "./GNN/model/predictions.pt")

print("\nSaved.")

# Save and plot training history
out_png = "./GNN/model/train_history.png"
out_npz = "./GNN/model/train_history.npz"
if len(loss_epochs) > 0:
    plot_and_save_history(loss_epochs, losses, rho_epochs, history_rhos, history_topks, out_png, out_npz)
    print(f"Training history plot saved to {out_png} and arrays to {out_npz}")
else:
    print("No loss history recorded (this should not happen). Check training loop.")
