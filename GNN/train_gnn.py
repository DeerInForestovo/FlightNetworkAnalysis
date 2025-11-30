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
for epoch in range(10001):
    model.train()
    optimizer.zero_grad()
    
    out = model(x, edge_index)
    loss = loss_fn(out[train_idx], y_log[train_idx])
    
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(out[test_idx], y_log[test_idx]).item()
            pred_np = out[test_idx].cpu().numpy().flatten()
            true_np = y_log[test_idx].cpu().numpy().flatten()
            rho, _ = spearmanr(pred_np, true_np)
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val Rho: {rho:.4f}")

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
