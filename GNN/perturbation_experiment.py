"""
Verify that the trained GNN model is robust to small perturbations in the graph structure.
"""

import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import random
from torch_geometric.nn import SAGEConv
from scipy.stats import spearmanr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GRAPH_PATH = "./output/global_airline_network.gml"
MODEL_PATH = "./GNN/model/graphsage.pt"

REMOVE_RATIO = 0.01   # 1% nodes removed
TOPK = 50
REMOVE_HUB = True

print("\nLoading original graph...")
G = nx.read_gml(GRAPH_PATH)

nodes = list(G.nodes())
num_remove = int(len(nodes) * REMOVE_RATIO)

if REMOVE_HUB:
    print(">> Removing HUB nodes (by degree)")
    degree_dict = dict(G.degree())
    sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)
    remove_nodes = sorted_nodes[:num_remove]
else:
    print(">> Removing RANDOM nodes")
    remove_nodes = random.sample(nodes, num_remove)

print(f"Removing {num_remove} nodes (2%) to simulate perturbation...")
G.remove_nodes_from(remove_nodes)

# Keep largest connected component
largest_cc = max(nx.connected_components(G), key=len)
G = G.subgraph(largest_cc).copy()

print(f"New graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Re-index nodes
mapping = {node: i for i, node in enumerate(G.nodes())}
G = nx.relabel_nodes(G, mapping)

# Build edge_index
edge_index = []
for u, v in G.edges():
    edge_index.append([u, v])
    edge_index.append([v, u])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)

# Build features
degrees = np.array([G.degree[n] for n in G.nodes()])
clustering = np.array([nx.clustering(G, n) for n in G.nodes()])
core = np.array([nx.core_number(G)[n] for n in G.nodes()])
lat = np.array([G.nodes[n]["lat"] for n in G.nodes()])
lon = np.array([G.nodes[n]["lon"] for n in G.nodes()])

lat = (lat - lat.mean()) / lat.std()
lon = (lon - lon.mean()) / lon.std()

X = torch.tensor(
    np.vstack([degrees, clustering, core, lat, lon]).T,
    dtype=torch.float
).to(device)

# Normalize
mean = torch.load("./GNN/model/feature_mean.pt").to(device)
std = torch.load("./GNN/model/feature_std.pt").to(device)
X = (X - mean) / (std + 1e-6)

print("\nComputing new betweenness (ground truth)...")
bet = nx.betweenness_centrality(G, normalized=True)
y_true = np.array([bet[n] for n in G.nodes()])

# Define model again
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
        return self.conv3(x, edge_index)

model = GraphSAGE(X.shape[1]).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

with torch.no_grad():
    pred_log = model(X, edge_index)
    pred = torch.exp(pred_log).cpu().numpy().flatten()

# ========================
# EVALUATION
# ========================
rho, _ = spearmanr(pred, y_true)
print(f"\nSpearman correlation (new graph): {rho:.4f}")

topk_true = set(np.argsort(y_true)[-TOPK:])
topk_pred = set(np.argsort(pred)[-TOPK:])
overlap = len(topk_true.intersection(topk_pred))

print(f"Top-{TOPK} overlap: {overlap}/{TOPK} ({overlap/TOPK*100:.1f}%)")

# Optional save
np.save("./GNN/model/pred_after_perturb.npy", pred)
