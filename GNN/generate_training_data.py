import os
import torch
import networkx as nx
import numpy as np
from torch_geometric.data import Data

DATA_DIR = "./GNN/data"
os.makedirs(DATA_DIR, exist_ok=True)

GRAPH_PATH = "./output/global_airline_network.gml"

print("Loading graph:", GRAPH_PATH)
G = nx.read_gml(GRAPH_PATH)

mapping = {node: idx for idx, node in enumerate(G.nodes())}
G = nx.relabel_nodes(G, mapping)

print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# -----------------------------
# 1. Build edge index for PyG
# -----------------------------
edge_index = []
for u, v in G.edges():
    edge_index.append([u, v])
    edge_index.append([v, u])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
print("Edge index shape:", edge_index.shape)

# -----------------------------
# 2. Node features
# -----------------------------
num_nodes = G.number_of_nodes()

degrees = np.array([G.degree(n) for n in G.nodes()])
clustering = np.array([nx.clustering(G, n) for n in G.nodes()])
core_number = np.array([nx.core_number(G)[n] for n in G.nodes()])

lat = np.array([G.nodes[n]['lat'] for n in G.nodes()], dtype=float)
lon = np.array([G.nodes[n]['lon'] for n in G.nodes()], dtype=float)

# Normalize lat/lon to [-1, 1]
lat = (lat - lat.mean()) / lat.std()
lon = (lon - lon.mean()) / lon.std()

# Features matrix
x = torch.tensor(
    np.vstack([degrees, clustering, core_number, lat, lon]).T,
    dtype=torch.float
)

print("Node features shape:", x.shape)

# -----------------------------
# 3. Labels (betweenness)
# -----------------------------
print("Computing betweenness centrality... (may take ~20–40 seconds)")
bet = nx.betweenness_centrality(G, normalized=True)
bet_arr = np.array([bet[n] for n in G.nodes()], dtype=float)

# Use log scale to stabilize range
y = torch.tensor(np.log1p(bet_arr), dtype=torch.float).unsqueeze(1)
print("Labels shape:", y.shape)

# -----------------------------
# 4. Save everything
# -----------------------------
torch.save(edge_index, f"{DATA_DIR}/edge_index.pt")
torch.save(x, f"{DATA_DIR}/features.pt")
torch.save(y, f"{DATA_DIR}/labels.pt")

print("\nTraining data generated in ./GNN/data/")
