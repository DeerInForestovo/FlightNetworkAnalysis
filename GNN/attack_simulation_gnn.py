# attack_simulation_gnn.py
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from torch_geometric.nn import SAGEConv
from scipy.stats import spearmanr

# If you saved visualize_network earlier in GNN/visualize_attack.py, we'll import it.
# Otherwise you can paste the visualize_network function into this file.
try:
    from visualize_attack import visualize_network
except Exception:
    def visualize_network(G, out_path, title="Global Airline Network"):
        import plotly.graph_objects as go
        gg = G.copy()
        if not nx.is_connected(gg):
            largest_cc = max(nx.connected_components(gg), key=len)
            gg = gg.subgraph(largest_cc)
        lats = [gg.nodes[n]["lat"] for n in gg.nodes()]
        lons = [gg.nodes[n]["lon"] for n in gg.nodes()]
        names = [gg.nodes[n]["name"] for n in gg.nodes()]
        edge_lats = []
        edge_lons = []
        for u, v in gg.edges():
            edge_lats += [gg.nodes[u]["lat"], gg.nodes[v]["lat"], None]
            edge_lons += [gg.nodes[u]["lon"], gg.nodes[v]["lon"], None]
        fig = go.Figure()
        fig.add_trace(go.Scattergeo(
            lon=edge_lons, lat=edge_lats, mode="lines",
            line=dict(width=0.5, color="rgba(100,100,100,0.3)"), hoverinfo="none"))
        fig.add_trace(go.Scattergeo(
            lon=lons, lat=lats, text=names, mode="markers",
            marker=dict(size=3, color="red", opacity=0.6)))
        fig.update_layout(
            title=title, showlegend=False,
            geo=dict(projection_type="natural earth", showland=True,
                     landcolor="rgb(230,230,230)", countrycolor="rgb(200,200,200)",
                     coastlinecolor="rgb(150,150,150)"))
        fig.write_html(out_path)

# ------------ Config ------------
GRAPH_PATH = "./output/global_airline_network.gml"
MODEL_PATH = "./GNN/model/graphsage.pt"
FEATURE_MEAN = "./GNN/model/feature_mean.pt"
FEATURE_STD = "./GNN/model/feature_std.pt"

OUT_DIR = "./output"
os.makedirs(OUT_DIR, exist_ok=True)

# Attack parameters (tweakable)
MODE = "random"     # options: "random", "static_bc", "gnn_static", "gnn_adaptive", "hub"
STEP_RATIO = 0.001        # fraction nodes removed each step
STEPS = 50                # remove 0.1% each step for 50 steps -> total 5%
VIS_STEPS = [0, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # which steps produce HTML visualizations
COMPUTE_GT = False        # whether to recompute ground-truth betweenness each step (expensive)
TOPK = 50                 # used for reporting overlap if COMPUTE_GT True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------ Model definition (must match training) ------------
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

# ------------ Helpers ------------
def build_edge_index_and_features(nxG):
    """Given a networkx graph with node attributes lat/lon/etc,
       return edge_index (torch.LongTensor [2, E]) and features tensor [N, F]."""
    # create a stable node ordering and mapping
    nodes = list(nxG.nodes())
    mapping = {n: i for i, n in enumerate(nodes)}

    # build edge lists (undirected -> add both directions)
    # pre-allocate list size estimate to avoid repeated resizing (2*edges)
    m = nxG.number_of_edges()
    if m == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        src = [0] * (2 * m)
        dst = [0] * (2 * m)
        idx = 0
        for u, v in nxG.edges():
            ui = mapping[u]
            vi = mapping[v]
            src[idx] = ui
            dst[idx] = vi
            idx += 1
            src[idx] = vi
            dst[idx] = ui
            idx += 1
        edge_index = torch.tensor([src, dst], dtype=torch.long)

    # features: compute expensive dicts once
    degree_seq = dict(nxG.degree())
    core_dict = nx.core_number(nxG)
    # use fromiter which is faster than list comprehensions for large N
    degrees = np.fromiter((degree_seq[n] for n in nodes), dtype=float, count=len(nodes))
    clustering = np.fromiter((nx.clustering(nxG, n) for n in nodes), dtype=float, count=len(nodes))
    core = np.fromiter((core_dict[n] for n in nodes), dtype=float, count=len(nodes))
    lat = np.fromiter((nxG.nodes[n].get('lat', 0.0) for n in nodes), dtype=float, count=len(nodes))
    lon = np.fromiter((nxG.nodes[n].get('lon', 0.0) for n in nodes), dtype=float, count=len(nodes))

    # normalize lat/lon (local zero-mean unit-std), will be globally standardized later
    if len(lat) > 0:
        lat = (lat - lat.mean()) / (lat.std() + 1e-9)
        lon = (lon - lon.mean()) / (lon.std() + 1e-9)

    feats = np.vstack([degrees, clustering, core, lat, lon]).T
    x = torch.from_numpy(feats.astype(np.float32))
    return mapping, edge_index, x

def predict_with_gnn_on_graph(nxG, model, mean, std):
    mapping, edge_index, x = build_edge_index_and_features(nxG)
    if edge_index.numel() == 0:
        return {n: 0.0 for n in nxG.nodes()}  # trivial empty
    # normalize with saved mean/std (broadcast)
    mean = mean.to(x.device)
    std = std.to(x.device)
    x = (x - mean) / (std + 1e-6)
    edge_index = edge_index.to(x.device)
    model.eval()
    with torch.no_grad():
        out = model(x.to(DEVICE), edge_index.to(DEVICE)).cpu().numpy().flatten()
        # model outputs log-scale (in your training you used log on labels)
        preds = np.exp(out)
    # map back to original node labels
    node_list = list(nxG.nodes())
    scores = {node_list[i]: float(preds[i]) for i in range(len(node_list))}
    return scores

# ------------ Main ------------
def main():
    print("Loading graph...")
    G_orig = nx.read_gml(GRAPH_PATH)
    # operate on LCC for baseline
    largest_cc = max(nx.connected_components(G_orig), key=len)
    G_orig = G_orig.subgraph(largest_cc).copy()
    print(f"Original (LCC) nodes: {G_orig.number_of_nodes()}, edges: {G_orig.number_of_edges()}")

    # load mean/std and model
    mean = torch.load(FEATURE_MEAN)
    std = torch.load(FEATURE_STD)
    # We assume feature length == mean length
    input_dim = mean.shape[0]
    model = GraphSAGE(input_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("Loaded GNN model.")

    # Precompute static scores if needed
    if MODE == "static_bc":
        print("Computing static betweenness on original graph (this may take time)...")
        static_bc = nx.betweenness_centrality(G_orig, normalized=True)
    elif MODE == "gnn_static":
        print("Computing GNN static predictions on original graph...")
        static_gnn_scores = predict_with_gnn_on_graph(G_orig, model, mean, std)

    # Prepare working copy
    G = G_orig.copy()

    lcc_history = []
    gt_history = []    # optional ground-truth aggregate (e.g., Spearman or topk overlap)
    step = 0
    total_nodes_init = G.number_of_nodes()

    for step in range(STEPS):
        n_remove = max(1, int(len(G.nodes()) * STEP_RATIO))
        print(f"\n--- Step {step} | Nodes remaining: {len(G.nodes())} | Removing: {n_remove} nodes ---")

        # choose nodes to remove according to MODE
        if MODE == "random":
            remove_nodes = random.sample(list(G.nodes()), n_remove)

        elif MODE == "hub":
            deg = dict(G.degree())
            sorted_nodes = sorted(deg, key=deg.get, reverse=True)
            remove_nodes = sorted_nodes[:n_remove]

        elif MODE == "static_bc":
            # static_bc computed on original graph; remove top among remaining nodes
            # sort by static_bc value descending, pick top n_remove that still exist
            candidates = sorted(static_bc.items(), key=lambda kv: kv[1], reverse=True)
            remove_nodes = []
            for node, _ in candidates:
                if node in G and len(remove_nodes) < n_remove:
                    remove_nodes.append(node)

        elif MODE == "gnn_static":
            # static GNN predictions on original graph; pick top among remaining nodes
            candidates = sorted(static_gnn_scores.items(), key=lambda kv: kv[1], reverse=True)
            remove_nodes = []
            for node, _ in candidates:
                if node in G and len(remove_nodes) < n_remove:
                    remove_nodes.append(node)

        elif MODE == "gnn_adaptive":
            # recompute features on the current graph, run model, pick top nodes
            gnn_scores = predict_with_gnn_on_graph(G, model, mean, std)
            candidates = sorted(gnn_scores.items(), key=lambda kv: kv[1], reverse=True)
            remove_nodes = [n for n, _ in candidates[:n_remove]]

        else:
            raise ValueError("Unknown MODE: " + MODE)

        print("Removing nodes (example):", remove_nodes[:5])
        G.remove_nodes_from(remove_nodes)

        if G.number_of_nodes() == 0:
            print("Graph empty, stopping.")
            break

        # measure LCC
        largest_cc = max(nx.connected_components(G), key=len)
        lcc_size = len(largest_cc)
        lcc_history.append(lcc_size)
        print(f"LCC size after step {step}: {lcc_size} ({lcc_size/total_nodes_init:.3f} of init)")

        # Optionally compute ground-truth betweenness & evaluation (costly)
        if COMPUTE_GT:
            bet = nx.betweenness_centrality(G, normalized=True)
            # If you want Spearman between GNN and GT at each step, compute:
            if MODE.startswith("gnn"):
                gnn_scores_current = predict_with_gnn_on_graph(G, model, mean, std)
                # align vectors
                nodes_list = list(G.nodes())
                pred_vec = np.array([gnn_scores_current[n] for n in nodes_list])
                true_vec = np.array([bet[n] for n in nodes_list])
                rho, _ = spearmanr(pred_vec, true_vec)
                gt_history.append(rho)
                print(f"Spearman (GNN vs GT) at step {step}: {rho:.4f}")
            else:
                # compute Top-K overlap between static strategy and true top-K for reporting
                nodes_list = list(G.nodes())
                true_vec = np.array([bet[n] for n in nodes_list])
                # get strategy vector
                if MODE == "static_bc":
                    pred_vec = np.array([static_bc.get(n,0.0) for n in nodes_list])
                elif MODE == "gnn_static":
                    pred_vec = np.array([static_gnn_scores.get(n,0.0) for n in nodes_list])
                elif MODE == "random":
                    pred_vec = np.random.rand(len(nodes_list))
                elif MODE == "hub":
                    deg = np.array([G.degree(n) for n in nodes_list], dtype=float)
                    pred_vec = deg
                rho, _ = spearmanr(pred_vec, true_vec)
                gt_history.append(rho)
                print(f"Spearman (policy vs GT) at step {step}: {rho:.4f}")

        # optionally visualize
        if step in VIS_STEPS:
            outpath = os.path.join(OUT_DIR, f"{MODE}_step_{step}.html")
            title = f"{MODE} attack - step {step}"
            try:
                visualize_network(G, outpath, title)
                print("Saved visualization to", outpath)
            except Exception as e:
                print("Visualization failed:", e)

    # Save results
    np.save(os.path.join(OUT_DIR, f"attack_{MODE}_lcc.npy"), np.array(lcc_history))
    if COMPUTE_GT:
        np.save(os.path.join(OUT_DIR, f"attack_{MODE}_gt.npy"), np.array(gt_history))
    print("\nDone. LCC history saved to:", os.path.join(OUT_DIR, f"attack_{MODE}_lcc.npy"))

if __name__ == "__main__":
    main()
