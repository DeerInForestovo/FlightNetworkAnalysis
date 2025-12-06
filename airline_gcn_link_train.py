import argparse
import math, os
from collections import defaultdict

import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
import torch.nn.functional as F
from torch import nn

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected


# ---------------------------------------------------
# 1. Read GML and keep only the largest connected component (drop isolated nodes)
# ---------------------------------------------------
def load_airline_graph(path: str) -> nx.Graph:
    # Read GML
    G = nx.read_gml(path)

    # If the original graph is directed / MultiGraph, convert to a simple undirected graph
    G = nx.Graph(G)

    # Keep only the largest connected component (automatically removes isolated nodes)
    cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(cc).copy()
    
    # Relabel nodes to 0..N-1, store original id in attribute "old_id"
    G = nx.convert_node_labels_to_integers(G, label_attribute="old_id")

    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ---------------------------------------------------
# 2. Build node features: xyz (from lat/lon), one-hot(country) + structural stats
# ---------------------------------------------------
def build_features(G: nx.Graph):
    """
    Per-node features include:
    - x, y, z (3D coordinates on the unit sphere converted from lat, lon)
    - country one-hot
    - degree
    - weighted_degree
    - clustering
    - core_number
    - international_routes
    - domestic_routes
    - total_mileage (sum(weight * distance))

    Continuous features are log-transformed / (optionally) normalized.
    """
    import numpy as np
    import networkx as nx

    num_nodes = G.number_of_nodes()

    # ==== 1) Country one-hot ====
    countries = sorted({data["country"] for _, data in G.nodes(data=True)})
    country_to_idx = {c: i for i, c in enumerate(countries)}
    num_countries = len(countries)

    # ==== 2) Graph structural statistics ====
    degrees = dict(G.degree())
    weighted_degrees = dict(G.degree(weight="weight"))
    clustering = nx.clustering(G)
    core_num = nx.core_number(G)

    international_count = {n: 0.0 for n in G.nodes()}
    domestic_count = {n: 0.0 for n in G.nodes()}
    total_mileage = {n: 0.0 for n in G.nodes()}

    for u, v, data in G.edges(data=True):
        cu = G.nodes[u].get("country")
        cv = G.nodes[v].get("country")

        w = float(data.get("weight", 1.0))        # In this undirected graph many routes have weight=2 (two directions)
        d = float(data.get("distance", 0.0))

        # Count domestic / international routes by edge count (not by weight)
        if cu is not None and cv is not None and cu == cv:
            domestic_count[u] += 1.0
            domestic_count[v] += 1.0
        else:
            international_count[u] += 1.0
            international_count[v] += 1.0

        # Total mileage: weight * distance, added to both endpoints
        miles = w * d
        total_mileage[u] += miles
        total_mileage[v] += miles

    # ==== 3) Build feature matrix ====
    # 3 (x,y,z) + num_countries (one-hot) + 7 structural features
    feat_dim = 3 + num_countries + 7
    X = np.zeros((num_nodes, feat_dim), dtype=np.float32)

    for node, data in G.nodes(data=True):
        idx = int(node)

        lat = float(data.get("lat", 0.0))   # degrees
        lon = float(data.get("lon", 0.0))   # degrees
        country = data["country"]
        c_idx = country_to_idx[country]

        # ---- lat/lon -> spherical coordinates (unit sphere x,y,z) ----
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)

        offset = 0
        # x, y, z
        X[idx, offset] = x; offset += 1
        X[idx, offset] = y; offset += 1
        X[idx, offset] = z; offset += 1

        # Country one-hot
        X[idx, offset + c_idx] = 1.0
        offset += num_countries

        # Graph-related features
        X[idx, offset] = degrees[node];             offset += 1
        X[idx, offset] = weighted_degrees[node];    offset += 1
        X[idx, offset] = clustering[node];          offset += 1
        X[idx, offset] = core_num[node];            offset += 1
        X[idx, offset] = international_count[node]; offset += 1
        X[idx, offset] = domestic_count[node];      offset += 1
        X[idx, offset] = total_mileage[node];       offset += 1

    # ==== 4) Log-transform / (optional) standardize continuous features ====
    # Continuous feature indices: x, y, z, degree, w_degree, clustering, core, intl, dom, total_mileage
    cont_idx = []

    # x, y, z
    cont_idx.extend([0, 1, 2])

    # Start index of 7 structural features
    base = 3 + num_countries
    idx_degree = base + 0
    idx_wdegree = base + 1
    idx_clust = base + 2
    idx_core = base + 3
    idx_intl = base + 4
    idx_dom = base + 5
    idx_miles = base + 6

    # For counts / large values, apply log1p
    for k in [idx_degree, idx_wdegree, idx_intl, idx_dom, idx_miles]:
        X[:, k] = np.log1p(X[:, k])

    cont_idx.extend([idx_degree, idx_wdegree, idx_clust,
                     idx_core, idx_intl, idx_dom, idx_miles])

    cont_idx = np.array(cont_idx, dtype=int)

    # # Standardization: subtract mean / divide by std
    # mu = X[:, cont_idx].mean(axis=0)
    # sigma = X[:, cont_idx].std(axis=0) + 1e-6
    # X[:, cont_idx] = (X[:, cont_idx] - mu) / sigma

    x_tensor = torch.from_numpy(X)
    return x_tensor, countries, country_to_idx




# ---------------------------------------------------
# 3. NetworkX -> PyG Data
#    edge_attr stores [weight, distance] (not used yet)
# ---------------------------------------------------
def graph_to_pyg(G: nx.Graph, x: torch.Tensor) -> Data:
    # edge_index: shape [2, E]
    edges = np.array(list(G.edges()), dtype=np.int64).T
    edge_index = torch.from_numpy(edges)
    edge_index = to_undirected(edge_index, num_nodes=x.size(0))

    # Store weight and distance into edge_attr (for potential weighted GCN usage)
    w_list = []
    d_list = []
    # Note: G.edges() gives each edge once (u < v)
    for u, v in G.edges():
        data = G[u][v]
        w_list.append(float(data.get("weight", 1.0)))
        d_list.append(float(data.get("distance", 0.0)))
    edge_weight = torch.tensor(w_list, dtype=torch.float32)
    edge_distance = torch.tensor(d_list, dtype=torch.float32)

    # Because we use to_undirected, we need to duplicate attributes
    edge_attr = torch.stack([edge_weight, edge_distance], dim=1)
    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)  # symmetric edges

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


# ---------------------------------------------------
# 4. GCN encoder + dot-product decoder
# ---------------------------------------------------
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=4, dropout=0.1):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x



def decode_dot(z, edge_index):
    """
    Inner-product decoder: given z and a set of (u, v), output logit scores.
    edge_index: [2, num_edges]
    """
    src = z[edge_index[0]]
    dst = z[edge_index[1]]
    return (src * dst).sum(dim=-1)


# ---------------------------------------------------
# 5. Training / evaluation functions (standard link prediction)
# ---------------------------------------------------
def train(model, train_data, device, optimizer):
    model.train()
    optimizer.zero_grad()

    x = train_data.x.to(device)
    ei = train_data.edge_index.to(device)

    z = model(x, ei)

    edge_label_index = train_data.edge_label_index.to(device)
    edge_label = train_data.edge_label.to(device).float()  # 0/1

    logits = decode_dot(z, edge_label_index)  # [num_edges]
    loss = F.binary_cross_entropy_with_logits(logits, edge_label)
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def eval_link_pred(model, data, device):
    model.eval()
    x = data.x.to(device)
    ei = data.edge_index.to(device)

    z = model(x, ei)

    edge_label_index = data.edge_label_index.to(device)
    edge_label = data.edge_label.to(device).float()

    logits = decode_dot(z, edge_label_index)
    prob = logits.sigmoid().cpu().numpy()
    labels = edge_label.cpu().numpy()

    auc = roc_auc_score(labels, prob)
    ap = average_precision_score(labels, prob)
    return auc, ap


# ---------------------------------------------------
# 5.1 Build weight regression targets (positive edges: G[u][v]['weight'], negative: 0)
# ---------------------------------------------------
def build_edge_weight_targets(G, edge_label_index, edge_label, device, log_target=True):
    """
    For each (u, v) in edge_label_index:
      - if edge_label = 1, use G[u][v]['weight'] as the ground-truth weight
      - if edge_label = 0, use weight = 0
    If log_target=True, return log(1 + weight) for regression.
    """
    src = edge_label_index[0].cpu().numpy()
    dst = edge_label_index[1].cpu().numpy()
    lbl = edge_label.cpu().numpy()

    weights = np.zeros_like(lbl, dtype=np.float32)
    for k in range(len(lbl)):
        if lbl[k] > 0.5:
            u = int(src[k])
            v = int(dst[k])
            w = float(G[u][v].get("weight", 1.0))  # positive edges take weight from the graph
            weights[k] = w
        else:
            weights[k] = 0.0                        # negative edges have weight=0

    if log_target:
        weights = np.log1p(weights)                # log(1 + w)

    return torch.from_numpy(weights).to(device)


# ---------------------------------------------------
# 5.2 Training / evaluation functions (weight regression)
# ---------------------------------------------------
def train_weight(model, train_data, device, optimizer, G, log_target=True):
    """
    Training objective: predict log(1 + weight), with negative samples having weight=0.
    """
    model.train()
    optimizer.zero_grad()

    x = train_data.x.to(device)
    ei = train_data.edge_index.to(device)
    z = model(x, ei)

    edge_label_index = train_data.edge_label_index.to(device)
    edge_label = train_data.edge_label.to(device)

    pred = decode_dot(z, edge_label_index)  # predicted log(1+w) (real-valued)
    target = build_edge_weight_targets(G, edge_label_index, edge_label,
                                       device, log_target=log_target)

    loss = F.mse_loss(pred, target)
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def eval_weight(model, data, device, G, log_target=True):
    """
    Evaluation for weight regression:
      - MSE / MAE (in log(1+w) space)
      - additionally, an existence AUC using prediction as a proxy (values > 0 as "edge exists")
    """
    model.eval()
    x = data.x.to(device)
    ei = data.edge_index.to(device)
    z = model(x, ei)

    edge_label_index = data.edge_label_index.to(device)
    edge_label = data.edge_label.to(device)

    pred = decode_dot(z, edge_label_index)
    target = build_edge_weight_targets(G, edge_label_index, edge_label,
                                       device, log_target=log_target)

    mse = F.mse_loss(pred, target).item()
    mae = F.l1_loss(pred, target).item()

    # Use predicted log(1+w_hat) for existence AUC (positive edges have log(1+w)>0, negatives=0)
    labels_exist = edge_label.cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    try:
        auc = roc_auc_score(labels_exist, pred_np)
    except ValueError:
        auc = float("nan")  # In corner cases (all-1 or all-0), AUC may not be defined

    return mse, mae, auc


# ---------------------------------------------------
# 7. Main
# ---------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) Read graph and build features
    G = load_airline_graph(args.gml_path)
    x, countries, country_to_idx = build_features(G)
    data_full = graph_to_pyg(G, x)

    # 2) Random train/val/test split for link prediction
    split = RandomLinkSplit(
        num_val=0.05,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=True,
        # edge_attr=True,  # keep edge_attr (although GCN does not use it yet)
    )
    train_data, val_data, test_data = split(data_full)

    model = GCNEncoder(
        in_channels=data_full.num_features,
        hidden_channels=args.hidden_dim,
        num_layers=args.layers,        # can be 3 or 4
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 3) Training
    if args.task == "exist":
        print("Task = exist (edge existence prediction)")
        for epoch in range(1, args.epochs + 1):
            loss = train(model, train_data, device, optimizer)
            if epoch % 100 == 0 or epoch == 1:
                val_auc, val_ap = eval_link_pred(model, val_data, device)
                print(
                    f"Epoch {epoch:03d} | loss={loss:.4f} | "
                    f"val AUC={val_auc:.4f}, val AP={val_ap:.4f}"
                )

        # 4) Test evaluation (existence)
        test_auc, test_ap = eval_link_pred(model, test_data, device)
        print(f"\n[Existence] Test AUC={test_auc:.4f}, Test AP={test_ap:.4f}")

        os.makedirs("./GCN/model", exist_ok=True)
        torch.save(model.state_dict(), "./GCN/model/GCNEncoder_exist.pt")

    else:  # args.task == "weight"
        print("Task = weight (edge weight regression)")
        for epoch in range(1, args.epochs + 1):
            loss = train_weight(model, train_data, device, optimizer, G,
                                log_target=True)
            if epoch % 100 == 0 or epoch == 1:
                val_mse, val_mae, val_auc = eval_weight(
                    model, val_data, device, G, log_target=True
                )
                print(
                    f"Epoch {epoch:03d} | loss={loss:.4f} | "
                    f"val MSE={val_mse:.4f}, MAE={val_mae:.4f}, "
                    f"exist-AUC={val_auc:.4f}"
                )

        # 4) Test evaluation (weight regression)
        test_mse, test_mae, test_auc = eval_weight(
            model, test_data, device, G, log_target=True
        )
        print(
            f"\n[Weight] Test MSE={test_mse:.4f}, MAE={test_mae:.4f}, "
            f"exist-AUC={test_auc:.4f}"
        )
        
        os.makedirs("./GCN/model", exist_ok=True)
        torch.save(model.state_dict(), "./GCN/model/GCNEncoder_weight.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GCN link prediction on OpenFlights airline network"
    )
    parser.add_argument(
        "--gml-path",
        type=str,
        default="GCN/global_airline_network.gml",
        help="Path to global_airline_network.gml",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=6400)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--task",
        type=str,
        default="exist",
        choices=["exist", "weight"],
        help="exist: edge existence prediction; weight: edge weight regression",
    )
    args = parser.parse_args()
    main(args)
