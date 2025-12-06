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
    def __init__(self, in_channels, hidden_channels, num_layers=3, dropout=0.1):
        super().__init__()
        assert num_layers >= 2

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:  # usually no ReLU on the last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # node embeddings z



def decode_dot(z, edge_index):
    """
    Inner-product decoder: given z and a set of (u, v), output logit scores
    edge_index: [2, num_edges]
    """
    src = z[edge_index[0]]
    dst = z[edge_index[1]]
    return (src * dst).sum(dim=-1)


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


def build_candidate_pairs(
    G,
    xyz,
    sp_length,
    min_geo_km=None,   # minimum geographic distance (too close -> skip)
    max_geo_km=None,   # maximum geographic distance (too far -> skip)
    min_sp=None,       # minimum shortest-path distance (e.g., 3 to target A–B–C–D → A–D)
    max_sp=None,       # maximum shortest-path distance
    same_country=None, # True: only domestic; False: only international; None: no constraint
):
    """
    Return a list of candidate node pairs [(i, j), ...], with i < j and no existing edge in G.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {u: int(u) for u in nodes}  # here nodes are already 0..N-1

    def geo_dist_km(i, j):
        # xyz are on unit sphere; chord length approximates great-circle distance
        vi = xyz[i]
        vj = xyz[j]
        # angle = arccos(dot)
        dot = float(np.clip(np.dot(vi, vj), -1.0, 1.0))
        angle = math.acos(dot)
        R_earth = 6371.0  # km
        return R_earth * angle

    candidates = []

    for i in range(n):
        for j in range(i + 1, n):
            u, v = nodes[i], nodes[j]
            d = None
            is_same = None
            d_geo = None
            
            # skip if edge already exists
            if G.has_edge(u, v):
                continue

            # === Shortest-path length filter ===
            if min_sp is not None or max_sp is not None:
                # If cutoff is too small, some distances may be missing
                d = None
                if v in sp_length.get(u, {}):
                    d = sp_length[u][v]
                elif u in sp_length.get(v, {}):
                    d = sp_length[v][u]

                if d is None:
                    # no path or path length > cutoff → skip
                    continue

                if min_sp is not None and d < min_sp:
                    continue
                if max_sp is not None and d > max_sp:
                    continue

            # === Country relation filter (domestic/international) ===
            if same_country is not None:
                cu = G.nodes[u].get("country")
                cv = G.nodes[v].get("country")
                if cu is None or cv is None:
                    # drop nodes with missing country information (could also keep them if desired)
                    continue
                is_same = (cu == cv)
                if same_country is True and not is_same:
                    continue
                if same_country is False and is_same:
                    continue

            # === Geographic distance filter ===
            if min_geo_km is not None or max_geo_km is not None:
                d_geo = geo_dist_km(i, j)
                if min_geo_km is not None and d_geo < min_geo_km:
                    continue
                if max_geo_km is not None and d_geo > max_geo_km:
                    continue

            candidates.append((i, j, d, is_same, d_geo))

    return candidates


# ---------------------------------------------------
# 6. On the full graph, search for high-scoring missing edges
#    (supports both existence and weight modes)
# ---------------------------------------------------
@torch.no_grad()
def suggest_missing_edges_filtered(
    model,
    full_data,
    G,
    device,
    top_k=20,
    min_geo_km=None,
    max_geo_km=None,
    min_sp=None,
    max_sp=None,
    same_country=None,
    mode="exist",  # "exist": edge existence, "weight": edge weight regression
):
    model.eval()
    x = full_data.x.to(device)
    ei = full_data.edge_index.to(device)

    z = model(x, ei).cpu()      # [N, d]
    xyz = full_data.x[:, :3].cpu().numpy()  # assume first 3 dims are xyz

    # Pre-compute shortest paths (compute once; could be cached in main if needed)
    sp_length = dict(nx.all_pairs_shortest_path_length(G, cutoff=5))

    # Build candidate set
    candidates = build_candidate_pairs(
        G,
        xyz,
        sp_length,
        min_geo_km=min_geo_km,
        max_geo_km=max_geo_km,
        min_sp=min_sp,
        max_sp=max_sp,
        same_country=same_country,
    )

    print(f"Candidate pairs after filtering: {len(candidates)}")

    # Score candidate pairs
    scored = []
    for i, j, d, is_same, d_geo in candidates:
        zi = z[i]
        zj = z[j]
        score = float((zi * zj).sum())  # matches decode_dot

        nu = G.nodes[i]
        nv = G.nodes[j]
        name_u = nu.get("name", str(i))
        name_v = nv.get("name", str(j))
        country_u = nu.get("country", "")
        country_v = nv.get("country", "")

        if mode == "exist":
            prob = 1.0 / (1.0 + math.exp(-score))   # logit -> probability
            entry = {
                "u": i,
                "v": j,
                "airport_u": name_u,
                "airport_v": name_v,
                "country_u": country_u,
                "country_v": country_v,
                "score": score,
                "prob": prob,
                "weight": None,
                "hop": d,
                "same_country": is_same,
                "distance": d_geo,
            }
        elif mode == "weight":
            # Interpret score as predicted log(1 + w_hat)
            # Clip to avoid overflow in exp
            logw = max(score, 0.0)
            w_hat = math.exp(logw) - 1.0
            entry = {
                "u": i,
                "v": j,
                "airport_u": name_u,
                "airport_v": name_v,
                "country_u": country_u,
                "country_v": country_v,
                "score": score,
                "prob": None,
                "weight": w_hat,
                "hop": d,
                "same_country": is_same,
                "distance": d_geo,
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")

        scored.append(entry)

    # Sort by score in descending order and take top_k
    # (score is monotonic for both existence and log(1+w))
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]



# ---------------------------------------------------
# 7. Main
# ---------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if args.same_country == "any":
        same_country_flag = None
    elif args.same_country == "same":
        same_country_flag = True
    else:  # "diff"
        same_country_flag = False
        
    # 1) Read graph and build features
    G = load_airline_graph(args.gml_path)
    x, countries, country_to_idx = build_features(G)
    data_full = graph_to_pyg(G, x)
    
    model = GCNEncoder(
        in_channels=data_full.num_features,
        hidden_channels=args.hidden_dim,
        num_layers=args.layers,        # can be 3 or 4
        dropout=0.1,
    ).to(device)
    
    if args.task == "exist":
        print("Task = exist (edge existence prediction)")
        model.load_state_dict(torch.load("./GCN/model/GCNEncoder_exist.pt"))
        model.eval()

        # Find potentially missing edges that should exist
        top_edges = suggest_missing_edges_filtered(
            model,
            data_full,
            G,
            device,
            top_k=args.top_k,
            min_geo_km=args.min_geo_km,
            max_geo_km=args.max_geo_km,
            min_sp=args.min_sp,
            max_sp=args.max_sp,
            same_country=same_country_flag,
            mode="exist",
        )

        print(f"\nTop {args.top_k} predicted missing edges (existence):")
        for e in top_edges:
            u = e["u"]
            v = e["v"]
            airport_u = e["airport_u"]
            airport_v = e["airport_v"]
            country_u = e["country_u"]
            country_v = e["country_v"]
            prob = e["prob"]
            hop = e["hop"]
            distance = e['distance']
            is_same = e['same_country']

            nu = G.nodes[u]
            nv = G.nodes[v]
            old_u = nu.get("old_id", u)
            old_v = nv.get("old_id", v)
            print(
                f"{old_u} ({airport_u}, {country_u})  "
                f"<-->  {old_v} ({airport_v}, {country_v})  "
                f"prob={prob:.4f}  "
                f"hop={hop}  "
                f"distance={distance:.4f}  "
                + ("not " if not is_same else "") +
                f"in same country"
            )

    else:  # args.task == "weight"
        print("Task = weight (edge weight regression)")
        model.load_state_dict(torch.load("./GCN/model/GCNEncoder_weight.pt"))
        model.eval()

        # Find missing edges with large predicted weight
        top_edges = suggest_missing_edges_filtered(
            model,
            data_full,
            G,
            device,
            top_k=args.top_k,
            min_geo_km=args.min_geo_km,
            max_geo_km=args.max_geo_km,
            min_sp=args.min_sp,
            max_sp=args.max_sp,
            same_country=same_country_flag,
            mode="weight",
        )

        print(f"\nTop {args.top_k} predicted missing edges (by weight):")
        for e in top_edges:
            u = e["u"]
            v = e["v"]
            airport_u = e["airport_u"]
            airport_v = e["airport_v"]
            country_u = e["country_u"]
            country_v = e["country_v"]
            w_hat = e["weight"]
            hop = e["hop"]
            distance = e['distance']
            is_same = e['same_country']

            nu = G.nodes[u]
            nv = G.nodes[v]
            old_u = nu.get("old_id", u)
            old_v = nv.get("old_id", v)
            print(
                f"{old_u} ({airport_u}, {country_u})  "
                f"<-->  {old_v} ({airport_v}, {country_v})  "
                f"pred_weight≈{w_hat:.2f}  "
                f"hop={hop}  "
                f"distance={distance:.4f}  "
                + ("not " if not is_same else "") +
                f"in same country"
            )



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
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--task",
        type=str,
        default="exist",
        choices=["exist", "weight"],
        help="exist: edge existence prediction; weight: edge weight regression",
    )
    parser.add_argument(
        "--min-geo-km",
        type=float,
        default=100.0,
        help="Minimum geographic distance (km) for candidate edges, e.g., 100km to avoid same-city airports.",
    )
    parser.add_argument(
        "--max-geo-km",
        type=float,
        default=5000.0,
        help="Maximum geographic distance (km) for candidate edges, default 5000km.",
    )
    parser.add_argument(
        "--min-sp",
        type=int,
        default=1,
        help="Lower bound on shortest-path length in the existing graph, e.g., 3 means A-B-C-D -> A-D.",
    )
    parser.add_argument(
        "--max-sp",
        type=int,
        default=4,
        help="Upper bound on shortest-path length.",
    )
    parser.add_argument(
        "--same-country",
        type=str,
        default="any",
        choices=["any", "same", "diff"],
        help="Whether to constrain country relation for candidate edges: 'any' no constraint; 'same' only domestic; 'diff' only international.",
    )
    args = parser.parse_args()
    main(args)
