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
# 1. 读取 GML 并只保留最大连通分量（孤立点去掉）
# ---------------------------------------------------
def load_airline_graph(path: str) -> nx.Graph:
    # 读 GML
    G = nx.read_gml(path)

    # 如果原图是有向 / MultiGraph，这里统一成无向简单图
    G = nx.Graph(G)


    # 只保留最大连通分量（自动去掉孤立节点）
    cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(cc).copy()
    
    # 重新编号成 0..N-1，原来的 id 存到 old_id 属性
    G = nx.convert_node_labels_to_integers(G, label_attribute="old_id")

    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ---------------------------------------------------
# 2. 构建节点特征：lat, lon, one-hot(country)
# ---------------------------------------------------
def build_features(G: nx.Graph):
    """
    每个节点特征：
    - x, y, z（由 lat, lon 转成单位球坐标）
    - country one-hot
    - degree
    - weighted_degree
    - clustering
    - core_number
    - international_routes
    - domestic_routes
    - total_mileage (sum(weight * distance))

    并对连续特征做 log/标准化。
    """
    import numpy as np
    import networkx as nx

    num_nodes = G.number_of_nodes()

    # ==== 1) 国家 one-hot ====
    countries = sorted({data["country"] for _, data in G.nodes(data=True)})
    country_to_idx = {c: i for i, c in enumerate(countries)}
    num_countries = len(countries)

    # ==== 2) 图结构统计量 ====
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

        w = float(data.get("weight", 1.0))        # 无向图很多是 2（往返）
        d = float(data.get("distance", 0.0))

        # 国际/国内航线数量（按边条数计，不按 weight）
        if cu is not None and cv is not None and cu == cv:
            domestic_count[u] += 1.0
            domestic_count[v] += 1.0
        else:
            international_count[u] += 1.0
            international_count[v] += 1.0

        # 总里程数：weight * distance，两端节点都加
        miles = w * d
        total_mileage[u] += miles
        total_mileage[v] += miles

    # ==== 3) 构建特征矩阵 ====
    # 3 (x,y,z) + num_countries (one-hot) + 7 结构特征
    feat_dim = 3 + num_countries + 7
    X = np.zeros((num_nodes, feat_dim), dtype=np.float32)

    for node, data in G.nodes(data=True):
        idx = int(node)

        lat = float(data.get("lat", 0.0))   # 单位：度
        lon = float(data.get("lon", 0.0))   # 单位：度
        country = data["country"]
        c_idx = country_to_idx[country]

        # ---- 经纬 -> 球面坐标 (单位球上的 x,y,z) ----
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

        # 国家 one-hot
        X[idx, offset + c_idx] = 1.0
        offset += num_countries

        # 图特征
        X[idx, offset] = degrees[node];             offset += 1
        X[idx, offset] = weighted_degrees[node];    offset += 1
        X[idx, offset] = clustering[node];          offset += 1
        X[idx, offset] = core_num[node];            offset += 1
        X[idx, offset] = international_count[node]; offset += 1
        X[idx, offset] = domestic_count[node];      offset += 1
        X[idx, offset] = total_mileage[node];       offset += 1

    # ==== 4) 连续特征做 log/标准化 ====
    # 连续特征列：x, y, z, degree, w_degree, clustering, core, intl, dom, total_mileage
    cont_idx = []

    # x, y, z
    cont_idx.extend([0, 1, 2])

    # 后面 7 个结构特征起始位置
    base = 3 + num_countries
    idx_degree = base + 0
    idx_wdegree = base + 1
    idx_clust = base + 2
    idx_core = base + 3
    idx_intl = base + 4
    idx_dom = base + 5
    idx_miles = base + 6

    # 对“计数/大数”先 log1p
    for k in [idx_degree, idx_wdegree, idx_intl, idx_dom, idx_miles]:
        X[:, k] = np.log1p(X[:, k])

    cont_idx.extend([idx_degree, idx_wdegree, idx_clust,
                     idx_core, idx_intl, idx_dom, idx_miles])

    cont_idx = np.array(cont_idx, dtype=int)

    # # 标准化：减均值 / 除标准差
    # mu = X[:, cont_idx].mean(axis=0)
    # sigma = X[:, cont_idx].std(axis=0) + 1e-6
    # X[:, cont_idx] = (X[:, cont_idx] - mu) / sigma

    x_tensor = torch.from_numpy(X)
    return x_tensor, countries, country_to_idx




# ---------------------------------------------------
# 3. NetworkX -> PyG Data
#    edge_attr 中保存 [weight, distance]（暂时不用）
# ---------------------------------------------------
def graph_to_pyg(G: nx.Graph, x: torch.Tensor) -> Data:
    # edge_index: shape [2, E]
    edges = np.array(list(G.edges()), dtype=np.int64).T
    edge_index = torch.from_numpy(edges)
    edge_index = to_undirected(edge_index, num_nodes=x.size(0))

    # 把 weight 和 distance 存到 edge_attr（方便以后用加权 GCN）
    w_list = []
    d_list = []
    # 注意：G.edges() 只给每条边一次（u<v）
    for u, v in G.edges():
        data = G[u][v]
        w_list.append(float(data.get("weight", 1.0)))
        d_list.append(float(data.get("distance", 0.0)))
    edge_weight = torch.tensor(w_list, dtype=torch.float32)
    edge_distance = torch.tensor(d_list, dtype=torch.float32)

    # 由于我们用 to_undirected，会复制边，因此也复制属性
    edge_attr = torch.stack([edge_weight, edge_distance], dim=1)
    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)  # 对称边

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


# ---------------------------------------------------
# 4. GCN 编码器 + 内积解码器
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
            if i != len(self.convs) - 1:  # 最后一层一般不再 ReLU
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # 节点嵌入 z



def decode_dot(z, edge_index):
    """
    内积解码器：给定 z 和若干 (u, v)，输出 logit 得分
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
# 5.1 构造 weight 回归的标签（正边用 G 里的 weight，负边=0）
# ---------------------------------------------------
def build_edge_weight_targets(G, edge_label_index, edge_label, device, log_target=True):
    """
    对于 edge_label_index 中的每条 (u, v)：
      - 如果 edge_label = 1，则从 G[u][v]['weight'] 里取真实权重
      - 如果 edge_label = 0，则权重 = 0
    如果 log_target=True，则返回 log(1 + weight)，方便回归训练
    """
    src = edge_label_index[0].cpu().numpy()
    dst = edge_label_index[1].cpu().numpy()
    lbl = edge_label.cpu().numpy()

    weights = np.zeros_like(lbl, dtype=np.float32)
    for k in range(len(lbl)):
        if lbl[k] > 0.5:
            u = int(src[k])
            v = int(dst[k])
            w = float(G[u][v].get("weight", 1.0))  # 正边从图上取 weight
            weights[k] = w
        else:
            weights[k] = 0.0                        # 负边权重=0

    if log_target:
        weights = np.log1p(weights)                # log(1 + w)

    return torch.from_numpy(weights).to(device)

@torch.no_grad()
def eval_weight(model, data, device, G, log_target=True):
    """
    weight 回归的评估：
      - MSE / MAE （在 log(1+w) 空间上）
      - 额外给一个基于 pred 的存在性 AUC（把 >0 当成“有边”）
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

    # 用预测的 log(1+w_hat) 做 existence AUC（正边 log(1+w)>0，负边=0）
    labels_exist = edge_label.cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    try:
        auc = roc_auc_score(labels_exist, pred_np)
    except ValueError:
        auc = float("nan")  # 极端情况下（全1/全0）AUC 可能算不出来

    return mse, mae, auc

def build_candidate_pairs(
    G,
    xyz,
    sp_length,
    min_geo_km=None,   # 地理距离下限（太近的不考虑）
    max_geo_km=None,   # 地理距离上限（太远的不考虑）
    min_sp=None,       # 最短路径长度下限（比如 3：只考虑 A–B–C–D → A–D）
    max_sp=None,       # 最短路径长度上限
    same_country=None, # True: 只国内; False: 只国际; None: 不限制
):
    """
    返回候选节点对列表 [(i, j), ...]，i < j 且目前 G 中没有边。
    """
    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {u: int(u) for u in nodes}  # 这里节点本来就应该是 0..N-1

    def geo_dist_km(i, j):
        # xyz 在单位球上；弦长 ~ chord distance
        vi = xyz[i]
        vj = xyz[j]
        # 夹角 = arccos( dot )
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
            
            # 已经有边就跳过
            if G.has_edge(u, v):
                continue

            # === 最短路径长度筛选 ===
            if min_sp is not None or max_sp is not None:
                # 如果 cutoff 太小，可能拿不到某些 dist
                d = None
                if v in sp_length.get(u, {}):
                    d = sp_length[u][v]
                elif u in sp_length.get(v, {}):
                    d = sp_length[v][u]

                if d is None:
                    # 没有路径 or 路径长度 > cutoff，当成太远，直接跳过
                    continue

                if min_sp is not None and d < min_sp:
                    continue
                if max_sp is not None and d > max_sp:
                    continue

            # === 国家关系筛选（国内/国际） ===
            if same_country is not None:
                cu = G.nodes[u].get("country")
                cv = G.nodes[v].get("country")
                if cu is None or cv is None:
                    # 缺失信息就跳过（你也可以选择保留）
                    continue
                is_same = (cu == cv)
                if same_country is True and not is_same:
                    continue
                if same_country is False and is_same:
                    continue

            # === 地理距离筛选 ===
            if min_geo_km is not None or max_geo_km is not None:
                d_geo = geo_dist_km(i, j)
                if min_geo_km is not None and d_geo < min_geo_km:
                    continue
                if max_geo_km is not None and d_geo > max_geo_km:
                    continue

            candidates.append((i, j, d, is_same, d_geo))

    return candidates
# ---------------------------------------------------
# 6. 在整张图上找“高分缺失边”（支持存在性 / weight 两种模式）
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
    mode="exist",  # "exist"：存在性，"weight"：预测权重
):
    model.eval()
    x = full_data.x.to(device)
    ei = full_data.edge_index.to(device)

    z = model(x, ei).cpu()      # [N, d]
    xyz = full_data.x[:, :3].cpu().numpy()  # 假设前 3 维是 xyz

    # 预先算 shortest path（只算一次，可以挪到 main 里缓存）
    sp_length = dict(nx.all_pairs_shortest_path_length(G, cutoff=5))

    # 构造候选边集合
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

    # 对这些候选对打分
    scored = []
    for i, j, d, is_same, d_geo in candidates:
        zi = z[i]
        zj = z[j]
        score = float((zi * zj).sum())  # 对应 decode_dot 的输出

        nu = G.nodes[i]
        nv = G.nodes[j]
        name_u = nu.get("name", str(i))
        name_v = nv.get("name", str(j))
        country_u = nu.get("country", "")
        country_v = nv.get("country", "")

        if mode == "exist":
            prob = 1.0 / (1.0 + math.exp(-score))   # logit -> 概率
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
            # score 视为预测的 log(1 + w_hat)
            # 为了安全，clip 一下，避免 exp 爆炸
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

    # 按 score 降序取 top_k（score 对存在性和 log(1+w) 都是单调的）
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]



# ---------------------------------------------------
# 7. 主流程
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
        
    # # 1) 读图并构建特征
    G = load_airline_graph(args.gml_path)
    x, countries, country_to_idx = build_features(G)
    data_full = graph_to_pyg(G, x)
    
    model = GCNEncoder(
        in_channels=data_full.num_features,
        hidden_channels=args.hidden_dim,
        num_layers=args.layers,        # 改成 3 或 4 都可以
        dropout=0.1,
    ).to(device)
    
    if args.task == "exist":
        print("Task = exist (边存在性预测)")
        model.load_state_dict(torch.load("./GCN/model/GCNEncoder_exist.pt"))
        model.eval()

        # 5) 在整图上找“潜在应存在的边”
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
        print("Task = weight (边权重回归)")
        model.load_state_dict(torch.load("./GCN/model/GCNEncoder_weight.pt"))
        model.eval()

        # 5) 在整图上找“潜在权重很大的缺失边”
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
    parser.add_argument("--task", type=str, default="exist",
                    choices=["exist", "weight"],
                    help="exist: 边存在性预测; weight: 边权重回归")
    parser.add_argument("--min-geo-km", type=float, default=100.0,
                        help="候选边最小地理距离（km），例如 100km 避免同城机场。")
    parser.add_argument("--max-geo-km", type=float, default=5000.0,
                        help="候选边最大地理距离（km），默认5000km。")
    parser.add_argument("--min-sp", type=int, default=1,
                        help="候选边最短路径长度下限（基于现有图），例如 3 表示 A-B-C-D -> A-D。")
    parser.add_argument("--max-sp", type=int, default=4,
                        help="候选边最短路径长度上限。")
    parser.add_argument("--same-country", type=str, default="any",
                        choices=["any", "same", "diff"],
                        help="候选边是否限制国家关系："
                             "'any' 不限制, 'same' 只考虑国内航线, 'diff' 只考虑国际航线。")
    args = parser.parse_args()
    main(args)
