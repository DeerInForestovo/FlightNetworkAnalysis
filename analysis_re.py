#!/usr/bin/env python3
"""
Global Airline Network Analysis (OpenFlights-based)

This script:
  1. Parses OpenFlights-style airports.dat and routes.dat
  2. Builds an undirected airport-to-airport network
  3. Keeps only the giant connected component ("global air transport network")
  4. Computes:
      - Basic global stats (size, avg path length, clustering, etc.)
      - Node-level centrality metrics (degree, betweenness, ...)
      - Distributions / diagnostic plots (degree CCDF, k-core histogram, etc.)
      - Country-level aggregation graphs
      - Country knock-out impact (who is systemically critical?)
      - Mixing patterns by country & assortativity

Artifacts saved under OUTPUT_DIR:
    global_airline_network.gml
    centrality_metrics.csv
    top10_hubs.csv
    degree_distribution.png
    degree_ccdf.png
    betweenness_distribution.png
    kcore_hist.png
    mixing_matrix_country.png
    assortativity.txt
    country_graph.gml
    country_centrality.csv
    country_impact.csv
    run_meta.json

Heavy robustness curves (progressive node/edge removal) are included as
helper functions (`removal_curve`, etc.) but are not executed by default
for runtime reasons.
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, List
import json
from collections import Counter

plt.style.use("ggplot")

# ----------- Paths / constants -----------
OUTPUT_DIR = Path("./data/output")
AIRPORTS_PATH = Path("./data/airports.dat")
ROUTES_PATH = Path("./data/routes.dat")

BETWEENNESS_SAMPLE_K = 200   # sampling size for betweenness approximation
BETWEENNESS_SEED = 42

# ----------- Data loading / graph build -----------

def load_raw_openflights(
    airports_path: Path,
    routes_path: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load OpenFlights-style airports.dat and routes.dat into DataFrames
    with reasonable column names / dtypes.
    """
    airports_raw = pd.read_csv(
        airports_path, header=None, dtype=object
    )
    routes_raw = pd.read_csv(
        routes_path, header=None, dtype=object
    )

    airports_cols = [
        "AirportID","Name","City","Country","IATA","ICAO",
        "Latitude","Longitude","Altitude","Timezone","DST",
        "TZ","Type","Source"
    ]
    if airports_raw.shape[1] > len(airports_cols):
        extras = [f"extra_{i}" for i in range(airports_raw.shape[1]-len(airports_cols))]
        airports_raw.columns = airports_cols + extras
    else:
        airports_raw.columns = airports_cols[:airports_raw.shape[1]]

    routes_cols = [
        "Airline","AirlineID","SrcIATA","SrcAirportID",
        "DstIATA","DstAirportID","Codeshare","Stops","Equipment"
    ]
    if routes_raw.shape[1] > len(routes_cols):
        extras_r = [f"extra_{i}" for i in range(routes_raw.shape[1]-len(routes_cols))]
        routes_raw.columns = routes_cols + extras_r
    else:
        routes_raw.columns = routes_cols[:routes_raw.shape[1]]

    # numeric conversions for coords
    airports_raw["Latitude"] = pd.to_numeric(airports_raw["Latitude"], errors="coerce")
    airports_raw["Longitude"] = pd.to_numeric(airports_raw["Longitude"], errors="coerce")

    return airports_raw, routes_raw


def build_full_graph(
    airports_df: pd.DataFrame,
    routes_df: pd.DataFrame
) -> nx.Graph:
    """
    Build an undirected graph from airports (nodes) and routes (edges).
    Node attributes:
        - name, country, city, IATA, lat, lon
    Edge: unweighted (presence of route)
    """
    G_full = nx.Graph()

    # add airport nodes
    for _, row in airports_df.iterrows():
        aid = row.get("AirportID")
        if pd.isna(aid):
            continue
        aid = str(aid)

        G_full.add_node(
            aid,
            name=row.get("Name", ""),
            country=row.get("Country", ""),
            city=row.get("City", ""),
            iata=row.get("IATA", ""),
            lat=row.get("Latitude", np.nan),
            lon=row.get("Longitude", np.nan),
        )

    # add edges for routes
    for _, row in routes_df.iterrows():
        src = row.get("SrcAirportID", None)
        dst = row.get("DstAirportID", None)

        if src is None or dst is None:
            continue
        if src == r"\\N" or dst == r"\\N" or src == r"\N" or dst == r"\N":
            continue

        src = str(src)
        dst = str(dst)

        if src == dst:
            continue  # skip self loops

        if src in G_full.nodes and dst in G_full.nodes:
            G_full.add_edge(src, dst)

    return G_full


def get_giant_component(G_full: nx.Graph) -> nx.Graph:
    """
    Extract the largest connected component to get a single global network.
    """
    largest_cc_nodes = max(nx.connected_components(G_full), key=len)
    G = G_full.subgraph(largest_cc_nodes).copy()
    return G

# ----------- Basic stats -----------

def compute_basic_stats(G: nx.Graph) -> Dict[str, float]:
    """
    Compute descriptive stats for the connected airport network.
    """
    avg_degree = float(np.mean([deg for _, deg in G.degree()]))
    avg_clustering = nx.average_clustering(G)
    avg_path_length = nx.average_shortest_path_length(G)

    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "avg_degree": avg_degree,
        "avg_clustering": avg_clustering,
        "avg_path_length": avg_path_length,
    }

# ----------- Centrality metrics -----------

def compute_centrality_table(
    G: nx.Graph,
    betw_sample_k: int = BETWEENNESS_SAMPLE_K,
    betw_seed: int = BETWEENNESS_SEED,
) -> pd.DataFrame:
    """
    Compute per-airport centralities:
      - Degree
      - DegreeCentrality
      - Betweenness (approx via node sampling)
      - Closeness
      - Eigenvector (power iteration)
    Return DataFrame with airport metadata + metrics.
    """
    degree_dict = dict(G.degree())
    degree_centrality = nx.degree_centrality(G)

    betweenness_centrality = nx.betweenness_centrality(
        G, k=min(betw_sample_k, len(G)), seed=betw_seed
    )

    closeness_centrality = nx.closeness_centrality(G)

    try:
        eigen_centrality = nx.eigenvector_centrality(G, max_iter=500)
    except Exception:
        # fallback in case of convergence issues
        eigen_centrality = {n: np.nan for n in G.nodes()}

    df = pd.DataFrame({
        "Airport_ID": list(G.nodes()),
        "Name":        [G.nodes[n].get("name", "Unknown")    for n in G.nodes()],
        "Country":     [G.nodes[n].get("country", "Unknown") for n in G.nodes()],
        "City":        [G.nodes[n].get("city", "")           for n in G.nodes()],
        "IATA":        [G.nodes[n].get("iata", "")           for n in G.nodes()],
        "Latitude":    [G.nodes[n].get("lat", np.nan)        for n in G.nodes()],
        "Longitude":   [G.nodes[n].get("lon", np.nan)        for n in G.nodes()],
        "Degree":      [degree_dict[n]                       for n in G.nodes()],
        "DegreeCentrality":   [degree_centrality[n]          for n in G.nodes()],
        "Betweenness":        [betweenness_centrality[n]     for n in G.nodes()],
        "Closeness":          [closeness_centrality[n]       for n in G.nodes()],
        "Eigenvector":        [eigen_centrality[n]           for n in G.nodes()],
    })
    return df


def save_centrality_tables(
    df: pd.DataFrame,
    out_csv_all: Path,
    out_csv_top10: Path,
) -> None:
    """
    Save full centrality metrics, and also the top-10 hubs by Betweenness.
    """
    df.to_csv(out_csv_all, index=False)

    top10 = df.sort_values("Betweenness", ascending=False).head(10).copy()
    top10.to_csv(out_csv_top10, index=False)

# ----------- Plots / distributions -----------

def plot_degree_distribution(G: nx.Graph, out_png: Path) -> None:
    """
    Histogram of degrees on log-log axes (heavy-tail structure).
    """
    degrees = np.array([deg for _, deg in G.degree()])
    dmin, dmax = degrees.min(), degrees.max()
    bins = np.logspace(np.log10(max(dmin,1)), np.log10(dmax), 40)

    plt.figure(figsize=(7,4))
    plt.hist(degrees, bins=bins, edgecolor="white", alpha=0.8)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Airport degree (# direct connections)")
    plt.ylabel("Count of airports")
    plt.title("Global Airline Network — Degree Distribution (log-log)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_degree_ccdf(G: nx.Graph, out_png: Path) -> None:
    """
    Complementary CDF of degree. Classic way to show scale-free-like tail.
    """
    degrees = np.array([deg for _, deg in G.degree()])
    xs = np.sort(degrees)
    ccdf = 1.0 - np.arange(1, len(xs)+1)/len(xs)

    plt.figure(figsize=(7,4))
    plt.loglog(xs, ccdf, marker='.', linestyle='none')
    plt.xlabel("Degree")
    plt.ylabel("CCDF (1 - CDF)")
    plt.title("Degree CCDF (log-log)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_betweenness_hist(df: pd.DataFrame, out_png: Path) -> None:
    """
    Distribution of betweenness centrality per airport.
    Shows 'chokepoint' hubs vs normal airports.
    """
    vals = df["Betweenness"].to_numpy()
    plt.figure(figsize=(7,4))
    plt.hist(vals, bins=50, edgecolor="white", alpha=0.8)
    plt.yscale("log")
    plt.xlabel("Betweenness centrality")
    plt.ylabel("Airport count (log scale)")
    plt.title("Betweenness Centrality Distribution")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_kcore_hist(G: nx.Graph, out_png: Path) -> None:
    """
    k-core number histogram:
    Higher core index ~ 'deeper' in global core.
    """
    cores = nx.core_number(G)
    vals = list(cores.values())
    plt.figure(figsize=(7,4))
    plt.hist(vals, bins=30, edgecolor="white", alpha=0.8)
    plt.xlabel("k-core index")
    plt.ylabel("Airport count")
    plt.title("k-core Decomposition Histogram")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_rich_club(G: nx.Graph, out_png: Path) -> None:
    """
    Rich-club coefficient φ(k) vs k.
    NOTE: can be somewhat expensive; not called by default.
    """
    rc = nx.rich_club_coefficient(G, normalized=False)
    ks, phis = zip(*sorted(rc.items()))
    plt.figure(figsize=(7,4))
    plt.plot(ks, phis, marker="o", ms=3)
    plt.xlabel("k (degree threshold)")
    plt.ylabel("Rich-club φ(k)")
    plt.title("Rich-club coefficient vs k")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

# ----------- Mixing / assortativity -----------

def save_degree_assortativity(G: nx.Graph, out_txt: Path) -> float:
    """
    Compute degree assortativity coefficient and save to text.
    r < 0 means disassortative (hubs connect to low-degree nodes),
    r > 0 means assortative (hubs connect to hubs).
    """
    r = nx.degree_assortativity_coefficient(G)
    out_txt.write_text(f"degree_assortativity: {r:.6f}\n")
    return r


def plot_country_mixing_heatmap(
    G: nx.Graph,
    out_png: Path,
    top_k: int = 20
) -> None:
    """
    Build a symmetric heatmap: how many edges connect country i <-> country j.
    Only plot the top_k most common countries by node count to keep matrix small.
    """
    # pick top_k most common countries by airport count
    country_list = [d.get("country","Unknown") for _, d in G.nodes(data=True)]
    freq = Counter(country_list)
    top_countries = [c for c,_ in freq.most_common(top_k)]

    idx = {c:i for i,c in enumerate(top_countries)}
    M = np.zeros((len(top_countries), len(top_countries)), dtype=float)

    for u, v in G.edges():
        cu = G.nodes[u].get("country","Unknown")
        cv = G.nodes[v].get("country","Unknown")
        if cu in idx and cv in idx:
            i, j = idx[cu], idx[cv]
            M[i,j] += 1
            if i != j:
                M[j,i] += 1

    M = np.log1p(M)  # log scale for better visualization
    plt.figure(figsize=(8,6))
    plt.imshow(M, aspect="auto")
    plt.xticks(range(len(top_countries)), top_countries, rotation=45, ha="right")
    plt.yticks(range(len(top_countries)), top_countries)
    plt.colorbar(label="Number of direct airport-to-airport routes(log scale)")
    plt.title("Country-to-Country Connectivity (Top countries)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

# ----------- Country-level aggregation -----------

def build_country_graph(G: nx.Graph) -> nx.Graph:
    """
    Collapse airports into countries. The new graph H has:
      - nodes: countries
      - edges: if any airport in country A connects to any airport in country B
      - edge weight: number of such inter-country airport-airport edges
    """
    H = nx.Graph()

    # ensure all country nodes exist
    for _, data in G.nodes(data=True):
        ctry = data.get("country","Unknown")
        H.add_node(ctry)

    # aggregate edges
    for u, v in G.edges():
        cu = G.nodes[u].get("country","Unknown")
        cv = G.nodes[v].get("country","Unknown")
        if cu == cv:
            continue
        if H.has_edge(cu, cv):
            H[cu][cv]["weight"] += 1
        else:
            H.add_edge(cu, cv, weight=1)

    return H


def compute_country_centrality(
    H: nx.Graph,
    country_counts: Dict[str,int],
    betw_sample_k: int = BETWEENNESS_SAMPLE_K,
    betw_seed: int = BETWEENNESS_SEED,
) -> pd.DataFrame:
    """
    Centrality table for countries (super-nodes).
    """
    degree_dict = dict(H.degree())
    degree_centrality = nx.degree_centrality(H)
    betweenness_centrality = nx.betweenness_centrality(
        H, k=min(betw_sample_k, len(H)), seed=betw_seed
    )
    closeness_centrality = nx.closeness_centrality(H)
    try:
        eigen_centrality = nx.eigenvector_centrality(H, max_iter=500)
    except Exception:
        eigen_centrality = {n: np.nan for n in H.nodes()}

    df_country = pd.DataFrame({
        "Country": list(H.nodes()),
        "NumAirports": [country_counts.get(n, 0) for n in H.nodes()],
        "Degree": [degree_dict[n] for n in H.nodes()],
        "DegreeCentrality": [degree_centrality[n] for n in H.nodes()],
        "Betweenness": [betweenness_centrality[n] for n in H.nodes()],
        "Closeness": [closeness_centrality[n] for n in H.nodes()],
        "Eigenvector": [eigen_centrality[n] for n in H.nodes()],
    })

    return df_country


def country_knockout_impact_light(G: nx.Graph) -> pd.DataFrame:
    """
    For each country C:
      - remove all airports in that country
      - measure how much the size of the giant component (relative to original)
        drops.
    Returns a sorted DataFrame with most critical countries first.
    """
    base_n = G.number_of_nodes()
    gcc0_size = len(max(nx.connected_components(G), key=len))
    S0 = gcc0_size / base_n

    results = []
    all_countries = sorted({
        data.get("country","Unknown")
        for _, data in G.nodes(data=True)
    })

    for c in all_countries:
        remove_nodes = [
            n for n, d in G.nodes(data=True)
            if d.get("country","Unknown") == c
        ]
        if not remove_nodes:
            continue
        H_tmp = G.copy()
        H_tmp.remove_nodes_from(remove_nodes)

        if H_tmp.number_of_nodes() == 0:
            gcc_after_size = 0
            S_after_ratio = 0.0
        else:
            gcc_after_size = len(max(nx.connected_components(H_tmp), key=len))
            S_after_ratio = gcc_after_size / base_n

        results.append({
            "Country": c,
            "RemovedAirports": len(remove_nodes),
            "GCC_nodes_after": gcc_after_size,
            "S_ratio_after": S_after_ratio,
            "Drop_in_S_ratio": S0 - S_after_ratio,
        })

    df_impact = (
        pd.DataFrame(results)
        .sort_values("Drop_in_S_ratio", ascending=False)
        .reset_index(drop=True)
    )
    return df_impact

# ----------- (Optional) Robustness curves (node removal) -----------

def sample_pairs(nodes, m, rng):
    """
    Helper for approx_efficiency(): sample m distinct node pairs.
    """
    nodes = list(nodes)
    pairs = set()
    while len(pairs) < m:
        u = rng.choice(nodes)
        v = rng.choice(nodes)
        if u != v:
            if u > v:
                u, v = v, u
            pairs.add((u, v))
    return list(pairs)


def approx_efficiency(G: nx.Graph, m: int = 2000, seed: int = 42) -> float:
    """
    Approximate network efficiency:
      average_{u!=v} 1 / dist(u,v)
    using a random sample of node pairs.
    """
    rng = np.random.default_rng(seed)
    pairs = sample_pairs(G.nodes(), m, rng)

    sp = nx.shortest_path_length
    s = 0.0
    cnt = 0
    for u, v in pairs:
        try:
            d = sp(G, source=u, target=v)
            if d > 0:
                s += 1.0 / d
                cnt += 1
        except nx.NetworkXNoPath:
            pass
    return s / max(cnt, 1)


def nodes_order_by(G: nx.Graph, key: str = 'degree', k_sample: int = BETWEENNESS_SAMPLE_K):
    """
    Pre-compute an attack order for targeted removals.
    """
    if key == 'degree':
        return [n for n,_ in sorted(G.degree(), key=lambda x:x[1], reverse=True)]
    if key == 'eigenvector':
        ec = nx.eigenvector_centrality(G, max_iter=500)
        return [n for n,_ in sorted(ec.items(), key=lambda x:x[1], reverse=True)]
    if key == 'betweenness':
        bc = nx.betweenness_centrality(G, k=min(k_sample, len(G)), seed=42)
        return [n for n,_ in sorted(bc.items(), key=lambda x:x[1], reverse=True)]
    raise ValueError(f"Unknown key '{key}'")


def nodes_order_random(G: nx.Graph, seed: int = 42):
    rng = np.random.default_rng(seed)
    arr = list(G.nodes())
    rng.shuffle(arr)
    return arr


def removal_curve(G: nx.Graph, nodes_order: List[str], qs: np.ndarray, m_pairs=2000):
    """
    Simulate removing increasing fractions of nodes (according to nodes_order).
    For each fraction q in qs:
       - remove first q*N nodes from original order
       - compute:
           * S(q): size of GCC / original N
           * E(q): approx_efficiency on GCC
           * L(q): avg shortest path length of GCC
    NOTE: This can be expensive; not run in main().
    """
    H = G.copy()
    n0 = H.number_of_nodes()
    res = []
    remove_count = 0
    order_iter = iter(nodes_order)

    for q in qs:
        target_remove = int(q*n0)
        while remove_count < target_remove:
            try:
                u = next(order_iter)
            except StopIteration:
                break
            if u in H:
                H.remove_node(u)
                remove_count += 1

        if H.number_of_nodes() == 0:
            res.append((q,0,0,np.nan))
            continue

        gcc_nodes = max(nx.connected_components(H), key=len)
        S = len(gcc_nodes)/n0
        Gsub = H.subgraph(gcc_nodes)

        E = approx_efficiency(Gsub, m=m_pairs, seed=123)
        try:
            L = nx.average_shortest_path_length(Gsub)
        except Exception:
            L = np.nan

        res.append((q,S,E,float(L)))
    return res


def plot_robustness_curves(curves_dict: Dict[str, List[Tuple[float,float,float,float]]],
                           out_png: Path) -> None:
    """
    Plot S(q) for different attack strategies on one figure.
    curves_dict[label] = [(q,S,E,L), ...]
    """
    plt.figure(figsize=(7,4))
    for lbl, rows in curves_dict.items():
        qs = [r[0] for r in rows]
        Ss = [r[1] for r in rows]
        plt.plot(qs, Ss, label=lbl)
    plt.xlabel("Removed fraction q")
    plt.ylabel("Giant component ratio S(q)")
    plt.title("Node-removal robustness")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

# ----------- Main pipeline -----------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load raw data
    airports_df, routes_df = load_raw_openflights(AIRPORTS_PATH, ROUTES_PATH)
    print(f"[LOAD] airports: {airports_df.shape}, routes: {routes_df.shape}")

    # 2. Build full graph and giant component
    G_full = build_full_graph(airports_df, routes_df)
    print(f"[GRAPH] full graph: {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges")

    G = get_giant_component(G_full)
    print(f"[GRAPH] giant component: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 3. Basic stats
    stats = compute_basic_stats(G)
    print("[STATS]")
    for k,v in stats.items():
        print(f"  {k}: {v}")

    # 4. Centrality metrics (airport-level)
    print("[CENTRALITY] computing airport-level centrality metrics...")
    centrality_df = compute_centrality_table(G)
    save_centrality_tables(
        centrality_df,
        OUTPUT_DIR / "centrality_metrics.csv",
        OUTPUT_DIR / "top10_hubs.csv",
    )
    print("[CENTRALITY] saved centrality_metrics.csv and top10_hubs.csv")

    # 5. Figures (degree dist, CCDF, betweenness hist, k-core hist)
    print("[PLOTS] generating distribution figures...")
    plot_degree_distribution(G, OUTPUT_DIR / "degree_distribution.png")
    plot_degree_ccdf(G, OUTPUT_DIR / "degree_ccdf.png")
    plot_betweenness_hist(centrality_df, OUTPUT_DIR / "betweenness_distribution.png")
    plot_kcore_hist(G, OUTPUT_DIR / "kcore_hist.png")
    # rich club omitted in default run due to possible extra cost
    # plot_rich_club(G, OUTPUT_DIR / "rich_club_curve.png")

    # 6. Assortativity + country mixing heatmap
    print("[MIXING] computing assortativity and country mixing heatmap...")
    assortativity_val = save_degree_assortativity(G, OUTPUT_DIR / "assortativity.txt")
    plot_country_mixing_heatmap(G, OUTPUT_DIR / "mixing_matrix_country.png", top_k=20)
    print(f"[MIXING] degree assortativity r={assortativity_val:.4f}")

    # 7. Export airport-level graph (giant component only)
    gml_path = OUTPUT_DIR / "global_airline_network.gml"
    nx.write_gml(G, gml_path)
    print(f"[EXPORT] wrote airport-level GML: {gml_path}")

    # 8. Country-level aggregation graph
    print("[COUNTRY] building country-level super-node graph...")
    H_country = build_country_graph(G)
    nx.write_gml(H_country, OUTPUT_DIR / "country_graph.gml")

    # 8a. Country centrality
    country_counts = Counter([
        data.get("country","Unknown")
        for _, data in G.nodes(data=True)
    ])
    df_country = compute_country_centrality(H_country, country_counts)
    df_country.to_csv(OUTPUT_DIR / "country_centrality.csv", index=False)

    # 8b. Knock-out impact (lightweight version)
    df_impact = country_knockout_impact_light(G)
    df_impact.to_csv(OUTPUT_DIR / "country_impact.csv", index=False)

    print("[COUNTRY] saved country_graph.gml, country_centrality.csv, country_impact.csv")

    # 9. Save run meta for reproducibility
    run_meta = {
        "BETWEENNESS_SAMPLE_K": BETWEENNESS_SAMPLE_K,
        "BETWEENNESS_SEED": BETWEENNESS_SEED,
        "num_nodes_giant": G.number_of_nodes(),
        "num_edges_giant": G.number_of_edges(),
        "basic_stats": stats,
    }
    with open(OUTPUT_DIR / "run_meta.json","w") as f:
        json.dump(run_meta, f, indent=2)

    # 10. Print preview: top 5 hubs by betweenness
    print("\n[TOP HUBS PREVIEW]")
    top5 = (
        centrality_df.sort_values("Betweenness", ascending=False)
        .head(5)[["Name","Country","City","IATA","Degree","Betweenness"]]
    )
    print(top5.to_string(index=False))


if __name__ == "__main__":
    main()
