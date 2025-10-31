#!/usr/bin/env python3
"""
Interactive world maps for the global airline network.

This script:
  - loads airport-level graph (global_airline_network.gml)
    and centrality_metrics.csv
  - attaches metrics back to node attributes
  - computes k-core and community assignments
  - generates interactive Plotly maps:
      * map_betweenness.html  (color = betweenness, continuous scale)
      * map_kcore.html        (color = k-core index, continuous scale)
      * map_community.html    (color = community id, DISTINCT categorical colors)

Each map shows:
  - gray edges for routes
  - node size ~ sqrt(degree)
  - node color ~ chosen metric
  - hover text with airport info
  - small text labels for top bridging hubs
"""

from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from networkx.algorithms.community import greedy_modularity_communities

OUTPUT_DIR = Path("./data/output")
GML_PATH = OUTPUT_DIR / "global_airline_network.gml"
CENTRALITY_CSV = OUTPUT_DIR / "centrality_metrics.csv"


def load_network_with_metrics(
    gml_path: Path,
    centrality_csv: Path
):
    """
    Load graph and join per-airport centrality metrics.
    Returns (G, df_metrics).
    """
    G = nx.read_gml(str(gml_path))
    df = pd.read_csv(centrality_csv)

    # Attach metrics to nodes so plotting is easy
    for _, row in df.iterrows():
        node_id = str(row["Airport_ID"])
        if node_id in G.nodes:
            G.nodes[node_id]["Degree"]      = row["Degree"]
            G.nodes[node_id]["Betweenness"] = row["Betweenness"]
            G.nodes[node_id]["Closeness"]   = row["Closeness"]
            G.nodes[node_id]["Eigenvector"] = row["Eigenvector"]
            G.nodes[node_id]["Name"]        = row["Name"]
            G.nodes[node_id]["Country"]     = row["Country"]
            G.nodes[node_id]["City"]        = row["City"]
            G.nodes[node_id]["IATA"]        = row["IATA"]

    return G, df


def assign_kcore(G: nx.Graph):
    """
    Compute each node's k-core index and store in node attribute 'kcore'.
    """
    cores = nx.core_number(G)
    nx.set_node_attributes(G, cores, "kcore")


def assign_community(G: nx.Graph):
    """
    Run greedy modularity community detection.
    Each airport node gets an integer community ID (0,1,2,...) stored in 'community'.
    """
    comms = list(greedy_modularity_communities(G))
    comm_id = {}
    for i, cset in enumerate(comms):
        for n in cset:
            comm_id[n] = i
    print(f"[COMMUNITY] Detected {len(comms)} communities.")
    nx.set_node_attributes(G, comm_id, "community")


def build_edge_trace(G: nx.Graph) -> go.Scattergeo:
    """
    Draw thin gray flight edges.
    """
    lats, lons = [], []
    for u, v in G.edges():
        u_lat = float(G.nodes[u].get("lat", np.nan))
        u_lon = float(G.nodes[u].get("lon", np.nan))
        v_lat = float(G.nodes[v].get("lat", np.nan))
        v_lon = float(G.nodes[v].get("lon", np.nan))
        if np.isnan(u_lat) or np.isnan(u_lon) or np.isnan(v_lat) or np.isnan(v_lon):
            continue
        lats += [u_lat, v_lat, None]
        lons += [u_lon, v_lon, None]

    return go.Scattergeo(
        lon=lons,
        lat=lats,
        mode="lines",
        line=dict(width=0.3, color="rgba(120,120,120,0.3)"),
        hoverinfo="none",
        showlegend=False,
    )


def _compute_continuous_colors(G: nx.Graph, nodes: list, color_by: str):
    """
    For numeric metrics (Betweenness, kcore, etc.), produce
    normalized [0,1] values + a colorbar title.
    """
    raw = np.array([G.nodes[n].get(color_by, 0.0) for n in nodes], dtype=float)

    if color_by == "Betweenness":
        # compress dynamic range so you can actually see mid-tier hubs
        scaled = np.log1p(raw * 1e4)
        mn, mx = scaled.min(), scaled.max()
        normed = (scaled - mn) / (mx - mn + 1e-9)
        colorbar_title = "Betweenness (log-scaled)"
    else:
        # default min-max
        mn, mx = raw.min(), raw.max()
        normed = (raw - mn) / (mx - mn + 1e-9)
        colorbar_title = color_by

    return normed, colorbar_title


def _compute_categorical_colors(G: nx.Graph, nodes: list, color_by: str):
    """
    For categorical metrics (like 'community'), assign a distinct color
    to each unique community ID.

    Returns:
      color_strings: list[str] same length as nodes,
                     e.g. ["#1f77b4", "#ff7f0e", ...]
      cmap: {community_id -> hex_color}
    """
    # A long qualitative palette we can cycle through if > len(palette) communities.
    palette = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
        "#393b79","#637939","#8c6d31","#843c39","#7b4173",
        "#3182bd","#6baed6","#9ecae1","#c6dbef","#e6550d",
        "#fd8d3c","#fdae6b","#fdd0a2","#31a354","#74c476",
        "#a1d99b","#e7ba52","#c7c7c7","#ad494a","#9edae5",
        "#e7969c","#c49c94","#e7cb94","#c7e9c0","#a2dba9",
        "#7bccc4","#43a2ca","#0868ac","#810f77","#8856a7",
    ]

    vals = [G.nodes[n].get(color_by, -1) for n in nodes]
    unique_vals = sorted(set(vals))
    cmap = {}
    for idx, val in enumerate(unique_vals):
        cmap[val] = palette[idx % len(palette)]

    color_strings = [cmap[v] for v in vals]
    return color_strings, cmap


def build_node_trace(
    G: nx.Graph,
    color_by: str,
    size_by: str = "Degree"
) -> go.Scattergeo:
    """
    Build the airport layer:
      - marker size ~ sqrt(degree)
      - marker color:
          * continuous colorscale (Viridis + colorbar)
            for numeric metrics (Betweenness, kcore)
          * distinct category colors (no colorscale, no colorbar)
            for 'community'
    """
    nodes = list(G.nodes())

    # We'll build arrays for plotting
    lats, lons, sizes, texts = [], [], [], []
    colors_plotted = []

    # Precompute colors differently depending on metric
    categorical_mode = (color_by == "community")
    if categorical_mode:
        color_strings, _legend_map = _compute_categorical_colors(G, nodes, color_by)
    else:
        normed_vals, colorbar_title = _compute_continuous_colors(G, nodes, color_by)

    for i, n in enumerate(nodes):
        data = G.nodes[n]
        lat = float(data.get("lat", np.nan))
        lon = float(data.get("lon", np.nan))
        if np.isnan(lat) or np.isnan(lon):
            continue

        deg_val = data.get(size_by, 1.0)
        marker_size = 3 + (np.sqrt(deg_val) * 1.5)

        lats.append(lat)
        lons.append(lon)
        sizes.append(marker_size)

        # choose point color
        if categorical_mode:
            colors_plotted.append(color_strings[i])
        else:
            colors_plotted.append(normed_vals[i])

        # hover text
        texts.append(
            f"{data.get('Name','Unknown')} ({data.get('Country','')})<br>"
            f"{data.get('City','')} {data.get('IATA','')}<br>"
            f"Degree: {data.get('Degree','?')}<br>"
            f"Betweenness: {data.get('Betweenness',0):.4f}<br>"
            f"kcore: {data.get('kcore','?')} community: {data.get('community','?')}"
        )

    # marker config:
    if categorical_mode:
        # distinct fixed colors; do NOT attach colorscale/colorbar
        marker_dict = dict(
            size=sizes,
            color=colors_plotted,
            line=dict(width=0.3, color="black"),
            opacity=0.85,
        )
    else:
        # continuous colorscale with colorbar
        marker_dict = dict(
            size=sizes,
            color=colors_plotted,
            colorscale="Viridis",
            cmin=0,
            cmax=1,
            colorbar=dict(
                title=colorbar_title,
                len=0.4,
            ),
            line=dict(width=0.3, color="black"),
            opacity=0.85,
        )

    node_trace = go.Scattergeo(
        lon=lons,
        lat=lats,
        mode="markers",
        marker=marker_dict,
        text=texts,
        hoverinfo="text",
        showlegend=False,
        name="Airports",
    )
    return node_trace


def build_label_trace(
    G: nx.Graph,
    df_metrics: pd.DataFrame,
    sort_by: str = "Betweenness",
    k: int = 10
) -> go.Scattergeo:
    """
    Add small text labels for top-k hubs by a given metric (Betweenness by default).
    """
    topk = df_metrics.sort_values(sort_by, ascending=False).head(k)
    label_lats, label_lons, label_txt = [], [], []

    for _, row in topk.iterrows():
        node_id = str(row["Airport_ID"])
        if node_id not in G.nodes:
            continue
        data = G.nodes[node_id]
        lat = float(data.get("lat", np.nan))
        lon = float(data.get("lon", np.nan))
        if np.isnan(lat) or np.isnan(lon):
            continue
        label_lats.append(lat)
        label_lons.append(lon)
        label_txt.append(f"{data.get('Name','Unknown')}")

    label_trace = go.Scattergeo(
        lon=label_lons,
        lat=label_lats,
        mode="text",
        text=label_txt,
        textfont=dict(size=9, color="black"),
        hoverinfo="none",
        showlegend=False,
        name="Top hubs",
    )
    return label_trace


def make_figure(
    G: nx.Graph,
    df_metrics: pd.DataFrame,
    color_by: str,
    title: str
):
    """
    Compose edge layer, node layer, and labels into a world map.
    """
    edge_trace = build_edge_trace(G)
    node_trace = build_node_trace(G, color_by=color_by, size_by="Degree")
    label_trace = build_label_trace(G, df_metrics, sort_by="Betweenness", k=10)

    fig = go.Figure(data=[edge_trace, node_trace, label_trace])
    fig.update_layout(
        title=title,
        geo=dict(
            scope="world",
            projection_type="natural earth",
            showland=True,
            landcolor="rgb(240,240,240)",
            showcountries=True,
            countrycolor="gray",
            coastlinecolor="gray",
            lataxis=dict(range=[-60, 85]),  # trim Antarctica for cleaner frame
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. load airport-level graph and metrics
    G, df_metrics = load_network_with_metrics(GML_PATH, CENTRALITY_CSV)

    # 2. assign structural attributes
    assign_kcore(G)
    assign_community(G)

    # 3. export maps
    # (1) betweenness hubs: continuous colorscale, colorbar = betweenness
    fig_betw = make_figure(
        G, df_metrics,
        color_by="Betweenness",
        title="Global Airline Network — Betweenness hubs"
    )
    fig_betw.write_html(str(OUTPUT_DIR / "map_betweenness.html"))

    # (2) k-core structure: continuous colorscale, colorbar = kcore
    fig_kcore = make_figure(
        G, df_metrics,
        color_by="kcore",
        title="Global Airline Network — k-core structure"
    )
    fig_kcore.write_html(str(OUTPUT_DIR / "map_kcore.html"))

    # (3) community structure: CATEGORICAL COLORS, one unique color per community
    fig_comm = make_figure(
        G, df_metrics,
        color_by="community",
        title="Global Airline Network — community structure"
    )
    fig_comm.write_html(str(OUTPUT_DIR / "map_community.html"))

    print("[MAP] Saved map_betweenness.html, map_kcore.html, map_community.html (community = categorical colors)")


if __name__ == "__main__":
    main()
