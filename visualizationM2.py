"""
Visualization of the global airline network:
- Betweenness-colored network map
- k-core map
- Community map
"""

import plotly.graph_objects as go
import networkx as nx
from networkx.algorithms import community
import numpy as np

G = nx.read_gml("./output/global_airline_network.gml")

largest_cc = max(nx.connected_components(G), key=len)
G = G.subgraph(largest_cc).copy()

nodes = list(G.nodes())

lats = [G.nodes[n]["lat"] for n in nodes]
lons = [G.nodes[n]["lon"] for n in nodes]
names = [G.nodes[n]["name"] for n in nodes]

edge_lats = []
edge_lons = []
for u, v in G.edges():
    edge_lats += [G.nodes[u]["lat"], G.nodes[v]["lat"], None]
    edge_lons += [G.nodes[u]["lon"], G.nodes[v]["lon"], None]


def make_world_base_figure(title: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scattergeo(
        lon=edge_lons,
        lat=edge_lats,
        mode="lines",
        line=dict(width=0.4, color="rgba(120,120,120,0.18)"),
        hoverinfo="none"
    ))

    fig.update_layout(
        title=title,
        showlegend=False,
        geo=dict(
            projection_type="natural earth",
            showland=True,
            landcolor="rgb(235,235,235)",
            countrycolor="rgb(210,210,210)",
            coastlinecolor="rgb(170,170,170)",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ============================
# Betweenness-colored map
# ============================
print("Computing betweenness centrality ...")
bet = nx.betweenness_centrality(G, normalized=True)
bet_arr = np.array([bet[n] for n in nodes])

nonzero = bet_arr[bet_arr > 0]
if len(nonzero) > 0:
    vmax = np.quantile(nonzero, 0.99)
else:
    vmax = bet_arr.max()

bet_clipped = np.clip(bet_arr, 0, vmax)

sizes = 3 + 10 * (bet_clipped / (vmax + 1e-9))

single_hue_scale = [
    [0.0, "rgba(255,235,235,0.3)"],
    [1.0, "rgba(139,0,0,1.0)"],
]

fig_bet = make_world_base_figure("Global Airline Network - Betweenness Centrality")

fig_bet.add_trace(go.Scattergeo(
    lon=lons,
    lat=lats,
    text=[f"{names[i]}<br>betweenness={bet_arr[i]:.4g}" for i in range(len(nodes))],
    mode="markers",
    marker=dict(
        size=sizes,
        color=bet_clipped,
        colorscale=single_hue_scale,
        cmin=0,
        cmax=vmax,
        colorbar=dict(title="Betweenness"),
        opacity=0.9,
    ),
))

fig_bet.write_html("./output/global_airline_betweenness_map.html")
print("Saved betweenness map to ./output/global_airline_betweenness_map.html")


# ============================
# k-core map
# ============================
print("Computing k-core numbers ...")
core_num = nx.core_number(G)
core_values = [core_num[n] for n in nodes]

fig_kcore = make_world_base_figure("Global Airline Network - k-core Map")

fig_kcore.add_trace(go.Scattergeo(
    lon=lons,
    lat=lats,
    text=[f"{names[i]}<br>k-core index={core_values[i]}" for i in range(len(nodes))],
    mode="markers",
    marker=dict(
        size=4,
        color=core_values,
        colorscale="Plasma",
        colorbar=dict(title="k-core index"),
        opacity=0.85,
    ),
))

fig_kcore.write_html("./output/global_airline_kcore_map.html")
print("Saved k-core map to ./output/global_airline_kcore_map.html")


# ============================
# Community map
# ============================
print("Detecting communities (greedy modularity) ...")
communities = list(community.greedy_modularity_communities(G))

comm_id = {}
for cid, comm_nodes in enumerate(communities):
    for n in comm_nodes:
        comm_id[n] = cid

comm_values = [comm_id[n] for n in nodes]

fig_comm = make_world_base_figure("Global Airline Network - Community Map")

fig_comm.add_trace(go.Scattergeo(
    lon=lons,
    lat=lats,
    text=[f"{names[i]}<br>community={comm_values[i]}" for i in range(len(nodes))],
    mode="markers",
    marker=dict(
        size=4,
        color=comm_values,
        colorscale="Rainbow",
        colorbar=dict(title="Community ID"),
        opacity=0.85,
    ),
))

fig_comm.write_html("./output/global_airline_community_map.html")
print("Saved community map to ./output/global_airline_community_map.html")
