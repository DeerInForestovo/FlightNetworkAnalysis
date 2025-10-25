import plotly.graph_objects as go
import networkx as nx
import pandas as pd

G = nx.read_gml("./output/global_airline_network.gml")

largest_cc = max(nx.connected_components(G), key=len)
G = G.subgraph(largest_cc)

lats = [G.nodes[n]["lat"] for n in G.nodes()]
lons = [G.nodes[n]["lon"] for n in G.nodes()]
names = [G.nodes[n]["name"] for n in G.nodes()]

edge_lats = []
edge_lons = []
for u, v in G.edges():
    edge_lats += [G.nodes[u]["lat"], G.nodes[v]["lat"], None]
    edge_lons += [G.nodes[u]["lon"], G.nodes[v]["lon"], None]

fig = go.Figure()

fig.add_trace(go.Scattergeo(
    lon=edge_lons,
    lat=edge_lats,
    mode="lines",
    line=dict(width=0.5, color="rgba(100,100,100,0.3)"),
    hoverinfo="none"
))

fig.add_trace(go.Scattergeo(
    lon=lons,
    lat=lats,
    text=names,
    mode="markers",
    marker=dict(size=3, color="red", opacity=0.6),
))

fig.update_layout(
    title="Global Airline Network",
    showlegend=False,
    geo=dict(
        projection_type="natural earth",
        showland=True, landcolor="rgb(230,230,230)",
        countrycolor="rgb(200,200,200)",
        coastlinecolor="rgb(150,150,150)"
    )
)

fig.write_html("./output/global_airline_network.html")
