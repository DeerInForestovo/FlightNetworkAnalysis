import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import numpy as np

G = nx.read_gml("./output/global_airline_network.gml")
centrality_df = pd.read_csv("./output/centrality_metrics.csv")

for _, row in centrality_df.iterrows():
    node = str(row["Airport_ID"])
    if node in G.nodes:
        G.nodes[node]["Degree"] = row["Degree"]
        G.nodes[node]["Betweenness"] = row["Betweenness"]
        G.nodes[node]["Closeness"] = row["Closeness"]
        G.nodes[node]["Eigenvector"] = row["Eigenvector"]

betweenness = centrality_df["Betweenness"]
betweenness_scaled = np.log1p(betweenness * 1e4)

lats, lons, sizes, colors, texts = [], [], [], [], []

for node, data in G.nodes(data=True):
    lats.append(float(data.get("lat", 0)))
    lons.append(float(data.get("lon", 0)))
    deg = data.get("Degree", 1)
    sizes.append(3 + np.sqrt(deg) * 1.5)
    colors.append(betweenness_scaled.loc[centrality_df["Airport_ID"].astype(str) == str(node)].values[0]
                  if str(node) in centrality_df["Airport_ID"].astype(str).values
                  else 0)
    texts.append(f"{data.get('name','Unknown')} ({data.get('country','')})<br>"
                 f"Degree: {deg}<br>"
                 f"Betweenness: {data.get('Betweenness',0):.4f}")

edge_lats, edge_lons = [], []
for u, v in G.edges():
    u_lat, u_lon = G.nodes[u]["lat"], G.nodes[u]["lon"]
    v_lat, v_lon = G.nodes[v]["lat"], G.nodes[v]["lon"]
    edge_lats += [u_lat, v_lat, None]
    edge_lons += [u_lon, v_lon, None]

edge_trace = go.Scattergeo(
    lon=edge_lons, lat=edge_lats,
    mode='lines',
    line=dict(width=0.3, color='rgba(120,120,120,0.3)'),
    hoverinfo='none'
)

node_trace = go.Scattergeo(
    lon=lons, lat=lats,
    mode='markers',
    marker=dict(
        size=sizes,
        color=colors,
        colorscale='Turbo',
        cmin=min(colors),
        cmax=max(colors),
        colorbar=dict(title="Log-scaled Betweenness"),
        line=dict(width=0.3, color='gray'),
        opacity=0.85
    ),
    text=texts,
    hoverinfo='text'
)

fig = go.Figure(data=[edge_trace, node_trace])
fig.update_layout(
    title="Global Airline Network — Centrality Visualization",
    geo=dict(
        scope='world',
        projection_type='natural earth',
        showland=True,
        landcolor='rgb(240,240,240)',
        coastlinecolor='gray',
        showcountries=True,
        countrycolor='gray',
    ),
    margin=dict(l=0, r=0, t=40, b=0)
)

fig.write_html("./output/global_airline_centrality_map.html")
print("Visualization saved to ./output/global_airline_centrality_map.html")
