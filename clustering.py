import networkx as nx
import pandas as pd
import plotly.graph_objects as go

# ---------- Step 1: 载入网络和分析结果 ----------
G = nx.read_gml("./output/global_airline_network.gml")
centrality_df = pd.read_csv("./output/centrality_metrics.csv")

# 将中心性信息添加回图节点
for _, row in centrality_df.iterrows():
    node = str(row["Airport_ID"])
    if node in G.nodes:
        G.nodes[node]["Degree"] = row["Degree"]
        G.nodes[node]["Betweenness"] = row["Betweenness"]
        G.nodes[node]["Closeness"] = row["Closeness"]
        G.nodes[node]["Eigenvector"] = row["Eigenvector"]

# ---------- Step 2: 准备节点数据 ----------
lats, lons, sizes, texts = [], [], [], []

for node, data in G.nodes(data=True):
    lats.append(float(data.get("lat", 0)))
    lons.append(float(data.get("lon", 0)))
    size = data.get("Degree", 1)
    sizes.append(2 + size * 0.05)  # 调整比例
    texts.append(f"{data.get('name','Unknown')} ({data.get('country','')})<br>"
                 f"Degree: {data.get('Degree',0)}<br>"
                 f"Betweenness: {data.get('Betweenness',0):.4f}")

# ---------- Step 3: 准备航线（边） ----------
edge_lats, edge_lons = [], []
for u, v in G.edges():
    u_lat, u_lon = G.nodes[u]["lat"], G.nodes[u]["lon"]
    v_lat, v_lon = G.nodes[v]["lat"], G.nodes[v]["lon"]
    edge_lats += [u_lat, v_lat, None]
    edge_lons += [u_lon, v_lon, None]

# ---------- Step 4: 使用 Plotly 绘制 ----------
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
        color=centrality_df["Betweenness"],
        colorscale='YlOrRd',
        cmin=centrality_df["Betweenness"].min(),
        cmax=centrality_df["Betweenness"].max(),
        colorbar=dict(title="Betweenness"),
        line=dict(width=0.2, color='gray'),
        opacity=0.8
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

# ---------- Step 5: 输出 HTML ----------
fig.write_html("./output/global_airline_centrality_map.html")
print("Visualization saved to ./output/global_airline_centrality_map.html")
