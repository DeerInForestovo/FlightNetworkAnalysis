import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------- Step 1: 读取网络 ----------
G = nx.read_gml("./output/global_airline_network.gml")

# 仅保留最大连通分量，避免孤立节点影响
largest_cc = max(nx.connected_components(G), key=len)
G = G.subgraph(largest_cc).copy()

print(f"Network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ---------- Step 2: 计算基础统计 ----------
avg_degree = np.mean([deg for _, deg in G.degree()])
avg_clustering = nx.average_clustering(G)
avg_path_length = nx.average_shortest_path_length(G)

print(f"Average degree: {avg_degree:.2f}")
print(f"Average clustering coefficient: {avg_clustering:.3f}")
print(f"Average path length: {avg_path_length:.2f}")

# ---------- Step 3: 计算中心性指标 ----------
print("Calculating centrality measures...")

degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G, k=500, seed=42)  # 抽样加速
closeness_centrality = nx.closeness_centrality(G)
eigen_centrality = nx.eigenvector_centrality(G, max_iter=500)

# 组合成 DataFrame
centrality_df = pd.DataFrame({
    "Airport_ID": list(G.nodes()),
    "Name": [G.nodes[n]["name"] for n in G.nodes()],
    "Country": [G.nodes[n]["country"] for n in G.nodes()],
    "Degree": [G.degree(n) for n in G.nodes()],
    "DegreeCentrality": [degree_centrality[n] for n in G.nodes()],
    "Betweenness": [betweenness_centrality[n] for n in G.nodes()],
    "Closeness": [closeness_centrality[n] for n in G.nodes()],
    "Eigenvector": [eigen_centrality[n] for n in G.nodes()]
})

# ---------- Step 4: 结果输出 ----------
centrality_df.sort_values("Betweenness", ascending=False).head(10).to_csv("./output/top10_hubs.csv", index=False)
centrality_df.to_csv("./output/centrality_metrics.csv", index=False)
print("Centrality results saved to ./output/centrality_metrics.csv")

# ---------- Step 5: 度分布图 ----------
degrees = [G.degree(n) for n in G.nodes()]
plt.figure(figsize=(8,5))
plt.hist(degrees, bins=50, color='skyblue', edgecolor='gray')
plt.xlabel("Degree")
plt.ylabel("Number of airports")
plt.title("Degree Distribution of Global Airline Network")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./output/degree_distribution.png", dpi=300)
plt.close()

# ---------- Step 6: 输出最关键机场 ----------
top5 = centrality_df.sort_values("Betweenness", ascending=False).head(5)
print("\nTop 5 hubs by betweenness centrality:")
print(top5[["Name", "Country", "Degree", "Betweenness"]])
