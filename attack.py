# robustness_simulation.py
import networkx as nx
import pandas as pd
import numpy as np
import random
import time
import plotly.graph_objects as go
from copy import deepcopy
from collections import defaultdict

# ---------- Parameters ----------
G_PATH = "./output/global_airline_network.gml"
OUTPUT_PREFIX = "./output/robustness"
RANDOM_TRIALS = 1          # 随机攻击重复次数（取平均）
REMOVAL_STEPS = 3          # 将删除过程划分成多少个采样点（越多越精细）
MAX_REMOVE_FRAC = 0.1       # 最多删除比例（0~1）
ADAPTIVE_TARGETED = False   # 是否在每次删除后重新计算 centrality（非常耗时）
SEED = 42
# --------------------------------

random.seed(SEED)
np.random.seed(SEED)

def load_graph(path):
    G = nx.read_gml(path)
    # ensure node ids are consistent type (use strings)
    # networkx may store node ids as strings after GML read, so we keep them as-is.
    return G

def metrics_on_graph(G):
    """Compute summary metrics; returns dict.
       For path length and efficiency, use LCC only where appropriate."""
    metrics = {}
    metrics['n_nodes'] = G.number_of_nodes()
    metrics['n_edges'] = G.number_of_edges()
    if metrics['n_nodes'] == 0:
        metrics.update({
            'lcc_size': 0, 'lcc_fraction': 0.0,
            'avg_path_length': np.nan, 'avg_clustering': np.nan,
            'global_efficiency': 0.0, 'n_components': 0
        })
        return metrics

    comps = list(nx.connected_components(G))
    metrics['n_components'] = len(comps)
    lcc = max(comps, key=len)
    lcc_sub = G.subgraph(lcc)
    metrics['lcc_size'] = lcc_sub.number_of_nodes()
    metrics['lcc_fraction'] = metrics['lcc_size'] / metrics['n_nodes'] if metrics['n_nodes']>0 else 0.0
    metrics['avg_clustering'] = nx.average_clustering(G)
    # avg shortest path length only meaningful if LCC has >1 node
    if metrics['lcc_size'] > 1:
        try:
            metrics['avg_path_length'] = nx.average_shortest_path_length(lcc_sub)
        except Exception:
            metrics['avg_path_length'] = np.nan
    else:
        metrics['avg_path_length'] = np.nan
    # global efficiency works on the whole graph; networkx has global_efficiency
    try:
        metrics['global_efficiency'] = nx.global_efficiency(G)
    except Exception:
        metrics['global_efficiency'] = 0.0
    return metrics

def removal_sequence_by_fraction(nodes_list, steps, max_frac):
    """Return a list of integers: how many nodes to remove at each step (cumulative)."""
    n = len(nodes_list)
    max_remove = int(np.floor(n * max_frac))
    # cumulative numbers: 0, ceil(max_remove/steps), 2*..., ..., max_remove
    cum = [int(round(i * max_remove / steps)) for i in range(0, steps+1)]
    # ensure strictly non-decreasing and unique
    cum = sorted(list(dict.fromkeys(cum)))
    return cum

def simulate_random_attacks(G, trials=20, steps=50, max_frac=0.9):
    nodes = list(G.nodes())
    n = len(nodes)
    cum_removes = removal_sequence_by_fraction(nodes, steps, max_frac)
    results = []
    for t in range(trials):
        G0 = G.copy()
        perm = nodes.copy()
        random.shuffle(perm)
        removed_set = set()
        # we will remove nodes in order of perm; at each cum threshold compute metrics
        for k in cum_removes:
            # need to ensure we have removed 'k' nodes
            while len(removed_set) < k:
                node_to_remove = perm[len(removed_set)]
                removed_set.add(node_to_remove)
                if G0.has_node(node_to_remove):
                    G0.remove_node(node_to_remove)
            m = metrics_on_graph(G0)
            m.update({'trial': t, 'removed_count': k, 'removed_frac': k / n})
            results.append(m)
    df = pd.DataFrame(results)
    return df

def simulate_targeted_attack(G, key='degree', adaptive=False, steps=50, max_frac=0.9):
    """key can be 'degree' or 'betweenness' or 'closeness' or 'eigenvector'.
       adaptive=True means recompute centrality after each removal step (expensive)."""
    nodes = list(G.nodes())
    n = len(nodes)
    cum_removes = removal_sequence_by_fraction(nodes, steps, max_frac)
    results = []
    G_work = G.copy()

    # order initially
    if key == 'degree':
        ordering = sorted(G_work.degree(), key=lambda x: x[1], reverse=True)
        order_list = [n for n,_ in ordering]
    else:
        # compute centrality once (may be expensive for betweenness)
        if key == 'betweenness':
            centrality = nx.betweenness_centrality(G_work, normalized=True)
        elif key == 'closeness':
            centrality = nx.closeness_centrality(G_work)
        elif key == 'eigenvector':
            try:
                centrality = nx.eigenvector_centrality_numpy(G_work)
            except Exception:
                centrality = nx.eigenvector_centrality(G_work, max_iter=200)
        else:
            raise ValueError("Unknown key for targeted attack.")
        # sort descending
        order_list = sorted(centrality.keys(), key=lambda x: centrality[x], reverse=True)

    removed = set()
    current_order = order_list.copy()

    # iterate cumulative removals
    for k in cum_removes:
        # remove nodes until removed_count == k
        while len(removed) < k and len(current_order) > 0:
            # pick next target
            target = current_order[0]
            current_order = current_order[1:]
            if G_work.has_node(target):
                G_work.remove_node(target)
                removed.add(target)
            # if adaptive, recompute ordering after each removal (or after small batch)
            if adaptive:
                # recompute ordering on remaining graph
                if key == 'degree':
                    current_order = [n for n,_ in sorted(G_work.degree(), key=lambda x: x[1], reverse=True)
                                     if n not in removed]
                else:
                    # recompute chosen centrality
                    if key == 'betweenness':
                        centrality = nx.betweenness_centrality(G_work, normalized=True)
                    elif key == 'closeness':
                        centrality = nx.closeness_centrality(G_work)
                    elif key == 'eigenvector':
                        try:
                            centrality = nx.eigenvector_centrality_numpy(G_work)
                        except Exception:
                            centrality = nx.eigenvector_centrality(G_work, max_iter=200)
                    current_order = sorted(centrality.keys(), key=lambda x: centrality[x], reverse=True)
                    # filter removed
                    current_order = [n for n in current_order if n not in removed]

        m = metrics_on_graph(G_work)
        m.update({'method': f"targeted_{key}", 'adaptive': adaptive, 'removed_count': k, 'removed_frac': k/n})
        results.append(m)

    df = pd.DataFrame(results)
    return df

def aggregate_random_df(df_random):
    """Aggregate random trials: compute mean and std per removed_frac"""
    grouped = df_random.groupby('removed_frac').agg({
        'lcc_fraction': ['mean','std'],
        'avg_path_length': ['mean','std'],
        'global_efficiency': ['mean','std'],
        'n_components': ['mean','std']
    })
    # flatten columns
    grouped.columns = ['_'.join(c).strip() for c in grouped.columns.values]
    grouped = grouped.reset_index()
    return grouped

def make_plot(df_random_agg, df_deg, df_bet, out_html=OUTPUT_PREFIX + "_lcc.html"):
    # Build line chart for LCC fraction
    fig = go.Figure()
    # random mean
    fig.add_trace(go.Scatter(
        x=df_random_agg['removed_frac'],
        y=df_random_agg['lcc_fraction_mean'],
        mode='lines',
        name='Random (mean)',
        line=dict(width=2)
    ))
    # random +/- std as shaded area
    fig.add_trace(go.Scatter(
        x=np.concatenate([df_random_agg['removed_frac'], df_random_agg['removed_frac'][::-1]]),
        y=np.concatenate([df_random_agg['lcc_fraction_mean'] + df_random_agg['lcc_fraction_std'],
                          (df_random_agg['lcc_fraction_mean'] - df_random_agg['lcc_fraction_std'])[::-1]]),
        fill='toself',
        fillcolor='rgba(0,100,80,0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=True,
        name='Random ± std'
    ))
    # targeted degree
    fig.add_trace(go.Scatter(
        x=df_deg['removed_frac'],
        y=df_deg['lcc_fraction'],
        mode='lines+markers',
        name='Targeted (degree)',
        line=dict(dash='dash')
    ))
    # targeted betweenness
    fig.add_trace(go.Scatter(
        x=df_bet['removed_frac'],
        y=df_bet['lcc_fraction'],
        mode='lines+markers',
        name='Targeted (betweenness)',
        line=dict(dash='dot')
    ))

    fig.update_layout(
        title="Network Robustness: Largest Connected Component Fraction vs Fraction Nodes Removed",
        xaxis_title="Fraction of nodes removed",
        yaxis_title="LCC size (fraction of remaining nodes)",
        legend=dict(x=0.02, y=0.98),
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40)
    )
    fig.write_html(out_html)
    print(f"Plot saved to {out_html}")
    return fig

def main():
    print("Loading graph...")
    G = load_graph(G_PATH)
    print(f"Graph: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    # Basic initial metrics
    baseline = metrics_on_graph(G)
    print("Baseline metrics:", baseline)

    # --- Random attacks ---
    print("Simulating random attacks...")
    t0 = time.time()
    df_rand = simulate_random_attacks(G, trials=RANDOM_TRIALS, steps=REMOVAL_STEPS, max_frac=MAX_REMOVE_FRAC)
    t1 = time.time()
    print(f"Random sims done in {t1-t0:.1f}s, rows={len(df_rand)}")
    df_rand.to_csv(OUTPUT_PREFIX + "_random_trials.csv", index=False)

    df_rand_agg = aggregate_random_df(df_rand)
    df_rand_agg.to_csv(OUTPUT_PREFIX + "_random_agg.csv", index=False)

    # --- Targeted attacks (static ordering) ---
    print("Simulating targeted attack: degree...")
    df_deg = simulate_targeted_attack(G, key='degree', adaptive=False, steps=REMOVAL_STEPS, max_frac=MAX_REMOVE_FRAC)
    df_deg.to_csv(OUTPUT_PREFIX + "_targeted_degree.csv", index=False)

    print("Simulating targeted attack: betweenness (static)...")
    # betweenness may be slow, but static once is doable for moderate graphs
    df_bet = simulate_targeted_attack(G, key='betweenness', adaptive=ADAPTIVE_TARGETED, steps=REMOVAL_STEPS, max_frac=MAX_REMOVE_FRAC)
    df_bet.to_csv(OUTPUT_PREFIX + "_targeted_betweenness.csv", index=False)

    # Optionally simulate adaptive targeted (warning: very slow)
    if ADAPTIVE_TARGETED:
        print("Simulating adaptive targeted (degree) -- recomputing after removals...")
        df_deg_adapt = simulate_targeted_attack(G, key='degree', adaptive=True, steps=REMOVAL_STEPS, max_frac=MAX_REMOVE_FRAC)
        df_deg_adapt.to_csv(OUTPUT_PREFIX + "_targeted_degree_adaptive.csv", index=False)
        print("Adaptive degree saved.")

    # --- Plot ---
    print("Making plot...")
    make_plot(df_rand_agg, df_deg, df_bet, out_html=OUTPUT_PREFIX + "_lcc.html")

    print("All done. Results saved with prefix:", OUTPUT_PREFIX)

if __name__ == "__main__":
    main()
