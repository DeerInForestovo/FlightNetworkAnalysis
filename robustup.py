import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

G0 = nx.read_gml("./output/global_airline_network.gml")
largest_cc = max(nx.connected_components(G0), key=len)
G0 = G0.subgraph(largest_cc).copy()
n = G0.number_of_nodes()
m = G0.number_of_edges()
print(f"Baseline graph: n={n}, m={m}")

# Edge Addition
def edge_addition_strategy(G, add_frac=0.10, seed=0):
    rng = np.random.default_rng(seed)
    H = G.copy()
    num_add = int(add_frac * H.number_of_edges())
    if num_add <= 0:
        return H
    degrees = dict(H.degree())
    nodes_sorted = sorted(H.nodes(), key=lambda v: degrees[v], reverse=True)
    top_k = min(400, len(nodes_sorted))
    candidate_nodes = nodes_sorted[:top_k]
    candidate_pairs = [
        (u, v) for u in candidate_nodes for v in candidate_nodes
        if u < v and not H.has_edge(u, v)
    ]
    rng.shuffle(candidate_pairs)
    added = 0
    for u, v in candidate_pairs:
        H.add_edge(u, v)
        added += 1
        if added >= num_add:
            break
    print(f"[Edge addition] added {added} edges (target {num_add}).")
    return H

# Edge Rewiring
def edge_rewiring_strategy(G, rewiring_frac=0.20, seed=1):
    H = G.copy()
    num_swaps = int(rewiring_frac * H.number_of_edges())
    if num_swaps <= 0:
        return H
    H = nx.double_edge_swap(H, nswap=num_swaps, max_tries=num_swaps * 10, seed=seed)
    print(f"[Edge rewiring] performed {num_swaps} swaps (target frac {rewiring_frac}).")
    return H

# Mixed
def mixed_strategy(G, add_frac=0.05, rewiring_frac=0.10, seed=2):
    H = edge_addition_strategy(G, add_frac=add_frac, seed=seed)
    H = edge_rewiring_strategy(H, rewiring_frac=rewiring_frac, seed=seed + 1)
    print("[Mixed strategy] combination done.")
    return H

G_add = edge_addition_strategy(G0, add_frac=0.10, seed=42)
G_rew = edge_rewiring_strategy(G0, rewiring_frac=0.20, seed=43)
G_mix = mixed_strategy(G0, add_frac=0.05, rewiring_frac=0.10, seed=44)

# Random Failure
def robustness_curve_random_failures(G, fractions, num_trials=5, seed=0):
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    nodes = np.array(list(G.nodes()))
    S = np.zeros_like(fractions, dtype=float)
    for t in range(num_trials):
        perm = nodes.copy()
        rng.shuffle(perm)
        H = G.copy()
        removed_so_far = 0
        for i, f in enumerate(fractions):
            k_target = int(round(f * n))
            k_remove = max(0, k_target - removed_so_far)
            if k_remove > 0:
                failed = perm[removed_so_far:removed_so_far + k_remove]
                H.remove_nodes_from(failed)
                removed_so_far += k_remove
            if H.number_of_nodes() == 0:
                S[i] += 0.0
            else:
                lcc = max(nx.connected_components(H), key=len)
                S[i] += len(lcc) / n
    S /= num_trials
    return S

def lcc_curve_random_zoom(G, max_frac=0.05, steps=50, num_trials=5, seed=0):
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    nodes = np.array(list(G.nodes()))
    fractions = np.linspace(0.0, max_frac, steps + 1)
    L = np.zeros_like(fractions, dtype=float)
    for t in range(num_trials):
        perm = nodes.copy()
        rng.shuffle(perm)
        H = G.copy()
        removed_so_far = 0
        for i, f in enumerate(fractions):
            k_target = int(round(f * n))
            k_remove = max(0, k_target - removed_so_far)
            if k_remove > 0:
                failed = perm[removed_so_far:removed_so_far + k_remove]
                H.remove_nodes_from(failed)
                removed_so_far += k_remove
            if H.number_of_nodes() == 0:
                L[i] += 0.0
            else:
                lcc = max(nx.connected_components(H), key=len)
                L[i] += len(lcc)
    L /= num_trials
    return fractions, L

def betweenness_order(G, k_approx=100):
    bet = nx.betweenness_centrality(G, k=k_approx, normalized=True, seed=0)
    order = sorted(G.nodes(), key=lambda v: bet[v], reverse=True)
    return order

order_base = betweenness_order(G0,   k_approx=100)
order_add  = betweenness_order(G_add, k_approx=100)
order_rew  = betweenness_order(G_rew, k_approx=100)
order_mix  = betweenness_order(G_mix, k_approx=100)

# Betweeness-Targeted Attack
def robustness_curve_betweenness_attack_from_order(G, order, fractions):
    n = G.number_of_nodes()
    S = np.zeros_like(fractions, dtype=float)
    H = G.copy()
    removed_so_far = 0
    for i, f in enumerate(fractions):
        k_target = int(round(f * n))
        k_remove = max(0, k_target - removed_so_far)
        if k_remove > 0 and H.number_of_nodes() > 0:
            to_remove = order[removed_so_far:removed_so_far + k_remove]
            H.remove_nodes_from(to_remove)
            removed_so_far += k_remove
        if H.number_of_nodes() == 0:
            S[i] = 0.0
        else:
            lcc = max(nx.connected_components(H), key=len)
            S[i] = len(lcc) / n
    return S

def lcc_curve_betweenness_zoom_from_order(G, order, max_frac=0.05, steps=50):
    n = G.number_of_nodes()
    fractions = np.linspace(0.0, max_frac, steps + 1)
    L = np.zeros_like(fractions, dtype=float)
    H = G.copy()
    removed_so_far = 0
    for i, f in enumerate(fractions):
        k_target = int(round(f * n))
        k_remove = max(0, k_target - removed_so_far)
        if k_remove > 0 and H.number_of_nodes() > 0:
            to_remove = order[removed_so_far:removed_so_far + k_remove]
            H.remove_nodes_from(to_remove)
            removed_so_far += k_remove
        if H.number_of_nodes() == 0:
            L[i] = 0.0
        else:
            lcc = max(nx.connected_components(H), key=len)
            L[i] = len(lcc)
    return fractions, L

def compute_fc(fractions, S, threshold=0.05):
    idx = np.where(S >= threshold)[0]
    if len(idx) == 0:
        return 0.0
    return fractions[idx[-1]]


# Compute and visualize results
fractions = np.linspace(0.0, 1.0, 21)

print("Computing robustness curves (random failures)...")
S_base = robustness_curve_random_failures(G0,   fractions, num_trials=5, seed=0)
S_add  = robustness_curve_random_failures(G_add, fractions, num_trials=5, seed=1)
S_rew  = robustness_curve_random_failures(G_rew, fractions, num_trials=5, seed=2)
S_mix  = robustness_curve_random_failures(G_mix, fractions, num_trials=5, seed=3)

fc_base = compute_fc(fractions, S_base)
fc_add  = compute_fc(fractions, S_add)
fc_rew  = compute_fc(fractions, S_rew)
fc_mix  = compute_fc(fractions, S_mix)

print("Random-failure fc (S(f) >= 0.05):")
print(f"  Original      fc = {fc_base:.2f}")
print(f"  Edge addition fc = {fc_add:.2f}")
print(f"  Edge rewiring fc = {fc_rew:.2f}")
print(f"  Mixed         fc = {fc_mix:.2f}")

plt.figure(figsize=(8, 5))
plt.plot(fractions, S_base, "k--",  lw=2, label="Original")
plt.plot(fractions, S_add,  color="#1f77b4", lw=2, label="Edge addition")
plt.plot(fractions, S_rew,  color="#ff7f0e", lw=2, label="Edge rewiring")
plt.plot(fractions, S_mix,  color="#2ca02c", lw=2, label="Mixed strategy")
plt.xlabel("Fraction of failed nodes  $f$")
plt.ylabel(r"$S(f)$  (LCC size / $n$)")
plt.xlim(0, 1)
plt.ylim(0, 1.05)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("./output/robustness_random_failures_comparison.png", dpi=300)
plt.show()

print("Computing zoomed random-failure curves (0–5% range)...")
f_zoom_r, L_base_r = lcc_curve_random_zoom(G0,   max_frac=0.05, steps=50, num_trials=5, seed=0)
_,        L_add_r  = lcc_curve_random_zoom(G_add, max_frac=0.05, steps=50, num_trials=5, seed=1)
_,        L_rew_r  = lcc_curve_random_zoom(G_rew, max_frac=0.05, steps=50, num_trials=5, seed=2)
_,        L_mix_r  = lcc_curve_random_zoom(G_mix, max_frac=0.05, steps=50, num_trials=5, seed=3)

plt.figure(figsize=(8, 5))
plt.plot(f_zoom_r, L_base_r, "k--",  lw=2, label="Original (random)")
plt.plot(f_zoom_r, L_add_r,  color="#1f77b4", lw=2, label="Edge addition")
plt.plot(f_zoom_r, L_rew_r,  color="#ff7f0e", lw=2, label="Edge rewiring")
plt.plot(f_zoom_r, L_mix_r,  color="#2ca02c", lw=2, label="Mixed strategy")
plt.xlabel("Fraction of nodes removed  $f$ (random)")
plt.ylabel("LCC size")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.legend()
plt.savefig("./output/robustness_random_failures_zoom_0_05.png", dpi=300)
plt.show()

fractions = np.linspace(0.0, 1.0, 21)

print("Computing robustness curves (betweenness-targeted attacks)...")
S_base_bt = robustness_curve_betweenness_attack_from_order(G0,   order_base, fractions)
S_add_bt  = robustness_curve_betweenness_attack_from_order(G_add, order_add,  fractions)
S_rew_bt  = robustness_curve_betweenness_attack_from_order(G_rew, order_rew,  fractions)
S_mix_bt  = robustness_curve_betweenness_attack_from_order(G_mix, order_mix,  fractions)

fc_base_bt = compute_fc(fractions, S_base_bt)
fc_add_bt  = compute_fc(fractions, S_add_bt)
fc_rew_bt  = compute_fc(fractions, S_rew_bt)
fc_mix_bt  = compute_fc(fractions, S_mix_bt)

print("Betweenness-attack fc (S(f) >= 0.05):")
print(f"  Original      fc = {fc_base_bt:.2f}")
print(f"  Edge addition fc = {fc_add_bt:.2f}")
print(f"  Edge rewiring fc = {fc_rew_bt:.2f}")
print(f"  Mixed         fc = {fc_mix_bt:.2f}")

plt.figure(figsize=(8, 5))
plt.plot(fractions, S_base_bt, "k--",  lw=2, label="Original (betweenness attack)")
plt.plot(fractions, S_add_bt,  color="#1f77b4", lw=2, label="Edge addition")
plt.plot(fractions, S_rew_bt,  color="#ff7f0e", lw=2, label="Edge rewiring")
plt.plot(fractions, S_mix_bt,  color="#2ca02c", lw=2, label="Mixed strategy")
plt.xlabel("Fraction of removed nodes  $f$ (highest betweenness first)")
plt.ylabel(r"$S(f)$  (LCC size / $n$)")
plt.xlim(0, 1)
plt.ylim(0, 1.05)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("./output/robustness_betweenness_attack_comparison.png", dpi=300)
plt.show()

print("Computing zoomed betweenness-attack curves (0–5% range)...")
f_zoom_b, L_base_b = lcc_curve_betweenness_zoom_from_order(G0,   order_base, max_frac=0.05, steps=50)
_,        L_add_b  = lcc_curve_betweenness_zoom_from_order(G_add, order_add,  max_frac=0.05, steps=50)
_,        L_rew_b  = lcc_curve_betweenness_zoom_from_order(G_rew, order_rew,  max_frac=0.05, steps=50)
_,        L_mix_b  = lcc_curve_betweenness_zoom_from_order(G_mix, order_mix,  max_frac=0.05, steps=50)

plt.figure(figsize=(8, 5))
plt.plot(f_zoom_b, L_base_b, "k--",  lw=2, label="Original (betweenness attack)")
plt.plot(f_zoom_b, L_add_b,  color="#1f77b4", lw=2, label="Edge addition")
plt.plot(f_zoom_b, L_rew_b,  color="#ff7f0e", lw=2, label="Edge rewiring")
plt.plot(f_zoom_b, L_mix_b,  color="#2ca02c", lw=2, label="Mixed strategy")
plt.xlabel("Fraction of nodes removed  $f$ (highest betweenness first)")
plt.ylabel("LCC size")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.legend()
plt.savefig("./output/robustness_betweenness_attack_zoom_0_05.png", dpi=300)
plt.show()
