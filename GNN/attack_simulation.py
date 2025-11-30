import networkx as nx
import random
from visualize_attack import visualize_network

ATTACK_MODE = "random"   # "random" or "targeted"
STEP_RATIO = 0.001           # 每一步攻击 0.1%
STEPS = 50

G = nx.read_gml("./output/global_airline_network.gml")

lcc_sizes = []

for step in range(STEPS):

    num_remove = int(len(G.nodes()) * STEP_RATIO)

    if ATTACK_MODE == "random":
        remove_nodes = random.sample(list(G.nodes()), num_remove)

    elif ATTACK_MODE == "targeted":
        degree_dict = dict(G.degree())
        remove_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:num_remove]

    G.remove_nodes_from(remove_nodes)

    if len(G) == 0:
        break

    largest_cc = max(nx.connected_components(G), key=len)
    lcc_sizes.append(len(largest_cc))

    # 可视化关键节点
    if step in [0, 4, 9]:
        visualize_network(G,
            f"./output/{ATTACK_MODE}_step_{step}.html",
            f"{ATTACK_MODE.capitalize()} attack – step {step}")

print(f"{ATTACK_MODE} LCC sizes:", lcc_sizes)
