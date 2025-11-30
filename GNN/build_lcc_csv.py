import numpy as np
import pandas as pd
import os

INIT_NODES = 3188

files = {
    "random": "./output/attack_random_lcc.npy",
    "hub": "./output/attack_hub_lcc.npy",
    "static_bc": "./output/attack_static_bc_lcc.npy",
    "gnn_static": "./output/attack_gnn_static_lcc.npy",
    "gnn_adaptive": "./output/attack_gnn_adaptive_lcc.npy",
}

records = []

for attack_type, path in files.items():
    lcc_history = np.load(path)

    for step, lcc in enumerate(lcc_history):
        removed = step * 3
        removed_percent = removed / INIT_NODES

        records.append({
            "attack_type": attack_type,
            "step": step,
            "removed_percent": removed_percent,
            "lcc_fraction": lcc
        })

df = pd.DataFrame(records)
df.to_csv("./output/lcc_summary.csv", index=False)

print("CSV saved to ./output/lcc_summary.csv")
print(df.head())
