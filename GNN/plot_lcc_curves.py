import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./output/lcc_summary.csv")

plt.figure(figsize=(10, 6))

for attack_type in df["attack_type"].unique():
    sub = df[df["attack_type"] == attack_type]
    plt.plot(
        sub["removed_percent"],
        sub["lcc_fraction"],
        label=attack_type
    )

plt.xlabel("Fraction of nodes removed")
plt.ylabel("LCC fraction")
plt.title("Robustness of Global Airline Network under Different Attacks")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("./output/lcc_attack_comparison.png", dpi=300)
plt.show()
