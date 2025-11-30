# FlightNetworkAnalysis

This repository contains code for analyzing the global airline network and studying network robustness under node/edge removal attacks. The project is organized into two phases:

- Phase 1 (classical graph analysis) — scripts in the repository root
- Phase 2 (GNN-based experiments) — scripts under the `GNN/` directory

## Phase 1 — classical graph analysis (root scripts)

- `network_build.py`
    - Build the global airline network from raw OpenFlights data files. Produces a NetworkX graph you can save or inspect.

- `visualization.py`
    - Create interactive HTML visualizations (Plotly) of the network or node-level centrality maps.

- `analysis.py`
    - Compute centrality measures and basic analyses (degree, betweenness, coreness, etc.). Contains helper routines used by other scripts.

- `clustering_visual.py`
    - Visualize clustering and community structure on the world map.

- `attack.py`
    - Run classical attack/robustness experiments (random removal, targeted by degree/betweenness) and aggregate LCC / robustness statistics.

## Phase 2 — GNN experiments (`GNN/` folder)

The `GNN/` folder contains code that trains a Graph Neural Network to predict node betweenness and uses the model for attack strategies (static and adaptive). These scripts are more experimental and rely on PyTorch / PyTorch-Geometric.

Key files:

- `GNN/generate_training_data.py`
    - Build training graphs, compute ground-truth betweenness labels, and extract node features used to train the GNN.

- `GNN/train_gnn.py`
    - Train the GraphSAGE model on the prepared datasets. The script saves model weights and the training feature normalization statistics.

- `GNN/attack_simulation_gnn.py`
    - Use the trained GNN to run attack simulations (static and adaptive policies). Produces LCC/robustness curves and optional visualizations.

- `GNN/perturbation_experiment.py`
    - Evaluate model robustness to graph perturbations (node/edge removal experiments).

- `GNN/visualize_attack.py`
    - Helper utilities to produce Plotly visualizations used by the GNN attack scripts.

- `GNN/plot_lcc_curves.py`
    - Aggregate and plot LCC curves from different attack strategies for comparison.

- `GNN/train_gnn.py`, `GNN/model/`
    - Training entrypoint and serialized model weights / saved normalization stats.

## Quick notes

- Phase 2 scripts assume you have PyTorch and PyTorch-Geometric installed. See `requirements.txt` for dependencies.
