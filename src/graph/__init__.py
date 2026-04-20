"""
Graph construction for the PaySim transaction dataset.

Builds rolling 24-hour account-to-account transaction snapshots with
Z-score normalisation fitted on the training split only.

Run via:  python -m src.graph.build_graph --config configs/gat_config.yaml
Or SLURM: sbatch slurm/build_graph.sh
"""

from src.graph.build_graph import EDGE_FEATURE_COLS, NODE_FEATURE_COLS, CONT_EDGE_COUNT

__all__ = ["EDGE_FEATURE_COLS", "NODE_FEATURE_COLS", "CONT_EDGE_COUNT"]
