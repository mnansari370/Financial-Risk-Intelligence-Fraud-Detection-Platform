#!/bin/bash
#SBATCH --job-name=build_graph
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/build_graph_%j.out
#SBATCH --error=slurm/logs/build_graph_%j.err

set -euo pipefail

PROJECT_DIR="/mnt/aiongpfs/users/nmo/financial-fraud-detection"
CONDA_ENV="fraud-detection"

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$PROJECT_DIR"

echo "=== Build Graph Pipeline ==="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

python -m src.graph.build_graph \
    --paysim_path data/raw/paysim.csv \
    --ieee_path data/raw/ieee_cis_train.csv \
    --output_dir data/processed \
    --snapshot_hours 24 \
    --n_workers "$SLURM_CPUS_PER_TASK"

echo "=== Graph build complete: $(date) ==="
