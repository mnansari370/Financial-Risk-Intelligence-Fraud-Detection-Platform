#!/bin/bash
#SBATCH --job-name=build_graph
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/build_graph_%j.out
#SBATCH --error=slurm/logs/build_graph_%j.err

set -euo pipefail

PROJECT_DIR="/mnt/aiongpfs/users/nmo/financial-fraud-detection"
CONDA_ENV="fraud-detection"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$PROJECT_DIR"

echo "=== Build Graph Pipeline ==="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

python -m src.graph.build_graph \
    --paysim_path data/raw/paysim.csv \
    --output_dir data/processed \
    --snapshot_hours 24 \
    --stride_hours 12

echo "=== Graph build complete: $(date) ==="