#!/usr/bin/env bash
#SBATCH --job-name=microbench_variants
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=01:15:00
#SBATCH --account=torch_pr_508_tandon_advanced
## #SBATCH --exclusive
## #SBATCH --mem=0
## #SBATCH --partition=gpu
#
# Lightweight SLURM launcher: enter the Apptainer container (conda/torch) via
# ~/bin/singro and run the benchmark. Any args are forwarded to run_variants.sh:
#   $1 = size ("all"/omitted for every size), $2 = optimizer family.
#   sbatch submit_variants.slurm                     # all sizes, muon-family
#   sbatch submit_variants.slurm 7b                  # only 7b, muon-family
#   sbatch submit_variants.slurm 7b normuon-family   # only 7b, normuon-family
#   sbatch submit_variants.slurm all normuon-family  # all sizes, normuon-family
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
exec "$HOME/bin/singro" bash "$SLURM_SUBMIT_DIR/run_variants.sh" "$@"
