#!/usr/bin/env bash
#SBATCH --job-name=microbench_variants
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=00:35:00
#SBATCH --account=torch_pr_508_tandon_advanced
## #SBATCH --exclusive
## #SBATCH --mem=0
## #SBATCH --partition=gpu
#
# Lightweight SLURM launcher: enter the Apptainer container (conda/torch) via
# ~/bin/singro and run the benchmark. Submit with:  sbatch submit_variants.slurm
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
exec "$HOME/bin/singro" bash "$SLURM_SUBMIT_DIR/run_variants.sh"
