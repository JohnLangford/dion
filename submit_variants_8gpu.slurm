#!/usr/bin/env bash
#SBATCH --job-name=microbench_variants_8gpu
#SBATCH --output=slurm_logs/%x-%j.out
# Whole node: all 8 H200s + all 128 cores locks the node without --exclusive /
# --mem=0 (both banned) -- any other job needs >=1 CPU, and there are none left.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=2000G
#SBATCH --time=01:58:00
#SBATCH --account=torch_pr_508_tandon_advanced
#
# 8-GPU (FSDP) launcher: enter the Apptainer container (conda/torch) via
# ~/bin/singro and run the --fs_size sizes across the whole node. Args forward to
# run_variants.sh (defaults to all-8gpu).
#   sbatch submit_variants_8gpu.slurm                     # all-8gpu, muon-family
#   sbatch submit_variants_8gpu.slurm 7b-8gpu             # only 7b-8gpu
#   sbatch submit_variants_8gpu.slurm all-8gpu normuon-family
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
exec "$HOME/bin/singro" bash "$SLURM_SUBMIT_DIR/run_variants.sh" "${@:-all-8gpu}"
