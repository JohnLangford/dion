#!/usr/bin/env bash
# Run the microbench profiler across the flag variants, each writing its
# active_step_metrics.json into results/<variant>/. All variants share
# configs/1b_baseline.yml; the differing flags are passed on the CLI.
#
# Run this from an interactive session INSIDE the container (where torch is
# available). For a batch job, use submit_variants.slurm, which enters the
# container and calls this script.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# Pin to a single GPU: honor CUDA_VISIBLE_DEVICES if set (e.g. by SLURM), taking
# the first if several are exposed; otherwise default to GPU 0 (override: GPU=3).
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
else
    export CUDA_VISIBLE_DEVICES="${GPU:-0}"
fi
echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

CONFIG="configs/1b_baseline.yml"

run() {
    # run <name> [extra flags...]
    local name="$1"; shift
    local out="results/${name}"
    mkdir -p "$out"
    echo "=== ${name} ($*) ==="
    torchrun --standalone --nproc_per_node=1 \
        profile_training_step.py --config "$CONFIG" --profile_out "$out" \
        --no_profile --profile_wait 0 --profile_warmup 50 --profile_active 25 \
        "$@"
}

run 1-plain          --no_triton
run 2-triton
run 3-baseline       --use_gns_package --no_triton
run 4-cutlass        --use_gns_package
run 5-gns            --use_gns_package --no_triton --use_gns_alg
run 6-gns-cutlass    --use_gns_package --use_gns_alg

echo "Done. Metrics in results/*/active_step_metrics.json"
