#!/usr/bin/env bash
# Grid the microbench profiler over {model sizes} x {kernel/algorithm variants},
# all on configs/1b_baseline.yml. Model sizes are passed as CLI overrides
# (model_dim/n_layer/n_head), which take precedence over the config -- so no
# per-size config files are needed. Each run writes its active_step_metrics.json
# into results/<size>/<variant>/.
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

# Model sizes (ordered). Each entry: "name|<override flags>".
# FILL IN real model_dim/n_layer/n_head values.
SIZES=(
    "1b|--model_dim 1536 --n_layer 24 --n_head 16"
    "3b|--model_dim 2304 --n_layer 32 --n_head 18"
    "4b|--model_dim 2560 --n_layer 36 --n_head 32"
    "7b|--model_dim 4096 --n_layer 32 --n_head 32 --device_batch_size 4 --batch_size 4"
    "14b|--model_dim 5120 --n_layer 40 --n_head 40 --device_batch_size 4 --batch_size 4"
)

# Variants (ordered). Each entry: "name|<flags>" (empty flags allowed).
VARIANTS=(
    # "1-plain|--no_triton"
    # "2-triton|"
    "3-baseline|--use_gns_package --no_triton"
    "4-cutlass|--use_gns_package"
    "5-gns|--use_gns_package --no_triton --use_gns_alg"
    "6-gns-cutlass|--use_gns_package --use_gns_alg"
)

run() {
    # run <out_dir> [extra flags...]
    local out="$1"; shift
    mkdir -p "$out"
    torchrun --standalone --nproc_per_node=1 \
        profile_training_step.py --config "$CONFIG" --profile_out "$out" \
        --no_profile --profile_wait 0 --profile_warmup 50 --profile_active 25 \
        "$@"
}

# Optional first argument: run only the named size (e.g. `./run_variants.sh 7b`).
# With no argument, run all sizes.
if [[ $# -ge 1 ]]; then
    want="$1"
    selected=()
    for s in "${SIZES[@]}"; do
        [[ "${s%%|*}" == "$want" ]] && selected+=("$s")
    done
    if [[ ${#selected[@]} -eq 0 ]]; then
        echo "Unknown size '$want'. Valid sizes: ${SIZES[*]%%|*}" >&2
        exit 1
    fi
    SIZES=("${selected[@]}")
fi

for s in "${SIZES[@]}"; do
    sname="${s%%|*}"; sflags="${s#*|}"
    for v in "${VARIANTS[@]}"; do
        vname="${v%%|*}"; vflags="${v#*|}"
        echo "=== ${sname} / ${vname}  (${sflags} ${vflags}) ==="
        # sflags/vflags are intentionally word-split into separate CLI args.
        # shellcheck disable=SC2086
        run "results/${sname}/${vname}" $sflags $vflags
    done
done

echo "Done. Metrics in results/<size>/<variant>/active_step_metrics.json"
