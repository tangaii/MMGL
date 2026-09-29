#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/venvs/111/bin/python}"
NEPOCH="${NEPOCH:-1000}"
EARLY="${EARLY:-50}"
RAW_DIR="$ROOT_DIR/results/raw"
LOG_DIR="$ROOT_DIR/logs"

mkdir -p "$RAW_DIR" "$LOG_DIR"
declare -a PIDS=()
declare -a LABELS=()
JOB_INDEX=0

launch_job() {
    local gpu="$1" config="$2" seed="$3" align_mode="$4" align_lambda="$5"
    local stem="${config}_seed${seed}"
    CUDA_VISIBLE_DEVICES="$gpu" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" \
    "$PYTHON_BIN" "$ROOT_DIR/src/main.py" \
        --datadir "$ROOT_DIR/data/" \
        --datname ABIDE \
        --nepoch "$NEPOCH" \
        --early "$EARLY" \
        --lr 0.0038 \
        --reg 0.11 \
        --dropout 0.35 \
        --nlayer 1 \
        --n_hidden 18 \
        --n_head 2 \
        --nmodal 4 \
        --th 0.9 \
        --GC_mode weighted-cosine \
        --MP_mode GCN \
        --MF_mode ' ' \
        --alpha 0.5 \
        --theta_smooth 1 \
        --theta_degree 0.5 \
        --theta_sparsity 0 \
        --nclass 2 \
        --mode simple-2 \
        --config "$config" \
        --align_mode "$align_mode" \
        --align_lambda "$align_lambda" \
        --align_temperature 0.10 \
        --align_warmup 20 \
        --seed "$seed" \
        --fold_results "$RAW_DIR/${stem}.folds.csv" \
        --oof_results "$RAW_DIR/${stem}.oof.csv" \
        > "$LOG_DIR/${stem}.log" 2>&1 &
    PIDS+=("$!")
    LABELS+=("$stem")
}

reap_one() {
    local pid="${PIDS[0]}" label="${LABELS[0]}"
    if ! wait "$pid"; then
        echo "job failed: $label" >&2
        exit 1
    fi
    PIDS=("${PIDS[@]:1}")
    LABELS=("${LABELS[@]:1}")
}

for seed in 0 1 2; do
    while ((${#PIDS[@]} >= 4)); do
        reap_one
    done
    gpu=$((JOB_INDEX % 2))
    launch_job "$gpu" L0_legacy_baseline "$seed" none 0.0
    JOB_INDEX=$((JOB_INDEX + 1))

    while ((${#PIDS[@]} >= 4)); do
        reap_one
    done
    gpu=$((JOB_INDEX % 2))
    launch_job "$gpu" L1_legacy_uniform_global "$seed" hybrid 0.50
    JOB_INDEX=$((JOB_INDEX + 1))
done

while ((${#PIDS[@]} > 0)); do
    reap_one
done

"$PYTHON_BIN" "$ROOT_DIR/src/aggregate.py" \
    --raw-dir "$RAW_DIR" \
    --results-dir "$ROOT_DIR/results"

# Raw per-job files are intermediate artifacts; the six logs and aggregate
# tables remain as the reproducibility record.
find "$RAW_DIR" -type f -delete
find "$RAW_DIR" -type d -empty -delete

"$PYTHON_BIN" "$ROOT_DIR/src/final_report.py" "$ROOT_DIR/results"
