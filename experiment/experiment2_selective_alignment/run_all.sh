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
    local gpu="$1"
    local config="$2"
    local seed="$3"
    local strategy="$4"
    local mode="$5"
    local lambda_value="$6"
    local stem="${config}_seed${seed}"

    CUDA_VISIBLE_DEVICES="$gpu" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" \
    "$PYTHON_BIN" "$ROOT_DIR/src/main.py" \
        --datadir "$ROOT_DIR/data/" \
        --config "$config" \
        --align_strategy "$strategy" \
        --align_mode "$mode" \
        --align_lambda "$lambda_value" \
        --align_temperature 0.10 \
        --align_warmup 20 \
        --lambda_orth 0.02 \
        --lambda_balance 0.05 \
        --target_shared_ratio 0.50 \
        --seed "$seed" \
        --nepoch "$NEPOCH" \
        --early "$EARLY" \
        --fold_results "$RAW_DIR/${stem}.folds.csv" \
        --oof_results "$RAW_DIR/${stem}.oof.csv" \
        > "$LOG_DIR/${stem}.log" 2>&1 &

    PIDS+=("$!")
    LABELS+=("$stem")
}

reap_one() {
    local pid="${PIDS[0]}"
    local label="${LABELS[0]}"
    if ! wait "$pid"; then
        echo "job failed: $label" >&2
        exit 1
    fi
    PIDS=("${PIDS[@]:1}")
    LABELS=("${LABELS[@]:1}")
}

for seed in 0 1 2; do
    for config in C0_baseline C1_global_pair C2_global_hybrid_medium C3_global_hybrid_strong C4_selective_hybrid_medium C5_selective_hybrid_strong; do
        case "$config" in
            C0_baseline)
                strategy=none; mode=none; lambda_value=0.0
                ;;
            C1_global_pair)
                strategy=global; mode=pair_nce; lambda_value=0.20
                ;;
            C2_global_hybrid_medium)
                strategy=global; mode=hybrid; lambda_value=0.20
                ;;
            C3_global_hybrid_strong)
                strategy=global; mode=hybrid; lambda_value=0.50
                ;;
            C4_selective_hybrid_medium)
                strategy=selective; mode=hybrid; lambda_value=0.20
                ;;
            C5_selective_hybrid_strong)
                strategy=selective; mode=hybrid; lambda_value=0.50
                ;;
        esac

        while ((${#PIDS[@]} >= 4)); do
            reap_one
        done
        gpu=$((JOB_INDEX % 2))
        launch_job "$gpu" "$config" "$seed" "$strategy" "$mode" "$lambda_value"
        JOB_INDEX=$((JOB_INDEX + 1))
    done
done

while ((${#PIDS[@]} > 0)); do
    reap_one
done

"$PYTHON_BIN" "$ROOT_DIR/src/aggregate.py" \
    --raw-dir "$RAW_DIR" \
    --results-dir "$ROOT_DIR/results"

# Raw per-job inputs are temporary; final artifacts are under results/.
find "$RAW_DIR" -type f -delete
find "$RAW_DIR" -type d -empty -delete

echo "=== MMGL EXPERIMENT 2 AGGREGATION DONE ==="
