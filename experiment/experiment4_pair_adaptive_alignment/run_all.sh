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
    local gpu="$1" config="$2" seed="$3" strategy="$4" mode="$5" lambda_value="$6"
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
    local pid="${PIDS[0]}" label="${LABELS[0]}"
    if ! wait "$pid"; then
        echo "job failed: $label" >&2
        exit 1
    fi
    PIDS=("${PIDS[@]:1}")
    LABELS=("${LABELS[@]:1}")
}

for seed in 0 1 2; do
    for config in E0_baseline E1_uniform_global E2_diagnosis_adaptive E3_shuffled_adaptive E4_adaptive_top3 E5_balanced_sparse_control; do
        case "$config" in
            E0_baseline)
                strategy=none; mode=none; lambda_value=0.0 ;;
            E1_uniform_global)
                strategy=uniform; mode=hybrid; lambda_value=0.50 ;;
            E2_diagnosis_adaptive)
                strategy=adaptive; mode=hybrid; lambda_value=0.50 ;;
            E3_shuffled_adaptive)
                strategy=shuffled; mode=hybrid; lambda_value=0.50 ;;
            E4_adaptive_top3)
                strategy=top3; mode=hybrid; lambda_value=0.50 ;;
            E5_balanced_sparse_control)
                strategy=sparse; mode=hybrid; lambda_value=0.50 ;;
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

find "$RAW_DIR" -type f -delete
find "$RAW_DIR" -type d -empty -delete
"$PYTHON_BIN" "$ROOT_DIR/src/final_report.py" "$ROOT_DIR/results"
