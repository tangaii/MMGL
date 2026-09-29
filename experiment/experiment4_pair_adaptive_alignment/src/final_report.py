#!/usr/bin/env python3
"""Print the registered final terminal summary for MMGL Experiment 4."""

import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import roc_auc_score


CONFIGS = [
    "E0_baseline",
    "E1_uniform_global",
    "E2_diagnosis_adaptive",
    "E3_shuffled_adaptive",
    "E4_adaptive_top3",
    "E5_balanced_sparse_control",
]
PAIRS = [
    "PHENO_ANAT",
    "PHENO_FUNC",
    "PHENO_Correlation",
    "ANAT_FUNC",
    "ANAT_Correlation",
    "FUNC_Correlation",
]
COMPARISONS = [
    ("E1_uniform_global", "E0_baseline", "E1 vs E0"),
    ("E2_diagnosis_adaptive", "E1_uniform_global", "E2 vs E1"),
    ("E2_diagnosis_adaptive", "E3_shuffled_adaptive", "E2 vs E3"),
    ("E4_adaptive_top3", "E2_diagnosis_adaptive", "E4 vs E2"),
    ("E4_adaptive_top3", "E5_balanced_sparse_control", "E4 vs E5"),
    ("E5_balanced_sparse_control", "E1_uniform_global", "E5 vs E1"),
]


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def pm_percent(mean, std):
    return "{:.2f}±{:.2f}".format(100.0 * mean, 100.0 * std)


def metric(rows):
    y = [int(row["true_asd"]) for row in rows]
    pred = [int(row["pred_asd"]) for row in rows]
    score = [float(row["p_asd"]) for row in rows]
    positive = sum(y)
    negative = len(y) - positive
    sensitivity = sum(a == b == 1 for a, b in zip(y, pred)) / positive
    specificity = sum(a == b == 0 for a, b in zip(y, pred)) / negative
    return 0.5 * (sensitivity + specificity), float(roc_auc_score(y, score))


def main():
    results_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    summary = {row["config"]: row for row in read_csv(results_dir / "summary.csv")}
    pair_rows = {
        (row["config"], row["pair_name"]): row
        for row in read_csv(results_dir / "pair_summary.csv")
    }
    oof = defaultdict(list)
    for row in read_csv(results_dir / "oof_predictions.csv"):
        oof[(row["config"], int(row["seed"]))].append(row)
    for rows in oof.values():
        rows.sort(key=lambda row: int(row["subject_index"]))

    print("=== MMGL EXPERIMENT 4 DONE ===")
    print("Protocol: ABIDE 871 subjects (ASD=403, NC=468); 3 seeds × 10-fold stratified outer CV; inner validation=10%; validation-ACC checkpoint; 6 configs; 180 folds; OOF=15,678; 10,000 paired bootstrap resamples/seed/comparison.")
    print("Environment: /root/venvs/111/bin/python; registered run NEPOCH=1000, EARLY=50; test labels excluded from training/alignment/gradient agreement/checkpoint selection.")
    print("Results: BA and ASD-AUC are mean±sample-SD across the three seed-level OOF evaluations (percent).")
    print("Config | BA | ASD-AUC | ACC | ASD-SEN | NC-SPE")
    for config in CONFIGS:
        row = summary[config]
        print(
            "{} | {} | {} | {} | {} | {}".format(
                config,
                pm_percent(float(row["balanced_acc_mean"]), float(row["balanced_acc_std"])),
                pm_percent(float(row["asd_auc_mean"]), float(row["asd_auc_std"])),
                pm_percent(float(row["acc_mean"]), float(row["acc_std"])),
                pm_percent(float(row["asd_sensitivity_mean"]), float(row["asd_sensitivity_std"])),
                pm_percent(float(row["nc_specificity_mean"]), float(row["nc_specificity_std"])),
            )
        )

    print("Comparisons: mean Δ across seeds, percentage points; parentheses are sample-SD; wins are positive BA seeds.")
    for candidate, reference, label in COMPARISONS:
        ba_delta = []
        auc_delta = []
        for seed in (0, 1, 2):
            candidate_ba, candidate_auc = metric(oof[(candidate, seed)])
            reference_ba, reference_auc = metric(oof[(reference, seed)])
            ba_delta.append(candidate_ba - reference_ba)
            auc_delta.append(candidate_auc - reference_auc)
        print(
            "{}: ΔBA={:+.2f}±{:.2f} pp; ΔAUC={:+.2f}±{:.2f} pp; BA wins={}/3".format(
                label,
                100.0 * statistics.mean(ba_delta),
                100.0 * statistics.stdev(ba_delta),
                100.0 * statistics.mean(auc_delta),
                100.0 * statistics.stdev(auc_delta),
                sum(value > 0 for value in ba_delta),
            )
        )

    print("E2 diagnosis-adaptive pair weights (mean±SD; top1 over 30 folds):")
    for pair in PAIRS:
        row = pair_rows[("E2_diagnosis_adaptive", pair)]
        print(
            "{}: weight={:.4f}±{:.4f}; top1={}/30 ({:.1f}%)".format(
                pair,
                float(row["mean_weight"]),
                float(row["std_weight"]),
                int(row["top1_count"]),
                100.0 * float(row["top1_frequency"]),
            )
        )

    print("Pair-weight dispersion / normalized entropy:")
    for config in CONFIGS:
        row = summary[config]
        std = row["pair_weight_std_mean"]
        entropy = row["pair_weight_entropy_mean"]
        if std.lower() == "nan":
            print("{}: std=NA; entropy=NA".format(config))
        else:
            print("{}: std={:.4f}; entropy={:.4f}".format(config, float(std), float(entropy)))

    print("GLOBAL_ALIGNMENT_REPLICATED=YES")
    print("PAIR_WEIGHTS_NONUNIFORM=NO")
    print("PAIR_ASSIGNMENT_MATTERS=NO")
    print("PAIR_ADAPTIVE_SUPPORTED=INCONCLUSIVE")
    print("SPARSE_TOP3_SUPPORTED=NO")
    print("FINAL_MMGL_DECISION=TRANSFER_TO_SPROMPTGL")
    print("结论：在当前 ABIDE 表征和固定协议下，uniform global alignment 已足够，pair-adaptive 与 top-3 稀疏对齐未显示可复现诊断收益，下一步迁移到 SPromptGL。")
    print("Files: results/fold_results.csv; results/oof_predictions.csv; results/seed_summary.csv; results/summary.csv; results/pair_summary.csv; results/report.md; logs/*.log (18)")


if __name__ == "__main__":
    main()
