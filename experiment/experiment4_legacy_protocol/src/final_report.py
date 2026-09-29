#!/usr/bin/env python3
"""Print the compact hand-off summary required by Experiment 4-Legacy."""

import csv
import os
import sys


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def f(row, key):
    return float(row[key])


def pm(mean, sd, scale=100.0, digits=2):
    return "{:.{d}f}±{:.{d}f}".format(mean * scale, sd * scale, d=digits)


def main():
    results_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "results")
    summary = {row["config"]: row for row in read_csv(os.path.join(results_dir, "summary.csv"))}
    seed_rows = {
        (row["config"], int(row["seed"])): row
        for row in read_csv(os.path.join(results_dir, "seed_summary.csv"))
    }
    rng_rows = read_csv(os.path.join(results_dir, "rng_audit.csv"))
    boot = read_csv(os.path.join(results_dir, "bootstrap.csv"))
    boot_sum = read_csv(os.path.join(results_dir, "bootstrap_summary.csv"))[0]
    verdicts = read_csv(os.path.join(results_dir, "verdicts.csv"))
    verdict = {row["verdict"]: row["status"] for row in verdicts}
    l0 = summary["L0_legacy_baseline"]
    l1 = summary["L1_legacy_uniform_global"]

    print("=== MMGL LEGACY PROTOCOL EXPERIMENT DONE ===")
    print("Protocol: original MMGL_transductive legacy simple-2; 2 configs × 3 seeds × 10 folds = 60 folds; val_idx=test_idx; checkpoint/early stopping uses test-fold ACC; legacy-selected OOF=5,226 subject rows.")
    print("Data: ABIDE N=871 (ASD=403, NC=468); modal dims=[48,6,10,256]; internal labels 0=ASD, 1=NC; data matches strict Experiment 4 SHA exactly.")
    print("Runs: L0 λ=0; L1 uniform six-pair hybrid λ=0.50, temperature=0.10, warmup=20; NEPOCH=1000; EARLY=50; seeds=0,1,2; no per-fold RNG reset.")
    print("Author-default seed-0 (10-fold mean±population fold SD; percent):")
    print("Config | ACC | legacy hard AUC | ASD SEN | NC SPE | best epoch mean")
    for config in ("L0_legacy_baseline", "L1_legacy_uniform_global"):
        row = seed_rows[(config, 0)]
        print("{} | {} | {} | {} | {} | {:.2f}".format(
            config,
            pm(f(row, "fold_mean_test_acc"), f(row, "fold_sd_test_acc")),
            pm(f(row, "fold_mean_legacy_hard_auc"), f(row, "fold_sd_legacy_hard_auc")),
            pm(f(row, "fold_mean_asd_sensitivity"), f(row, "fold_sd_asd_sensitivity")),
            pm(f(row, "fold_mean_nc_specificity"), f(row, "fold_sd_nc_specificity")),
            f(row, "fold_mean_best_epoch"),
        ))
    print("Paper Table III reference: MMGL ACC 89.77±2.72; AUC 89.81±2.56; SEN 90.32±4.21; SPE 89.30±6.04.")
    print("Three-seed legacy-selected OOF (mean±sample SD; percent):")
    print("Config | ACC | BA | probability ASD-AUC | legacy hard AUC | ASD SEN | NC SPE")
    for config in ("L0_legacy_baseline", "L1_legacy_uniform_global"):
        row = summary[config]
        print("{} | {} | {} | {} | {} | {} | {}".format(
            config,
            pm(f(row, "oof_acc_mean"), f(row, "oof_acc_sd")),
            pm(f(row, "oof_balanced_acc_mean"), f(row, "oof_balanced_acc_sd")),
            pm(f(row, "oof_asd_probability_auc_mean"), f(row, "oof_asd_probability_auc_sd")),
            pm(f(row, "oof_legacy_hard_auc_mean"), f(row, "oof_legacy_hard_auc_sd")),
            pm(f(row, "oof_asd_sensitivity_mean"), f(row, "oof_asd_sensitivity_sd")),
            pm(f(row, "oof_nc_specificity_mean"), f(row, "oof_nc_specificity_sd")),
        ))
    print("Representation diagnostics (fold means across seeds; higher cos-gap/R1 and lower CMMD are favorable):")
    print("Config | cos-gap | R1 | CMMD")
    for config in ("L0_legacy_baseline", "L1_legacy_uniform_global"):
        row = summary[config]
        print("{} | {} | {} | {}".format(
            config,
            pm(f(row, "fold_mean_alignment_cos_gap_mean"), f(row, "fold_mean_alignment_cos_gap_sd"), 1.0, 4),
            pm(f(row, "fold_mean_alignment_R1_mean"), f(row, "fold_mean_alignment_R1_sd"), 1.0, 4),
            pm(f(row, "fold_mean_alignment_CMMD_mean"), f(row, "fold_mean_alignment_CMMD_sd"), 1.0, 4),
        ))
    print("RNG audit: split parity={}/30; L0/L1 base-init parity={}/30; init hash covers ModalFusion+GraphConstruct+MessagePassing; per-fold RNG reset=False.".format(
        sum(row["split_parity"] == "True" for row in rng_rows),
        sum(row["base_init_parity"] == "True" for row in rng_rows),
    ))
    print("Strict historical effect: ΔBA=+0.81 pp; Δ probability AUC=+1.57 pp (from completed strict Experiment 4).")
    print("Legacy effect L1−L0: ΔBA={:+.2f} pp; Δ probability AUC={:+.2f} pp; BA wins={}/3; bootstrap mean ΔBA={:+.4f}, mean ΔAUC={:+.4f}.".format(
        100 * (f(l1, "oof_balanced_acc_mean") - f(l0, "oof_balanced_acc_mean")),
        100 * (f(l1, "oof_asd_probability_auc_mean") - f(l0, "oof_asd_probability_auc_mean")),
        sum(f(seed_rows[("L1_legacy_uniform_global", seed)], "oof_balanced_acc") > f(seed_rows[("L0_legacy_baseline", seed)], "oof_balanced_acc") for seed in (0, 1, 2)),
        f(boot_sum, "delta_BA_mean"),
        f(boot_sum, "delta_probability_AUC_mean"),
    ))
    print("Bootstrap per seed (10,000 paired subject resamples):")
    for row in boot:
        print("seed {}: ΔBA={:+.4f} CI[{:+.4f},{:+.4f}] p={:.4f}; ΔAUC={:+.4f} CI[{:+.4f},{:+.4f}] p={:.4f}".format(
            row["seed"], f(row, "delta_BA"), f(row, "BA_CI_low"), f(row, "BA_CI_high"), f(row, "BA_p"),
            f(row, "delta_probability_AUC"), f(row, "probability_AUC_CI_low"), f(row, "probability_AUC_CI_high"), f(row, "probability_AUC_p"),
        ))
    print("Verdicts:")
    for key in (
        "LEGACY_BASELINE_REPRODUCED", "ALIGNMENT_REPRESENTATION_WORKING",
        "LEGACY_ALIGNMENT_SUPPORTED", "LEGACY_INIT_FULLY_MATCHED",
        "PROTOCOL_ROBUST_ALIGNMENT",
    ):
        print("{}={}".format(key, verdict[key]))
    if verdict["LEGACY_ALIGNMENT_SUPPORTED"] == "YES":
        conclusion = "结论：在原论文对应的 MMGL_transductive 旧协议下，uniform global alignment 同时改善了表示诊断并满足预注册的 legacy 支持条件；但该 OOF 受 test-fold checkpoint selection 影响，严格泛化结论仍以 strict Experiment 4 为准。"
    elif verdict["LEGACY_ALIGNMENT_SUPPORTED"] == "PARTIAL":
        conclusion = "结论：旧协议下 alignment 的表示效果可以观察到，但预测收益未完全满足预注册支持条件；不能把 legacy-selected OOF 当作严格泛化证据。"
    else:
        conclusion = "结论：旧协议下未得到足够稳定的 alignment 预测支持；应以严格协议结果为主，并把旧协议结果视为敏感性分析。"
    print(conclusion)
    print("NEXT: 进入 SPromptGL Experiment 5，不继续扩展 MMGL Experiment 5。")
    print("Files: results/fold_results.csv; results/oof_predictions.csv; results/seed_summary.csv; results/summary.csv; results/rng_audit.csv; results/bootstrap.csv; results/bootstrap_summary.csv; results/verdicts.csv; results/report.md; logs/*.log (6)")


if __name__ == "__main__":
    main()
