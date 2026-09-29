import argparse
import csv
import glob
import math
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from alignment import PAIR_NAMES


CONFIGS = [
    "E0_baseline",
    "E1_uniform_global",
    "E2_diagnosis_adaptive",
    "E3_shuffled_adaptive",
    "E4_adaptive_top3",
    "E5_balanced_sparse_control",
]
SEEDS = [0, 1, 2]
PRIMARY = ["acc", "balanced_acc", "asd_auc", "asd_f1", "asd_sensitivity", "nc_specificity"]
FOLD_DIAGNOSTICS = ["raw_global_cos_gap", "raw_global_R1", "raw_global_CMMD", "pair_weight_std", "pair_weight_entropy"]
COMPARISONS = [
    ("E1_uniform_global", "E0_baseline"),
    ("E2_diagnosis_adaptive", "E1_uniform_global"),
    ("E2_diagnosis_adaptive", "E3_shuffled_adaptive"),
    ("E4_adaptive_top3", "E2_diagnosis_adaptive"),
    ("E4_adaptive_top3", "E5_balanced_sparse_control"),
    ("E5_balanced_sparse_control", "E1_uniform_global"),
]


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def num(row, key, default=np.nan):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields, seen = [], set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    return float(values.mean()), float(values.std(ddof=1) if len(values) > 1 else 0.0)


def metric_arrays(y, p, pred):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=float)
    pred = np.asarray(pred, dtype=np.int64)
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tp = int(np.sum((y == 1) & (pred == 1)))
    return {
        "acc": float(accuracy_score(y, pred)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "asd_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "asd_f1": float(f1_score(y, pred, zero_division=0)),
        "asd_sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "nc_specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
    }


def oof_metrics(rows):
    rows = sorted(rows, key=lambda row: int(row["subject_index"]))
    if [int(row["subject_index"]) for row in rows] != list(range(871)):
        raise ValueError("OOF subjects are missing, duplicated or out of order")
    y = np.asarray([int(row["true_asd"]) for row in rows], dtype=np.int64)
    p = np.asarray([float(row["p_asd"]) for row in rows])
    pred = np.asarray([int(row["pred_asd"]) for row in rows])
    if tuple(np.bincount(y, minlength=2).tolist()) != (468, 403):
        raise ValueError("OOF class counts are inconsistent")
    return metric_arrays(y, p, pred)


def validate_raw(raw_dir):
    fold_paths = sorted(glob.glob(os.path.join(raw_dir, "*.folds.csv")))
    oof_paths = sorted(glob.glob(os.path.join(raw_dir, "*.oof.csv")))
    expected = {"{}_seed{}".format(config, seed) for config in CONFIGS for seed in SEEDS}
    fold_stems = {os.path.basename(path)[:-len(".folds.csv")] for path in fold_paths}
    oof_stems = {os.path.basename(path)[:-len(".oof.csv")] for path in oof_paths}
    if fold_stems != expected or oof_stems != expected:
        raise ValueError("Expected exactly 18 fold and OOF files")
    folds, oofs = {}, {}
    for stem in sorted(expected):
        fold_rows = read_csv(os.path.join(raw_dir, stem + ".folds.csv"))
        oof_rows = read_csv(os.path.join(raw_dir, stem + ".oof.csv"))
        if len(fold_rows) != 10 or sorted(int(row["fold"]) for row in fold_rows) != list(range(1, 11)):
            raise ValueError("{} does not contain exactly ten folds".format(stem))
        if len(oof_rows) != 871 or len({int(row["subject_index"]) for row in oof_rows}) != 871:
            raise ValueError("{} does not contain exactly 871 unique OOF subjects".format(stem))
        folds[stem], oofs[stem] = fold_rows, oof_rows
    return folds, oofs


def seed_summary_rows(folds, oofs):
    rows = []
    for config in CONFIGS:
        for seed in SEEDS:
            stem = "{}_seed{}".format(config, seed)
            fold_rows = folds[stem]
            row = {"config": config, "seed": seed, "n_oof": 871}
            row.update(oof_metrics(oofs[stem]))
            for metric in FOLD_DIAGNOSTICS:
                row["fold_mean_{}".format(metric)], row["fold_std_{}".format(metric)] = mean_std(
                    [num(item, metric) for item in fold_rows]
                )
            for pair_id in range(6):
                for metric in ("score", "weight", "cos_gap", "R1", "CMMD"):
                    key = "score_P{}".format(pair_id) if metric == "score" else (
                        "weight_P{}".format(pair_id) if metric == "weight" else "P{}_{}".format(pair_id, metric)
                    )
                    row["fold_mean_{}".format(key)], row["fold_std_{}".format(key)] = mean_std(
                        [num(item, key) for item in fold_rows]
                    )
            row["runtime_sec_mean"], row["runtime_sec_std"] = mean_std(
                [num(item, "runtime_sec") for item in fold_rows]
            )
            row["base_init_sha256"] = fold_rows[0].get("base_init_sha256", "")
            rows.append(row)
    return rows


def config_summary_rows(seed_rows):
    rows = []
    for config in CONFIGS:
        selected = [row for row in seed_rows if row["config"] == config]
        row = {"config": config, "n_seeds": len(selected)}
        for metric in PRIMARY:
            row[metric + "_mean"], row[metric + "_std"] = mean_std([num(item, metric) for item in selected])
        for metric in FOLD_DIAGNOSTICS:
            row[metric + "_mean"], row[metric + "_std"] = mean_std(
                [num(item, "fold_mean_{}".format(metric)) for item in selected]
            )
        for pair_id in range(6):
            for metric in ("score", "weight", "cos_gap", "R1", "CMMD"):
                key = "score_P{}".format(pair_id) if metric == "score" else (
                    "weight_P{}".format(pair_id) if metric == "weight" else "P{}_{}".format(pair_id, metric)
                )
                row["fold_mean_{}".format(key)], row["fold_std_{}".format(key)] = mean_std(
                    [num(item, "fold_mean_{}".format(key)) for item in selected]
                )
        row["runtime_sec_mean"], row["runtime_sec_std"] = mean_std(
            [num(item, "runtime_sec_mean") for item in selected]
        )
        rows.append(row)
    e1 = next(item for item in rows if item["config"] == "E1_uniform_global")
    for row in rows:
        row["delta_ba_vs_E1"] = row["balanced_acc_mean"] - e1["balanced_acc_mean"]
        row["delta_auc_vs_E1"] = row["asd_auc_mean"] - e1["asd_auc_mean"]
        seed_values = {item["seed"]: num(item, "balanced_acc") for item in seed_rows if item["config"] == row["config"]}
        e1_values = {item["seed"]: num(item, "balanced_acc") for item in seed_rows if item["config"] == "E1_uniform_global"}
        row["ba_seed_wins_vs_E1"] = sum(seed_values[s] > e1_values[s] for s in SEEDS) if row["config"] != "E1_uniform_global" else 0
    return rows


def bootstrap_metric_delta(candidate_rows, reference_rows, random_seed, n_bootstrap=10000):
    candidate = {int(row["subject_index"]): row for row in candidate_rows}
    reference = {int(row["subject_index"]): row for row in reference_rows}
    y = np.asarray([int(candidate[i]["true_asd"]) for i in range(871)], dtype=np.int64)
    cand_p = np.asarray([float(candidate[i]["p_asd"]) for i in range(871)])
    ref_p = np.asarray([float(reference[i]["p_asd"]) for i in range(871)])
    cand_pred = np.asarray([int(candidate[i]["pred_asd"]) for i in range(871)])
    ref_pred = np.asarray([int(reference[i]["pred_asd"]) for i in range(871)])
    full_candidate = metric_arrays(y, cand_p, cand_pred)
    full_reference = metric_arrays(y, ref_p, ref_pred)
    rng = np.random.default_rng(random_seed)
    ba_deltas = np.empty(n_bootstrap)
    auc_deltas = np.empty(n_bootstrap)
    cand_order = np.argsort(cand_p, kind="mergesort")
    ref_order = np.argsort(ref_p, kind="mergesort")

    def auc_from_counts(counts, order):
        ordered_counts = counts[order].astype(np.float64)
        ordered_y = y[order]
        pos_counts = ordered_counts * (ordered_y == 1)
        neg_counts = ordered_counts * (ordered_y == 0)
        neg_before = np.cumsum(neg_counts) - neg_counts
        numerator = np.sum(pos_counts * neg_before) + 0.5 * np.sum(pos_counts * neg_counts)
        positives, negatives = pos_counts.sum(), neg_counts.sum()
        return numerator / (positives * negatives) if positives and negatives else 0.5

    cursor = 0
    while cursor < n_bootstrap:
        size = min(250, n_bootstrap - cursor)
        sample = rng.integers(0, 871, size=(size, 871))
        ys = y[sample]
        cand_pred_s, ref_pred_s = cand_pred[sample], ref_pred[sample]
        pos_count = np.maximum(ys.sum(axis=1), 1)
        neg_count = np.maximum(ys.shape[1] - ys.sum(axis=1), 1)
        cand_tp = ((ys == 1) & (cand_pred_s == 1)).sum(axis=1)
        cand_tn = ((ys == 0) & (cand_pred_s == 0)).sum(axis=1)
        ref_tp = ((ys == 1) & (ref_pred_s == 1)).sum(axis=1)
        ref_tn = ((ys == 0) & (ref_pred_s == 0)).sum(axis=1)
        ba_deltas[cursor:cursor + size] = 0.5 * (
            cand_tp / pos_count + cand_tn / neg_count
            - ref_tp / pos_count - ref_tn / neg_count
        )
        for local in range(size):
            counts = np.bincount(sample[local], minlength=871)
            auc_deltas[cursor + local] = (
                auc_from_counts(counts, cand_order) - auc_from_counts(counts, ref_order)
            )
        cursor += size

    def summarize(values, full_delta):
        p_value = 2.0 * min(float(np.mean(values <= 0)), float(np.mean(values >= 0)))
        return {
            "mean_delta": float(full_delta),
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
            "bootstrap_p": float(min(1.0, p_value)),
        }
    return {
        "ba": summarize(ba_deltas, full_candidate["balanced_acc"] - full_reference["balanced_acc"]),
        "auc": summarize(auc_deltas, full_candidate["asd_auc"] - full_reference["asd_auc"]),
    }


def planned_comparisons(oofs):
    output = []
    for index, (candidate, reference) in enumerate(COMPARISONS):
        for seed in SEEDS:
            stats = bootstrap_metric_delta(
                oofs["{}_seed{}".format(candidate, seed)],
                oofs["{}_seed{}".format(reference, seed)],
                random_seed=9400 + index * 100 + seed,
            )
            output.append({"candidate": candidate, "reference": reference, "seed": seed, "ba": stats["ba"], "auc": stats["auc"]})
    return output


def comparison_stats(comparisons, candidate, reference):
    selected = [item for item in comparisons if item["candidate"] == candidate and item["reference"] == reference]
    ba_values = [item["ba"]["mean_delta"] for item in selected]
    auc_values = [item["auc"]["mean_delta"] for item in selected]
    ba_mean, ba_std = mean_std(ba_values)
    auc_mean, auc_std = mean_std(auc_values)
    return {
        "rows": selected, "ba_mean": ba_mean, "ba_std": ba_std,
        "auc_mean": auc_mean, "auc_std": auc_std,
        "ba_wins": sum(value > 0 for value in ba_values),
    }


def pair_summary_rows(all_fold_rows):
    rows = []
    for config in CONFIGS:
        selected = [row for row in all_fold_rows if row["config"] == config]
        for pair_id, pair_name in enumerate(PAIR_NAMES):
            weights = [num(row, "weight_P{}".format(pair_id)) for row in selected]
            scores = [num(row, "score_P{}".format(pair_id)) for row in selected]
            cos_gap = [num(row, "P{}_cos_gap".format(pair_id)) for row in selected]
            r1 = [num(row, "P{}_R1".format(pair_id)) for row in selected]
            cmmd = [num(row, "P{}_CMMD".format(pair_id)) for row in selected]
            top1 = sum(row.get("top1_pair") == pair_name for row in selected)
            top3 = sum(pair_name in row.get("top3_pairs", "").split(",") for row in selected)
            wm, ws = mean_std(weights)
            sm, ss = mean_std(scores)
            gm, _ = mean_std(cos_gap)
            rm, _ = mean_std(r1)
            cm, _ = mean_std(cmmd)
            rows.append({
                "config": config, "pair_name": pair_name,
                "mean_weight": wm, "std_weight": ws,
                "mean_score": sm, "std_score": ss,
                "top1_count": top1, "top1_frequency": top1 / 30.0,
                "top3_count": top3, "top3_frequency": top3 / 30.0,
                "test_cos_gap_mean": gm, "test_R1_mean": rm, "test_CMMD_mean": cm,
            })
    return rows


def heldout_spearman(all_fold_rows):
    rows = [row for row in all_fold_rows if row["config"] == "E2_diagnosis_adaptive"]
    output = []
    for pair_id, pair_name in enumerate(PAIR_NAMES):
        weight = np.asarray([num(row, "weight_P{}".format(pair_id)) for row in rows])
        results = {"weight_vs_cos_gap": np.nan, "weight_vs_R1": np.nan, "weight_vs_neg_CMMD": np.nan}
        for metric, values in (
            ("weight_vs_cos_gap", [num(row, "P{}_cos_gap".format(pair_id)) for row in rows]),
            ("weight_vs_R1", [num(row, "P{}_R1".format(pair_id)) for row in rows]),
            ("weight_vs_neg_CMMD", [-num(row, "P{}_CMMD".format(pair_id)) for row in rows]),
        ):
            values = np.asarray(values)
            mask = np.isfinite(weight) & np.isfinite(values)
            if mask.sum() >= 3:
                results[metric] = float(spearmanr(weight[mask], values[mask]).statistic)
        output.append({"pair_name": pair_name, **results})
    return output


def decisions(summary, comparisons):
    lookup = {row["config"]: row for row in summary}
    stats = {}
    for candidate, reference in COMPARISONS:
        stats[candidate + "_vs_" + reference] = comparison_stats(comparisons, candidate, reference)
    e1e0 = stats["E1_uniform_global_vs_E0_baseline"]
    e2e1 = stats["E2_diagnosis_adaptive_vs_E1_uniform_global"]
    e2e3 = stats["E2_diagnosis_adaptive_vs_E3_shuffled_adaptive"]
    e4e2 = stats["E4_adaptive_top3_vs_E2_diagnosis_adaptive"]
    e4e5 = stats["E4_adaptive_top3_vs_E5_balanced_sparse_control"]
    e5e1 = stats["E5_balanced_sparse_control_vs_E1_uniform_global"]

    global_status = "YES" if e1e0["ba_mean"] > 0 and e1e0["auc_mean"] > 0 else (
        "PARTIAL" if e1e0["ba_mean"] > 0 or e1e0["auc_mean"] > 0 else "NO"
    )
    nonuniform = np.isfinite(lookup["E2_diagnosis_adaptive"]["pair_weight_std_mean"]) and lookup["E2_diagnosis_adaptive"]["pair_weight_std_mean"] >= 0.08
    if not nonuniform:
        adaptive_status = "INCONCLUSIVE"
    elif (
        e2e1["ba_mean"] >= 0.005 and e2e1["ba_wins"] >= 2
        and e2e1["auc_mean"] >= -0.0025
        and e2e3["ba_mean"] > 0 and e2e3["ba_wins"] >= 2
    ):
        adaptive_status = "YES"
    elif e2e1["ba_mean"] > 0 and e2e3["ba_mean"] > 0:
        adaptive_status = "PARTIAL"
    else:
        adaptive_status = "NO"
    assignment_matters = (
        e2e3["ba_mean"] > 0 and e2e3["auc_mean"] > 0 and e2e3["ba_wins"] >= 2
    )
    sparse_a = e4e2["ba_mean"] >= -0.0025 and e4e2["auc_mean"] >= -0.0025
    sparse_b = e4e5["ba_mean"] > 0 and e4e5["ba_wins"] >= 2
    sparse_status = "YES" if sparse_a and sparse_b else ("PARTIAL" if sparse_a else "NO")
    if e4e2["ba_mean"] > e2e1["ba_mean"]:
        best_strategy = "adaptive-top3"
    elif e2e1["ba_mean"] > 0 and adaptive_status in ("YES", "PARTIAL"):
        best_strategy = "diagnosis-adaptive"
    else:
        best_strategy = "uniform global"
    return {
        "global_status": global_status,
        "pair_weights_nonuniform": "YES" if nonuniform else "NO",
        "pair_assignment_matters": "YES" if assignment_matters else "NO",
        "pair_adaptive_supported": adaptive_status,
        "sparse_top3_supported": sparse_status,
        "stats": stats,
        "best_strategy": best_strategy,
        "e5_vs_e1": e5e1,
    }


def fmt_pm(mean, std, scale=1.0, digits=2):
    return "NA" if not np.isfinite(mean) else "{:.{d}f}±{:.{d}f}".format(mean * scale, std * scale, d=digits)


def fmt(value, digits=4):
    return "NA" if not np.isfinite(value) else "{:.{}f}".format(value, digits)


def write_report(results_dir, summary, pair_rows, comparisons, decisions_out, spearman_rows):
    lookup = {row["config"]: row for row in summary}
    lines = [
        "# MMGL Experiment 4: Diagnosis-Aware Pair-Adaptive Alignment", "",
        "## 1. Executive Summary", "",
        "This final MMGL alignment mechanism study tests whether six heterogeneous modality pairs should receive diagnosis-aware alignment weights instead of one equal global weight.", "",
        "GLOBAL_ALIGNMENT_REPLICATED: **{}**".format(decisions_out["global_status"]), "",
        "PAIR_WEIGHTS_NONUNIFORM: **{}**".format(decisions_out["pair_weights_nonuniform"]), "",
        "PAIR_ASSIGNMENT_MATTERS: **{}**".format(decisions_out["pair_assignment_matters"]), "",
        "PAIR_ADAPTIVE_SUPPORTED: **{}**".format(decisions_out["pair_adaptive_supported"]), "",
        "SPARSE_TOP3_SUPPORTED: **{}**".format(decisions_out["sparse_top3_supported"]), "",
        "## 2. Previous Evidence", "",
        "Global alignment was replicated in Experiments 2 and 3. Experiment 3 formed an auditable shared/private decomposition, but it did not consistently exceed global alignment. Experiment 4 therefore tests pair-specific alignment while preserving the original MMGL prediction path.", "",
        "## 3. Dataset and Modalities", "",
        "ABIDE contains 871 subjects, including ASD=403 and NC=468. PHENO has 48 features, ANAT contains 6 anatomical QC features, FUNC contains 10 functional QC features, and Correlation contains 256 fMRI functional-connectivity-derived features. The MMGL target token shape is `[N,4,36]`.", "",
        "## 4. Six Modality Pairs", "",
        "| Pair | Modalities |", "|---|---|",
    ]
    for pair_id, name in enumerate(PAIR_NAMES):
        lines.append("| P{} | {} |".format(pair_id, name.replace("_", " ↔ ")))
    lines += [
        "", "## 5. Why Not Learn Free Pair Weights", "",
        "A free alpha parameter optimized together with alignment loss can favor pairs that are easy to align. That identifies alignment difficulty, not diagnostic utility. Experiment 4 uses the detached cosine agreement between classification and pair alignment gradients.", "",
        "## 6. Diagnosis Gradient Agreement", "",
        "For pair p, `s_p = cos(∇_H L_cls, ∇_H L_align,p)`. Positive scores indicate that reducing the pair loss is locally aligned with reducing classification loss; negative scores indicate local conflict. The gradients are computed on the full modal-token tensor and detached before the actual optimization backward pass.", "",
        "## 7. Constant Alignment Budget", "",
        "Every applied pair weight vector is normalized to mean alpha = 1, and `L_align = 1/6 Σ alpha_p L_p`. Uniform, adaptive, shuffled, top3 and balanced sparse controls therefore share the same total alignment budget. Top3 and sparse vectors have three nonzero entries whose sum is 6.", "",
        "## 8. Configurations", "",
        "| Config | Alignment strategy | λ |", "|---|---|---:|",
        "| E0_baseline | none | 0.00 |",
        "| E1_uniform_global | six equal pairs | 0.50 |",
        "| E2_diagnosis_adaptive | gradient agreement + EMA | 0.50 |",
        "| E3_shuffled_adaptive | adaptive weights with fixed derangement | 0.50 |",
        "| E4_adaptive_top3 | adaptive top 3 pairs | 0.50 |",
        "| E5_balanced_sparse_control | fixed alternating three pairs | 0.50 |",
        "", "## 9. OOF Diagnostic Results", "",
        "Three-seed subject-level OOF mean ± standard deviation. ASD is the positive class.", "",
        "| Config | ACC | BA | ASD-AUC | ASD-F1 | ASD SEN | NC SPE |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for config in CONFIGS:
        row = lookup[config]
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            config, fmt_pm(row["acc_mean"], row["acc_std"], 100),
            fmt_pm(row["balanced_acc_mean"], row["balanced_acc_std"], 100),
            fmt_pm(row["asd_auc_mean"], row["asd_auc_std"], 100),
            fmt_pm(row["asd_f1_mean"], row["asd_f1_std"], 100),
            fmt_pm(row["asd_sensitivity_mean"], row["asd_sensitivity_std"], 100),
            fmt_pm(row["nc_specificity_mean"], row["nc_specificity_std"], 100),
        ))
    lines += [
        "", "## 10. Global Alignment Replication", "",
        "E0 and E1 are the integrity controls for Experiment 3 D0 and D2. The reference values are E0 BA 83.79±1.01% / ASD-AUC 87.82±1.27% and E1 BA 84.60±1.14% / ASD-AUC 89.38±0.37%. Differences beyond 0.10 percentage point require protocol inspection before interpretation.", "",
        "## 11. Did Pair Weights Actually Differentiate?", "",
        "E2 mean pair-weight standard deviation is {} and normalized entropy is {}. Nonuniformity is pre-registered at pair-weight standard deviation ≥0.08.".format(
            fmt(lookup["E2_diagnosis_adaptive"]["pair_weight_std_mean"]),
            fmt(lookup["E2_diagnosis_adaptive"]["pair_weight_entropy_mean"]),
        ), "",
        "PAIR_WEIGHTS_NONUNIFORM: **{}**".format(decisions_out["pair_weights_nonuniform"]), "",
        "## 12. Learned Pair Weight Structure", "",
        "The frequencies describe how often a pair receives the largest or top-three effective weight under the fixed criterion. They are properties of this representation and training objective, not ASD biological biomarkers.", "",
        "| Config | Pair | Weight mean±std | Score mean±std | Top1 | Top3 | Held-out cos-gap | Held-out R@1 | Held-out CMMD |", "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config in ("E2_diagnosis_adaptive", "E3_shuffled_adaptive", "E4_adaptive_top3"):
        for item in pair_rows:
            if item["config"] != config:
                continue
            lines.append("| {} | {} | {} | {} | {}/30 | {}/30 | {} | {} | {} |".format(
                config, item["pair_name"], fmt_pm(item["mean_weight"], item["std_weight"]),
                fmt_pm(item["mean_score"], item["std_score"]), item["top1_count"], item["top3_count"],
                fmt(item["test_cos_gap_mean"]), fmt(item["test_R1_mean"]), fmt(item["test_CMMD_mean"]),
            ))
    lines += [
        "", "## 13. Adaptive vs Uniform", "",
        "E2 vs E1 is the primary pair-adaptive comparison.", "",
        "## 14. Correct Assignment Control", "",
        "E3 uses the same diagnosis-gradient scores and weight distribution as E2, then applies a fixed derangement. A positive E2 vs E3 result supports the value of assigning weights to the pair that produced the score.", "",
        "## 15. Sparse Alignment", "",
        "E4 vs E2 tests adaptive top3 against all-pair adaptive weighting. E4 vs E5 compares adaptive selection with a fixed balanced three-pair schedule. E5 vs E1 tests whether using fewer pairs alone changes performance.", "",
        "## 16. Pair-specific Held-out Alignment", "",
        "Held-out pair metrics are reported descriptively for the best checkpoint. E2 weight versus held-out cos-gap, R@1 and negative CMMD Spearman correlations are exploratory.", "",
        "| Pair | weight vs cos-gap | weight vs R@1 | weight vs -CMMD |", "|---|---:|---:|---:|",
    ]
    for item in spearman_rows:
        lines.append("| {} | {} | {} | {} |".format(item["pair_name"], fmt(item["weight_vs_cos_gap"]), fmt(item["weight_vs_R1"]), fmt(item["weight_vs_neg_CMMD"])))
    lines += ["", "## 17. OOF Paired Bootstrap", "", "Each planned comparison uses 10,000 paired subject bootstrap samples per seed. Repeated-CV training sets overlap, so bootstrap p values are exploratory evidence rather than independent clinical validation.", "", "| Comparison | Seed | Δ BA | BA 95% CI | BA p | Δ AUC | AUC 95% CI | AUC p |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for item in comparisons:
        lines.append("| {} vs {} | {} | {:.4f} | [{:.4f}, {:.4f}] | {:.4g} | {:.4f} | [{:.4f}, {:.4f}] | {:.4g} |".format(
            item["candidate"], item["reference"], item["seed"], item["ba"]["mean_delta"], item["ba"]["ci_low"], item["ba"]["ci_high"], item["ba"]["bootstrap_p"], item["auc"]["mean_delta"], item["auc"]["ci_low"], item["auc"]["ci_high"], item["auc"]["bootstrap_p"],
        ))
    lines += ["", "| Comparison | Mean Δ BA | Mean Δ AUC | BA seed wins |", "|---|---:|---:|---:|"]
    names = ["E1 vs E0", "E2 vs E1", "E2 vs E3", "E4 vs E2", "E4 vs E5", "E5 vs E1"]
    for name, (candidate, reference) in zip(names, COMPARISONS):
        item = decisions_out["stats"][candidate + "_vs_" + reference]
        lines.append("| {} | {} | {} | {}/3 |".format(name, fmt_pm(item["ba_mean"], item["ba_std"]), fmt_pm(item["auc_mean"], item["auc_std"]), item["ba_wins"]))
    lines += [
        "", "## 18. ASD Sensitivity", "",
        "ASD sensitivity and NC specificity are shown together with BA. This identifies whether a change reflects ASD detection, NC rejection, or both.", "",
        "## 19. Verdict", "",
        "GLOBAL_ALIGNMENT_REPLICATED: **{}**".format(decisions_out["global_status"]), "",
        "PAIR_WEIGHTS_NONUNIFORM: **{}**".format(decisions_out["pair_weights_nonuniform"]), "",
        "PAIR_ASSIGNMENT_MATTERS: **{}**".format(decisions_out["pair_assignment_matters"]), "",
        "PAIR_ADAPTIVE_SUPPORTED: **{}**".format(decisions_out["pair_adaptive_supported"]), "",
        "SPARSE_TOP3_SUPPORTED: **{}**".format(decisions_out["sparse_top3_supported"]), "",
        "## 20. Final MMGL Alignment Conclusion", "",
    ]
    if decisions_out["pair_adaptive_supported"] in ("YES", "PARTIAL") or decisions_out["sparse_top3_supported"] == "YES":
        conclusion = "在当前 ABIDE 表征和固定训练协议下，显式 alignment 有效；pair-adaptive 机制的证据为 {}，应将固定的最佳策略迁移到 SPromptGL 验证。".format(decisions_out["pair_adaptive_supported"])
    else:
        conclusion = "在当前 ABIDE 表征和固定训练协议下，uniform global alignment 已足够；额外的 pair-adaptive 复杂度没有提供可复现的诊断收益。"
    lines += [conclusion, "", "FINAL_MMGL_DECISION: **TRANSFER_TO_SPROMPTGL**", "", "Transfer strategy: **{}**".format(decisions_out["best_strategy"]), "", "## 21. Next Step", "", "将固定结论迁移到 SPromptGL，使用当前最优的 uniform global、diagnosis-adaptive 或 adaptive-top3 策略进行跨模型验证；Experiment 4 之后不再在 MMGL 上继续搜索 alignment 超参数。", "", "## Reproducibility", "", "The folder is self-contained and uses `/root/venvs/111/bin/python` with 32 CPU cores, 128 GB RAM and two approximately 96 GB GPUs. The original data and prior experiment folders are not modified."]
    with open(os.path.join(results_dir, "report.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate MMGL Experiment 4")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    folds, oofs = validate_raw(args.raw_dir)
    all_fold_rows = [row for config in CONFIGS for seed in SEEDS for row in folds["{}_seed{}".format(config, seed)]]
    all_oof_rows = [row for config in CONFIGS for seed in SEEDS for row in oofs["{}_seed{}".format(config, seed)]]
    seed_rows = seed_summary_rows(folds, oofs)
    summary = config_summary_rows(seed_rows)
    pair_rows = pair_summary_rows(all_fold_rows)
    comparisons = planned_comparisons(oofs)
    decisions_out = decisions(summary, comparisons)
    spearman_rows = heldout_spearman(all_fold_rows)
    write_csv(os.path.join(args.results_dir, "fold_results.csv"), all_fold_rows)
    write_csv(os.path.join(args.results_dir, "oof_predictions.csv"), all_oof_rows)
    write_csv(os.path.join(args.results_dir, "seed_summary.csv"), seed_rows)
    write_csv(os.path.join(args.results_dir, "summary.csv"), summary)
    write_csv(os.path.join(args.results_dir, "pair_summary.csv"), pair_rows)
    write_report(args.results_dir, summary, pair_rows, comparisons, decisions_out, spearman_rows)
    print("aggregated fold_rows={} oof_rows={} seed_rows={} configs={}".format(len(all_fold_rows), len(all_oof_rows), len(seed_rows), len(summary)))
    print("GLOBAL_ALIGNMENT_REPLICATED={} PAIR_WEIGHTS_NONUNIFORM={} PAIR_ASSIGNMENT_MATTERS={} PAIR_ADAPTIVE_SUPPORTED={} SPARSE_TOP3_SUPPORTED={}".format(
        decisions_out["global_status"], decisions_out["pair_weights_nonuniform"], decisions_out["pair_assignment_matters"], decisions_out["pair_adaptive_supported"], decisions_out["sparse_top3_supported"]
    ))
    return decisions_out


if __name__ == "__main__":
    main()
