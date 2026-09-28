import argparse
import csv
import glob
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


CONFIGS = [
    "C0_baseline",
    "C1_global_pair",
    "C2_global_hybrid_medium",
    "C3_global_hybrid_strong",
    "C4_selective_hybrid_medium",
    "C5_selective_hybrid_strong",
]
SEEDS = [0, 1, 2]
MODALITIES = ["PHENO", "ANAT", "FUNC", "Correlation"]
PRIMARY_METRICS = [
    "acc",
    "balanced_acc",
    "asd_auc",
    "asd_f1",
    "asd_sensitivity",
    "nc_specificity",
]
ALIGN_METRICS = [
    "raw_pos_cos",
    "raw_neg_cos",
    "raw_cos_gap",
    "raw_retrieval_r1",
    "raw_cmmd",
    "shared_pos_cos",
    "shared_neg_cos",
    "shared_cos_gap",
    "shared_retrieval_r1",
    "shared_cmmd",
]
COMPARISONS = [
    ("C1_global_pair", "C0_baseline"),
    ("C2_global_hybrid_medium", "C0_baseline"),
    ("C3_global_hybrid_strong", "C0_baseline"),
    ("C4_selective_hybrid_medium", "C2_global_hybrid_medium"),
    ("C5_selective_hybrid_strong", "C3_global_hybrid_strong"),
]


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def number(row, key, default=np.nan):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = []
    seen = set()
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


def oof_metrics(rows):
    rows = sorted(rows, key=lambda row: int(row["subject_index"]))
    indices = [int(row["subject_index"]) for row in rows]
    if indices != list(range(871)):
        raise ValueError("OOF subjects are missing or duplicated")
    y = np.asarray([int(row["true_asd"]) for row in rows], dtype=np.int64)
    p = np.asarray([float(row["p_asd"]) for row in rows], dtype=np.float64)
    pred = np.asarray([int(row["pred_asd"]) for row in rows], dtype=np.int64)
    return metric_arrays(y, p, pred)


def metric_arrays(y, p, pred):
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tp = int(np.sum((y == 1) & (pred == 1)))
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "acc": float(accuracy_score(y, pred)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "asd_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "asd_f1": float(f1_score(y, pred, zero_division=0)),
        "asd_sensitivity": float(sensitivity),
        "nc_specificity": float(specificity),
    }


def validate_raw(raw_dir):
    fold_paths = sorted(glob.glob(os.path.join(raw_dir, "*.folds.csv")))
    oof_paths = sorted(glob.glob(os.path.join(raw_dir, "*.oof.csv")))
    expected = {"{}_seed{}".format(config, seed) for config in CONFIGS for seed in SEEDS}
    fold_stems = {os.path.basename(path)[: -len(".folds.csv")] for path in fold_paths}
    oof_stems = {os.path.basename(path)[: -len(".oof.csv")] for path in oof_paths}
    if fold_stems != expected or oof_stems != expected:
        raise ValueError(
            "Expected 18 fold and OOF files, got folds={} oof={}".format(
                sorted(fold_stems), sorted(oof_stems)
            )
        )
    folds = {}
    oofs = {}
    for stem in sorted(expected):
        fold_rows = read_csv(os.path.join(raw_dir, stem + ".folds.csv"))
        oof_rows = read_csv(os.path.join(raw_dir, stem + ".oof.csv"))
        if len(fold_rows) != 10:
            raise ValueError("{} has {} fold rows".format(stem, len(fold_rows)))
        if len(oof_rows) != 871:
            raise ValueError("{} has {} OOF rows".format(stem, len(oof_rows)))
        if len({int(row["subject_index"]) for row in oof_rows}) != 871:
            raise ValueError("{} OOF subject indices are not unique".format(stem))
        folds[stem] = fold_rows
        oofs[stem] = oof_rows
    return folds, oofs


def seed_summary_rows(folds, oofs):
    output = []
    for config in CONFIGS:
        for seed in SEEDS:
            stem = "{}_seed{}".format(config, seed)
            fold_rows = folds[stem]
            oof_row = oof_metrics(oofs[stem])
            row = {"config": config, "seed": seed, "n_oof": 871, **oof_row}
            for metric in PRIMARY_METRICS + ALIGN_METRICS:
                values = [number(item, metric) for item in fold_rows]
                row["fold_mean_{}".format(metric)], row["fold_std_{}".format(metric)] = mean_std(values)
            for modality in MODALITIES:
                for suffix in ("mean", "std", "gt05"):
                    field = "gate_{}_{}".format(modality, suffix)
                    values = [number(item, field) for item in fold_rows]
                    row[field], row[field + "_std"] = mean_std(values) if suffix != "gt05" else (mean_std(values)[0], np.nan)
            row["gate_global_mean"], row["gate_global_std"] = mean_std([number(item, "gate_global_mean") for item in fold_rows])
            row["gate_global_std_across_folds"] = mean_std([number(item, "gate_global_std") for item in fold_rows])[0]
            output.append(row)
    return output


def config_summary_rows(seed_rows):
    output = []
    for config in CONFIGS:
        selected = [row for row in seed_rows if row["config"] == config]
        row = {"config": config, "n_seeds": len(selected)}
        for metric in PRIMARY_METRICS + ALIGN_METRICS:
            source_field = metric if metric in PRIMARY_METRICS else "fold_mean_{}".format(metric)
            mean, std = mean_std([row0.get(source_field, np.nan) for row0 in selected])
            row[metric + "_mean"] = mean
            row[metric + "_std"] = std
        for modality in MODALITIES:
            for suffix in ("mean", "std", "gt05"):
                field = "gate_{}_{}".format(modality, suffix)
                mean, std = mean_std([row0.get(field, np.nan) for row0 in selected])
                row[field + "_mean"] = mean
                row[field + "_std"] = std
        row["gate_global_mean_mean"], row["gate_global_mean_std"] = mean_std([row0.get("gate_global_mean", np.nan) for row0 in selected])
        output.append(row)
    return output


def bootstrap_metric_delta(candidate_rows, reference_rows, random_seed, n_bootstrap=10000):
    candidate = {int(row["subject_index"]): row for row in candidate_rows}
    reference = {int(row["subject_index"]): row for row in reference_rows}
    indices = np.arange(871)
    y = np.asarray([int(candidate[i]["true_asd"]) for i in indices], dtype=np.int64)
    cand_p = np.asarray([float(candidate[i]["p_asd"]) for i in indices], dtype=np.float64)
    ref_p = np.asarray([float(reference[i]["p_asd"]) for i in indices], dtype=np.float64)
    cand_pred = np.asarray([int(candidate[i]["pred_asd"]) for i in indices], dtype=np.int64)
    ref_pred = np.asarray([int(reference[i]["pred_asd"]) for i in indices], dtype=np.int64)
    candidate_full = metric_arrays(y, cand_p, cand_pred)
    reference_full = metric_arrays(y, ref_p, ref_pred)
    rng = np.random.default_rng(random_seed)
    ba_deltas = np.empty(n_bootstrap, dtype=np.float64)
    auc_deltas = np.empty(n_bootstrap, dtype=np.float64)
    chunk_size = 250

    def auc_from_counts(counts, scores):
        order = np.argsort(scores, kind="mergesort")
        ordered_counts = counts[order].astype(np.float64)
        ordered_y = y[order]
        pos_counts = ordered_counts * (ordered_y == 1)
        neg_counts = ordered_counts * (ordered_y == 0)
        neg_before = np.cumsum(neg_counts) - neg_counts
        numerator = np.sum(pos_counts * neg_before)
        numerator += 0.5 * np.sum(pos_counts * neg_counts)
        positives = pos_counts.sum()
        negatives = neg_counts.sum()
        return numerator / (positives * negatives) if positives and negatives else 0.5

    cursor = 0
    while cursor < n_bootstrap:
        size = min(chunk_size, n_bootstrap - cursor)
        sample = rng.integers(0, 871, size=(size, 871))
        ys = y[sample]
        cand_pred_s = cand_pred[sample]
        ref_pred_s = ref_pred[sample]
        pos_count = np.maximum(ys.sum(axis=1), 1)
        neg_count = np.maximum(ys.shape[1] - ys.sum(axis=1), 1)
        cand_tp = ((ys == 1) & (cand_pred_s == 1)).sum(axis=1)
        cand_tn = ((ys == 0) & (cand_pred_s == 0)).sum(axis=1)
        ref_tp = ((ys == 1) & (ref_pred_s == 1)).sum(axis=1)
        ref_tn = ((ys == 0) & (ref_pred_s == 0)).sum(axis=1)
        cand_ba = 0.5 * (cand_tp / pos_count + cand_tn / neg_count)
        ref_ba = 0.5 * (ref_tp / pos_count + ref_tn / neg_count)
        ba_deltas[cursor : cursor + size] = cand_ba - ref_ba
        for local in range(size):
            counts = np.bincount(sample[local], minlength=871)
            auc_deltas[cursor + local] = auc_from_counts(counts, cand_p) - auc_from_counts(counts, ref_p)
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
        "ba": summarize(
            ba_deltas,
            candidate_full["balanced_acc"] - reference_full["balanced_acc"],
        ),
        "auc": summarize(
            auc_deltas,
            candidate_full["asd_auc"] - reference_full["asd_auc"],
        ),
        "candidate_full": candidate_full,
        "reference_full": reference_full,
    }


def planned_comparisons(oofs):
    output = []
    for comparison_index, (candidate, reference) in enumerate(COMPARISONS):
        for seed in SEEDS:
            cand_rows = oofs["{}_seed{}".format(candidate, seed)]
            ref_rows = oofs["{}_seed{}".format(reference, seed)]
            stats = bootstrap_metric_delta(
                cand_rows,
                ref_rows,
                random_seed=7100 + comparison_index * 100 + seed,
            )
            output.append(
                {
                    "candidate": candidate,
                    "reference": reference,
                    "seed": seed,
                    "ba": stats["ba"],
                    "auc": stats["auc"],
                }
            )
    return output


def aggregate_metric(comparisons, key):
    values = [item[key]["mean_delta"] for item in comparisons]
    return float(np.mean(values)), float(np.std(values, ddof=1))


def fmt_pm(mean, std, scale=1.0, digits=2):
    if not np.isfinite(mean):
        return "NA"
    return "{:.{d}f}±{:.{d}f}".format(mean * scale, std * scale, d=digits)


def fmt(value, digits=4):
    return "NA" if not np.isfinite(value) else "{:.{}f}".format(value, digits)


def decision(summary, comparisons):
    lookup = {row["config"]: row for row in summary}

    def compare(candidate, reference):
        rows = [
            item
            for item in comparisons
            if item["candidate"] == candidate and item["reference"] == reference
        ]
        ba = aggregate_metric(rows, "ba")
        auc = aggregate_metric(rows, "auc")
        return rows, ba, auc

    c1_rows, c1_ba, c1_auc = compare("C1_global_pair", "C0_baseline")
    c2_rows, c2_ba, c2_auc = compare("C2_global_hybrid_medium", "C0_baseline")
    c3_rows, c3_ba, c3_auc = compare("C3_global_hybrid_strong", "C0_baseline")
    c4_rows, c4_ba, c4_auc = compare("C4_selective_hybrid_medium", "C2_global_hybrid_medium")
    c5_rows, c5_ba, c5_auc = compare("C5_selective_hybrid_strong", "C3_global_hybrid_strong")

    global_positive = [
        c1_ba[0] > 0 and c1_auc[0] > 0,
        c2_ba[0] > 0 and c2_auc[0] > 0,
    ]
    any_global_metric_positive = any(
        delta > 0
        for delta in (c1_ba[0], c1_auc[0], c2_ba[0], c2_auc[0])
    )
    if any(global_positive):
        global_status = "YES"
    elif any_global_metric_positive:
        global_status = "PARTIAL"
    else:
        global_status = "NO"

    c2 = lookup["C2_global_hybrid_medium"]
    c3 = lookup["C3_global_hybrid_strong"]
    over_alignment_better = (
        c3["raw_cos_gap_mean"] > c2["raw_cos_gap_mean"]
        and c3["raw_retrieval_r1_mean"] > c2["raw_retrieval_r1_mean"]
        and c3["raw_cmmd_mean"] < c2["raw_cmmd_mean"]
    )
    over_diagnosis_drop = c3["balanced_acc_mean"] < c2["balanced_acc_mean"] and c3["asd_auc_mean"] < c2["asd_auc_mean"]
    over_status = "YES" if over_alignment_better and over_diagnosis_drop else "NO"

    def selective_rule(comp_rows, comp_ba, comp_auc, config):
        gate_values = [lookup[config]["gate_{}_mean_mean".format(name)] for name in MODALITIES]
        gate_ok = all(np.isfinite(value) and 0.10 <= value <= 0.90 for value in gate_values)
        seed_wins = sum(item["ba"]["mean_delta"] > 0 for item in comp_rows)
        full = comp_ba[0] >= 0.005 and seed_wins >= 2 and comp_auc[0] >= -0.0025 and gate_ok
        positive = comp_ba[0] > 0 and comp_auc[0] >= -0.0025 and gate_ok
        return full, positive, gate_ok, seed_wins, gate_values

    c4_rule = selective_rule(c4_rows, c4_ba, c4_auc, "C4_selective_hybrid_medium")
    c5_rule = selective_rule(c5_rows, c5_ba, c5_auc, "C5_selective_hybrid_strong")
    if c4_rule[0] or c5_rule[0]:
        selective_status = "YES"
    elif c4_rule[1] or c5_rule[1]:
        selective_status = "PARTIAL"
    else:
        selective_status = "NO"

    return {
        "global_status": global_status,
        "over_status": over_status,
        "selective_status": selective_status,
        "global_drop_ba": c3["balanced_acc_mean"] - c2["balanced_acc_mean"],
        "selective_drop_ba": lookup["C5_selective_hybrid_strong"]["balanced_acc_mean"] - lookup["C4_selective_hybrid_medium"]["balanced_acc_mean"],
        "pair_stats": {
            "C1_vs_C0": (c1_rows, c1_ba, c1_auc),
            "C2_vs_C0": (c2_rows, c2_ba, c2_auc),
            "C3_vs_C0": (c3_rows, c3_ba, c3_auc),
            "C4_vs_C2": (c4_rows, c4_ba, c4_auc),
            "C5_vs_C3": (c5_rows, c5_ba, c5_auc),
        },
        "rules": {
            "C4": c4_rule,
            "C5": c5_rule,
        },
    }


def write_report(results_dir, summary, seed_rows, comparisons, decisions):
    lookup = {row["config"]: row for row in summary}
    lines = [
        "# MMGL Selective Alignment Validation",
        "",
        "## 1. Executive Summary",
        "",
        "This experiment tests whether aligning only a learned shared part of each modality token is better for ASD diagnosis than globally aligning the complete token. The primary results use 871 subject-level OOF predictions from strict transductive 10-fold CV over three independent seeds.",
        "",
        "Global alignment status: **{}**. Over-alignment status: **{}**. Selective alignment status: **{}**.".format(decisions["global_status"], decisions["over_status"], decisions["selective_status"]),
        "",
        "## 2. Motivation",
        "",
        "The first-stage experiment showed stronger alignment metrics from medium to strong hybrid alignment, while BA fell from 86.03% to 84.44%. That pattern motivates separating shared information from modality-specific information instead of forcing the complete token to agree across modalities.",
        "",
        "## 3. Protocol Corrections",
        "",
        "- RNG is reset independently at the start of every fold before module initialization, then reset again after all modules are initialized before the epoch loop.",
        "- Class weights use `np.bincount(..., minlength=2)`, so weights always map to internal classes 0 and 1 in the correct order.",
        "- DX_GROUP 1 is ASD and becomes internal class 0. All reported AUC, F1 and sensitivity metrics use ASD as the positive class; specificity is NC-specificity.",
        "",
        "## 4. Dataset",
        "",
        "- ABIDE: 871 subjects; ASD 403 and NC 468.",
        "- PHENO 48, ANAT 6, FUNC 10, Correlation 256.",
        "- Final MMGL token shape: `[N, 4, 36]`.",
        "",
        "## 5. Method",
        "",
        "```text",
        "H_m",
        "├─ original MMGL prediction path → flatten → OutputLayer → GraphLearn → GCN",
        "└─ SelectiveSharedPrivateGate",
        "   ├─ shared = gate × H_m → hybrid alignment",
        "   └─ private = (1 − gate) × H_m → no cross-modal alignment",
        "```",
        "The original prediction feature is unchanged. The gate, orthogonality loss and balance loss are used only during ModalFusion training.",
        "",
        "## 6. Configurations",
        "",
        "| Config | Strategy | Alignment | λ |",
        "|---|---|---|---:|",
        "| C0_baseline | none | none | 0.00 |",
        "| C1_global_pair | global | pair_nce | 0.20 |",
        "| C2_global_hybrid_medium | global | hybrid | 0.20 |",
        "| C3_global_hybrid_strong | global | hybrid | 0.50 |",
        "| C4_selective_hybrid_medium | selective | hybrid | 0.20 |",
        "| C5_selective_hybrid_strong | selective | hybrid | 0.50 |",
        "",
        "Selective runs use λ_orth=0.02, λ_balance=0.05, target shared ratio=0.50, and 20-epoch warmup. All other MMGL hyperparameters are fixed to the previous strict experiment.",
        "",
        "## 7. OOF Main Results",
        "",
        "Three-seed mean ± standard deviation from subject-level OOF predictions.",
        "",
        "| Config | ACC | BA | ASD-AUC | ASD-F1 | ASD Sensitivity | NC Specificity |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for config in CONFIGS:
        row = lookup[config]
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                config,
                fmt_pm(row["acc_mean"], row["acc_std"], 100),
                fmt_pm(row["balanced_acc_mean"], row["balanced_acc_std"], 100),
                fmt_pm(row["asd_auc_mean"], row["asd_auc_std"], 100),
                fmt_pm(row["asd_f1_mean"], row["asd_f1_std"], 100),
                fmt_pm(row["asd_sensitivity_mean"], row["asd_sensitivity_std"], 100),
                fmt_pm(row["nc_specificity_mean"], row["nc_specificity_std"], 100),
            )
        )

    lines.extend(
        [
            "",
            "## 8. Alignment Results",
            "",
            "Raw metrics are calculated on the complete held-out modal tokens. Shared metrics are calculated on the gated shared tokens for C4 and C5.",
            "",
            "| Config | Raw cos-gap | Raw R@1 | Raw CMMD | Shared cos-gap | Shared R@1 | Shared CMMD |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for config in CONFIGS:
        row = lookup[config]
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                config,
                fmt_pm(row["raw_cos_gap_mean"], row["raw_cos_gap_std"], 1, 4),
                fmt_pm(row["raw_retrieval_r1_mean"], row["raw_retrieval_r1_std"], 1, 4),
                fmt_pm(row["raw_cmmd_mean"], row["raw_cmmd_std"], 1, 4),
                fmt_pm(row["shared_cos_gap_mean"], row["shared_cos_gap_std"], 1, 4),
                fmt_pm(row["shared_retrieval_r1_mean"], row["shared_retrieval_r1_std"], 1, 4),
                fmt_pm(row["shared_cmmd_mean"], row["shared_cmmd_std"], 1, 4),
            )
        )

    lines.extend(
        [
            "",
            "## 9. Gate Analysis",
            "",
            "A high gate means that a latent component participates more strongly in the shared alignment regularizer. It is not a biomarker attribution and is not mapped directly to a brain region.",
            "",
            "| Config | PHENO | ANAT | FUNC | Correlation |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for config in ("C4_selective_hybrid_medium", "C5_selective_hybrid_strong"):
        row = lookup[config]
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                config,
                fmt_pm(row["gate_PHENO_mean_mean"], row["gate_PHENO_mean_std"], 1, 4),
                fmt_pm(row["gate_ANAT_mean_mean"], row["gate_ANAT_mean_std"], 1, 4),
                fmt_pm(row["gate_FUNC_mean_mean"], row["gate_FUNC_mean_std"], 1, 4),
                fmt_pm(row["gate_Correlation_mean_mean"], row["gate_Correlation_mean_std"], 1, 4),
            )
        )

    lines.extend(
        [
            "",
            "## 10. Planned Paired Comparisons",
            "",
            "Each comparison uses paired OOF subjects within a seed and 10,000 paired subject-level bootstrap samples. The bootstrap p value is exploratory; repeated-CV training sets overlap.",
            "",
            "| Comparison | Seed | Δ BA | BA 95% CI | BA p | Δ AUC | AUC 95% CI | AUC p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for comparison in comparisons:
        lines.append(
            "| {} vs {} | {} | {:.4f} | [{:.4f}, {:.4f}] | {:.4g} | {:.4f} | [{:.4f}, {:.4f}] | {:.4g} |".format(
                comparison["candidate"],
                comparison["reference"],
                comparison["seed"],
                comparison["ba"]["mean_delta"],
                comparison["ba"]["ci_low"],
                comparison["ba"]["ci_high"],
                comparison["ba"]["bootstrap_p"],
                comparison["auc"]["mean_delta"],
                comparison["auc"]["ci_low"],
                comparison["auc"]["ci_high"],
                comparison["auc"]["bootstrap_p"],
            )
        )
    lines.extend(
        [
            "",
            "| Comparison | Mean Δ BA across seeds | Mean Δ AUC across seeds | Positive BA seeds |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, (candidate, reference) in zip(
        ("C1 vs C0", "C2 vs C0", "C3 vs C0", "C4 vs C2", "C5 vs C3"), COMPARISONS
    ):
        selected = [item for item in comparisons if item["candidate"] == candidate and item["reference"] == reference]
        mean_ba, std_ba = aggregate_metric(selected, "ba")
        mean_auc, std_auc = aggregate_metric(selected, "auc")
        wins = sum(item["ba"]["mean_delta"] > 0 for item in selected)
        lines.append("| {} | {} | {} | {}/3 |".format(name, fmt_pm(mean_ba, std_ba), fmt_pm(mean_auc, std_auc), wins))

    lines.extend(
        [
            "",
            "## 11. Over-Alignment Replication",
            "",
            "C3 is compared with C2. The observed decision is **{}**. Global ΔBA (C3−C2) = {:.4f}; selective ΔBA (C5−C4) = {:.4f}.".format(decisions["over_status"], decisions["global_drop_ba"], decisions["selective_drop_ba"]),
            "",
            "## 12. Does Selective Alignment Fix It?",
            "",
            "C4 vs C2 tests matched medium strength and C5 vs C3 tests matched strong strength. The preregistered selective rule requires mean BA gain ≥0.5 percentage points, positive BA deltas in at least 2/3 seeds, mean AUC delta ≥−0.25 percentage points, and no gate collapse.",
            "",
            "- C4 rule: {}.".format("PASS" if decisions["rules"]["C4"][0] else ("PARTIAL" if decisions["rules"]["C4"][1] else "FAIL")),
            "- C5 rule: {}.".format("PASS" if decisions["rules"]["C5"][0] else ("PARTIAL" if decisions["rules"]["C5"][1] else "FAIL")),
            "",
            "## 13. Interpretation",
            "",
            "Observed facts are the OOF metrics, alignment metrics, gate means and paired bootstrap intervals reported above. The statuses use the fixed rules in the experiment prompt.",
            "",
            "Interpretation: selective alignment can support the hypothesis only when shared-token gains coincide with stable diagnosis and non-collapsed gates. Gate differences describe participation in the regularizer; they do not identify disease biology.",
            "",
            "## 14. Verdict",
            "",
            "GLOBAL_ALIGNMENT_REPLICATED: **{}**".format(decisions["global_status"]),
            "",
            "OVER_ALIGNMENT_REPLICATED: **{}**".format(decisions["over_status"]),
            "",
            "SELECTIVE_ALIGNMENT_SUPPORTED: **{}**".format(decisions["selective_status"]),
            "",
            "## 15. Next Step",
            "",
            "{}".format(
                "Transfer the fixed selective configuration to SPromptGL and test whether the shared/private separation remains useful on the stronger baseline."
                if decisions["selective_status"] == "YES"
                else "Stop adding alignment terms to the four heterogeneous MMGL modalities until a selective/shared-private design is justified by a new representation analysis."
            ),
            "",
            "## Reproducibility",
            "",
            "The experiment folder includes local ABIDE copies, independent source files, `run_all.sh`, and logs for all 18 config-seed jobs. The requested `~/venvs/mmgl` environment was absent; execution uses `/root/venvs/111/bin/python` on 32 CPU cores, 128 GB RAM and two GPUs.",
        ]
    )
    with open(os.path.join(results_dir, "report.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate selective alignment experiment")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    folds, oofs = validate_raw(args.raw_dir)
    all_fold_rows = [row for config in CONFIGS for seed in SEEDS for row in folds["{}_seed{}".format(config, seed)]]
    all_oof_rows = [row for config in CONFIGS for seed in SEEDS for row in oofs["{}_seed{}".format(config, seed)]]
    write_csv(os.path.join(args.results_dir, "fold_results.csv"), all_fold_rows)
    write_csv(os.path.join(args.results_dir, "oof_predictions.csv"), all_oof_rows)

    seed_rows = seed_summary_rows(folds, oofs)
    write_csv(os.path.join(args.results_dir, "seed_summary.csv"), seed_rows)
    summary = config_summary_rows(seed_rows)
    write_csv(os.path.join(args.results_dir, "summary.csv"), summary)

    comparisons = planned_comparisons(oofs)
    decisions = decision(summary, comparisons)
    write_report(args.results_dir, summary, seed_rows, comparisons, decisions)
    print(
        "aggregated fold_rows={} oof_rows={} seed_rows={} configs={}".format(
            len(all_fold_rows), len(all_oof_rows), len(seed_rows), len(summary)
        )
    )
    print(
        "GLOBAL_ALIGNMENT_REPLICATED={} OVER_ALIGNMENT_REPLICATED={} SELECTIVE_ALIGNMENT_SUPPORTED={}".format(
            decisions["global_status"], decisions["over_status"], decisions["selective_status"]
        )
    )


if __name__ == "__main__":
    main()
