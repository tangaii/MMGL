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
    "D0_baseline",
    "D1_global_hybrid_medium",
    "D2_global_hybrid_strong",
    "D3_shared_projector_strong",
    "D4_disentangled_medium",
    "D5_disentangled_strong",
]
SEEDS = [0, 1, 2]
PRIMARY = [
    "acc", "balanced_acc", "asd_auc", "asd_f1",
    "asd_sensitivity", "nc_specificity",
]
DIAGNOSTICS = [
    "raw_pos_cos", "raw_neg_cos", "raw_cos_gap", "raw_retrieval_r1", "raw_cmmd",
    "raw_modality_probe_acc", "shared_pos_cos", "shared_neg_cos", "shared_cos_gap",
    "shared_retrieval_r1", "shared_cmmd", "shared_modality_probe_acc",
    "private_modality_probe_acc", "private_shared_probe_gap", "shared_mean_std",
    "private_mean_std", "reconstruction_nmse", "shared_private_xcov",
]
COMPARISONS = [
    ("D1_global_hybrid_medium", "D0_baseline"),
    ("D2_global_hybrid_strong", "D0_baseline"),
    ("D3_shared_projector_strong", "D2_global_hybrid_strong"),
    ("D4_disentangled_medium", "D1_global_hybrid_medium"),
    ("D5_disentangled_strong", "D2_global_hybrid_strong"),
    ("D5_disentangled_strong", "D3_shared_projector_strong"),
    ("D5_disentangled_strong", "D4_disentangled_medium"),
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
    indices = [int(row["subject_index"]) for row in rows]
    if indices != list(range(871)):
        raise ValueError("OOF subjects are missing, duplicated or out of order")
    y = np.asarray([int(row["true_asd"]) for row in rows], dtype=np.int64)
    p = np.asarray([float(row["p_asd"]) for row in rows], dtype=np.float64)
    pred = np.asarray([int(row["pred_asd"]) for row in rows], dtype=np.int64)
    return metric_arrays(y, p, pred)


def validate_raw(raw_dir):
    fold_paths = sorted(glob.glob(os.path.join(raw_dir, "*.folds.csv")))
    oof_paths = sorted(glob.glob(os.path.join(raw_dir, "*.oof.csv")))
    expected = {"{}_seed{}".format(config, seed) for config in CONFIGS for seed in SEEDS}
    fold_stems = {os.path.basename(p)[: -len(".folds.csv")] for p in fold_paths}
    oof_stems = {os.path.basename(p)[: -len(".oof.csv")] for p in oof_paths}
    if fold_stems != expected or oof_stems != expected:
        raise ValueError("Expected exactly 18 fold and OOF files")
    folds, oofs = {}, {}
    for stem in sorted(expected):
        fold_rows = read_csv(os.path.join(raw_dir, stem + ".folds.csv"))
        oof_rows = read_csv(os.path.join(raw_dir, stem + ".oof.csv"))
        if len(fold_rows) != 10:
            raise ValueError("{} has {} fold rows".format(stem, len(fold_rows)))
        if len(oof_rows) != 871:
            raise ValueError("{} has {} OOF rows".format(stem, len(oof_rows)))
        if len({int(row["subject_index"]) for row in oof_rows}) != 871:
            raise ValueError("{} OOF subject indices are not unique".format(stem))
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
            for metric in DIAGNOSTICS:
                row["fold_mean_{}".format(metric)], row["fold_std_{}".format(metric)] = mean_std(
                    [number(item, metric) for item in fold_rows]
                )
            row["runtime_sec_mean"], row["runtime_sec_std"] = mean_std(
                [number(item, "runtime_sec") for item in fold_rows]
            )
            row["base_init_sha256"] = fold_rows[0].get("base_init_sha256", "")
            row["aux_init_sha256"] = fold_rows[0].get("aux_init_sha256", "")
            rows.append(row)
    return rows


def config_summary_rows(seed_rows):
    rows = []
    for config in CONFIGS:
        selected = [row for row in seed_rows if row["config"] == config]
        row = {"config": config, "n_seeds": len(selected)}
        for metric in PRIMARY:
            row[metric + "_mean"], row[metric + "_std"] = mean_std(
                [number(item, metric) for item in selected]
            )
        for metric in DIAGNOSTICS:
            source = "fold_mean_{}".format(metric)
            row[metric + "_mean"], row[metric + "_std"] = mean_std(
                [number(item, source) for item in selected]
            )
        row["runtime_sec_mean"], row["runtime_sec_std"] = mean_std(
            [number(item, "runtime_sec_mean") for item in selected]
        )
        rows.append(row)
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
        numerator = np.sum(pos_counts * neg_before)
        numerator += 0.5 * np.sum(pos_counts * neg_counts)
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
        ba_deltas[cursor : cursor + size] = 0.5 * (
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
        "ba": summarize(
            ba_deltas,
            full_candidate["balanced_acc"] - full_reference["balanced_acc"],
        ),
        "auc": summarize(
            auc_deltas,
            full_candidate["asd_auc"] - full_reference["asd_auc"],
        ),
    }


def planned_comparisons(oofs):
    output = []
    for index, (candidate, reference) in enumerate(COMPARISONS):
        for seed in SEEDS:
            stats = bootstrap_metric_delta(
                oofs["{}_seed{}".format(candidate, seed)],
                oofs["{}_seed{}".format(reference, seed)],
                random_seed=9300 + index * 100 + seed,
            )
            output.append({
                "candidate": candidate, "reference": reference, "seed": seed,
                "ba": stats["ba"], "auc": stats["auc"],
            })
    return output


def comparison_stats(comparisons, candidate, reference):
    selected = [
        row for row in comparisons
        if row["candidate"] == candidate and row["reference"] == reference
    ]
    ba_values = [row["ba"]["mean_delta"] for row in selected]
    auc_values = [row["auc"]["mean_delta"] for row in selected]
    ba_mean, ba_std = mean_std(ba_values)
    auc_mean, auc_std = mean_std(auc_values)
    return {
        "rows": selected,
        "ba_mean": ba_mean, "ba_std": ba_std,
        "auc_mean": auc_mean, "auc_std": auc_std,
        "ba_wins": sum(value > 0 for value in ba_values),
    }


def decomposition_rule(summary_row, seed_rows, config):
    selected = [row for row in seed_rows if row["config"] == config]
    private_mean = summary_row["private_modality_probe_acc_mean"]
    gap_mean = summary_row["private_shared_probe_gap_mean"]
    shared_std_mean = summary_row["shared_mean_std_mean"]
    private_std_mean = summary_row["private_mean_std_mean"]
    recon_mean = summary_row["reconstruction_nmse_mean"]
    private_wins = sum(
        number(row, "fold_mean_private_modality_probe_acc")
        > number(row, "fold_mean_shared_modality_probe_acc")
        for row in selected
    )
    checks = {
        "private_probe": np.isfinite(private_mean) and private_mean >= 0.40,
        "gap": np.isfinite(gap_mean) and gap_mean >= 0.10 and private_wins >= 2,
        "shared_std": np.isfinite(shared_std_mean) and shared_std_mean > 0.10,
        "private_std": np.isfinite(private_std_mean) and private_std_mean > 0.10,
        "reconstruction": np.isfinite(recon_mean) and recon_mean < 1.0,
    }
    return {
        "working": all(checks.values()),
        "checks": checks,
        "private_wins": private_wins,
    }


def decisions(summary, seed_rows, comparisons):
    lookup = {row["config"]: row for row in summary}
    d1 = comparison_stats(comparisons, "D1_global_hybrid_medium", "D0_baseline")
    d2 = comparison_stats(comparisons, "D2_global_hybrid_strong", "D0_baseline")
    d3 = comparison_stats(comparisons, "D3_shared_projector_strong", "D2_global_hybrid_strong")
    d4 = comparison_stats(comparisons, "D4_disentangled_medium", "D1_global_hybrid_medium")
    d5 = comparison_stats(comparisons, "D5_disentangled_strong", "D2_global_hybrid_strong")
    d5_d3 = comparison_stats(comparisons, "D5_disentangled_strong", "D3_shared_projector_strong")
    d5_d4 = comparison_stats(comparisons, "D5_disentangled_strong", "D4_disentangled_medium")

    global_positive = [d1["ba_mean"] > 0 and d1["auc_mean"] > 0,
                       d2["ba_mean"] > 0 and d2["auc_mean"] > 0]
    any_global_positive = any(
        value > 0 for value in (d1["ba_mean"], d1["auc_mean"], d2["ba_mean"], d2["auc_mean"])
    )
    global_status = "YES" if any(global_positive) else ("PARTIAL" if any_global_positive else "NO")

    rules = {
        "D4": decomposition_rule(lookup["D4_disentangled_medium"], seed_rows, "D4_disentangled_medium"),
        "D5": decomposition_rule(lookup["D5_disentangled_strong"], seed_rows, "D5_disentangled_strong"),
    }

    def matched_pass(stats, rule):
        return (
            rule["working"] and stats["ba_mean"] >= 0.005
            and stats["ba_wins"] >= 2 and stats["auc_mean"] >= -0.0025
        )

    d4_pass, d5_pass = matched_pass(d4, rules["D4"]), matched_pass(d5, rules["D5"])
    if d4_pass or d5_pass:
        disentangled_status = "YES"
    elif rules["D4"]["working"] or rules["D5"]["working"]:
        positive_working = (
            (rules["D4"]["working"] and d4["ba_mean"] > 0)
            or (rules["D5"]["working"] and d5["ba_mean"] > 0)
        )
        disentangled_status = "PARTIAL" if positive_working else "NO"
    else:
        disentangled_status = "INCONCLUSIVE"

    if d3["ba_mean"] > 0 and d3["auc_mean"] > 0:
        projection_status = "POSITIVE"
    elif d3["ba_mean"] < 0 and d3["auc_mean"] < 0:
        projection_status = "NEGATIVE"
    else:
        projection_status = "NEUTRAL"

    matched = [("D4_disentangled_medium", "D1_global_hybrid_medium", d4),
               ("D5_disentangled_strong", "D2_global_hybrid_strong", d5)]
    best_matched = max(matched, key=lambda item: item[2]["ba_mean"])
    return {
        "global_status": global_status,
        "decomposition_rules": rules,
        "disentangled_status": disentangled_status,
        "projection_status": projection_status,
        "best_matched": best_matched,
        "stats": {"D1_vs_D0": d1, "D2_vs_D0": d2, "D3_vs_D2": d3,
                   "D4_vs_D1": d4, "D5_vs_D2": d5, "D5_vs_D3": d5_d3,
                   "D5_vs_D4": d5_d4},
    }


def fmt_pm(mean, std, scale=1.0, digits=2):
    if not np.isfinite(mean):
        return "NA"
    return "{:.{d}f}±{:.{d}f}".format(mean * scale, std * scale, d=digits)


def fmt(value, digits=4):
    return "NA" if not np.isfinite(value) else "{:.{}f}".format(value, digits)


def write_report(results_dir, summary, seed_rows, comparisons, decision):
    lookup = {row["config"]: row for row in summary}
    lines = [
        "# MMGL Experiment 3: True Shared–Private Disentangled Alignment", "",
        "## 1. Executive Summary", "",
        "This experiment tests whether a genuinely disentangled shared/private representation improves ASD diagnosis when only the shared representation is aligned. D4 and D5 use independent projection spaces, reconstruction, cross-covariance decorrelation and private modality classification.", "",
        "GLOBAL_ALIGNMENT_REPLICATED: **{}**".format(decision["global_status"]), "",
        "D4_DECOMPOSITION_WORKING: **{}**".format("YES" if decision["decomposition_rules"]["D4"]["working"] else "NO"), "",
        "D5_DECOMPOSITION_WORKING: **{}**".format("YES" if decision["decomposition_rules"]["D5"]["working"] else "NO"), "",
        "DISENTANGLED_ALIGNMENT_SUPPORTED: **{}**".format(decision["disentangled_status"]), "",
        "## 2. Previous Findings", "",
        "Experiment 2 found a positive global alignment trend, but its gate decomposition was invalid: gate means were approximately 0.5, gate standard deviations were around 1e-3, orthogonality loss was approximately 1, and raw and shared alignment metrics were nearly identical.", "",
        "## 3. Research Hypothesis", "",
        "Shared features should become more modality-invariant and receive cross-modal alignment. Private features should retain modality identity and complementary information while reconstructing the original MMGL token.", "",
        "## 4. Dataset", "",
        "ABIDE has 871 subjects: ASD 403 and NC 468. The four modalities are PHENO (48), ANAT (6), FUNC (10), and Correlation (256). The final MMGL modal token shape is `[N, 4, 36]`.", "",
        "## 5. Method", "",
        "```text",
        "H_m [36]",
        "│",
        "├── Original MMGL path → flatten → OutputLayer → GraphLearn → GCN → diagnosis",
        "├── Shared projector → Z_shared [18] → InfoNCE + CMMD",
        "└── Private projector_m → Z_private [18] → reconstruction + modality classification",
        "```", "",
        "The prediction path always uses the original complete modal tokens. The auxiliary modules are used only in Step A. D4 and D5 use lambda_rec=0.10, lambda_xcov=0.05, lambda_mod=0.05, 20-epoch warmup and auxiliary weight decay 1e-4.", "",
        "## 6. Why Experiment 2 Gate Failed", "",
        "The observed gate statistics and orthogonality values show that shared and private branches were almost scaled copies of the same token. Experiment 3 therefore uses parameter-independent projection spaces.", "",
        "## 7. Experimental Protocol", "",
        "Three seeds, strict stratified outer 10-fold CV, inner 10% validation, validation ACC checkpoint selection and subject-level OOF predictions were used. Test labels were excluded from training, alignment, and checkpoint selection. Transductive test features remain available to the MMGL graph construction.", "",
        "## 8. Main ASD Results", "",
        "Three-seed OOF mean ± standard deviation. ASD is the positive class.", "",
        "| Config | ACC | BA | ASD-AUC | ASD-F1 | ASD SEN | NC SPE |",
        "|---|---:|---:|---:|---:|---:|---:|",
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
        "", "## 9. Global Alignment Replication", "",
        "D1 and D2 replicate the medium and strong global hybrid conditions from Experiment 2. The integrity reference is D0 BA 83.79% / ASD-AUC 87.82%, D1 BA 84.24% / ASD-AUC 88.96%, and D2 BA 84.60% / ASD-AUC 89.38%; these are reference values, not tuning targets.", "",
        "## 10. Did Disentanglement Actually Work?", "",
        "The decomposition rule requires private probe ≥0.40, private-minus-shared probe gap ≥0.10 with private winning in at least 2/3 seeds, shared and private variation both >0.10, and reconstruction NMSE <1.0.", "",
        "| Config | Shared probe | Private probe | Probe gap | Shared std | Private std | Recon NMSE | Cross covariance | Working |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config in ("D4_disentangled_medium", "D5_disentangled_strong"):
        row = lookup[config]
        rule = decision["decomposition_rules"]["D4" if config.startswith("D4") else "D5"]
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            config, fmt_pm(row["shared_modality_probe_acc_mean"], row["shared_modality_probe_acc_std"]),
            fmt_pm(row["private_modality_probe_acc_mean"], row["private_modality_probe_acc_std"]),
            fmt_pm(row["private_shared_probe_gap_mean"], row["private_shared_probe_gap_std"]),
            fmt_pm(row["shared_mean_std_mean"], row["shared_mean_std_std"]),
            fmt_pm(row["private_mean_std_mean"], row["private_mean_std_std"]),
            fmt_pm(row["reconstruction_nmse_mean"], row["reconstruction_nmse_std"]),
            fmt_pm(row["shared_private_xcov_mean"], row["shared_private_xcov_std"]),
            "YES" if rule["working"] else "NO",
        ))
    lines += [
        "", "## 11. Alignment Quality", "",
        "Raw metrics use complete held-out modal tokens. Shared metrics use the projected shared tokens. Private representations are not evaluated with cross-modal similarity as a higher-is-better objective.", "",
        "| Config | Raw cos-gap | Raw R@1 | Raw CMMD | Shared cos-gap | Shared R@1 | Shared CMMD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for config in CONFIGS:
        row = lookup[config]
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            config, fmt_pm(row["raw_cos_gap_mean"], row["raw_cos_gap_std"]),
            fmt_pm(row["raw_retrieval_r1_mean"], row["raw_retrieval_r1_std"]),
            fmt_pm(row["raw_cmmd_mean"], row["raw_cmmd_std"]),
            fmt_pm(row["shared_cos_gap_mean"], row["shared_cos_gap_std"]),
            fmt_pm(row["shared_retrieval_r1_mean"], row["shared_retrieval_r1_std"]),
            fmt_pm(row["shared_cmmd_mean"], row["shared_cmmd_std"]),
        ))
    lines += [
        "", "## 12. Shared vs Private Modality Information", "",
        "Modality probe chance is 25%. The raw probe is computed on raw modal tokens; shared and private probes use the corresponding post-hoc representations.", "",
        "| Config | Raw probe | Shared probe | Private probe |",
        "|---|---:|---:|---:|",
    ]
    for config in CONFIGS:
        row = lookup[config]
        lines.append("| {} | {} | {} | {} |".format(
            config, fmt_pm(row["raw_modality_probe_acc_mean"], row["raw_modality_probe_acc_std"]),
            fmt_pm(row["shared_modality_probe_acc_mean"], row["shared_modality_probe_acc_std"]),
            fmt_pm(row["private_modality_probe_acc_mean"], row["private_modality_probe_acc_std"]),
        ))
    lines += ["", "## 13. Matched Global vs Disentangled", "",
              "Each planned comparison uses 10,000 paired subject-level bootstrap samples per seed. The bootstrap p value is exploratory because repeated-CV training sets overlap.", "",
              "| Comparison | Seed | Δ BA | BA 95% CI | BA p | Δ AUC | AUC 95% CI | AUC p |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for item in comparisons:
        lines.append("| {} vs {} | {} | {:.4f} | [{:.4f}, {:.4f}] | {:.4g} | {:.4f} | [{:.4f}, {:.4f}] | {:.4g} |".format(
            item["candidate"], item["reference"], item["seed"],
            item["ba"]["mean_delta"], item["ba"]["ci_low"], item["ba"]["ci_high"], item["ba"]["bootstrap_p"],
            item["auc"]["mean_delta"], item["auc"]["ci_low"], item["auc"]["ci_high"], item["auc"]["bootstrap_p"],
        ))
    lines += ["", "| Comparison | Mean Δ BA | Mean Δ AUC | BA seed wins |", "|---|---:|---:|---:|"]
    for name, (candidate, reference) in zip(
        ("D1 vs D0", "D2 vs D0", "D3 vs D2", "D4 vs D1", "D5 vs D2", "D5 vs D3", "D5 vs D4"), COMPARISONS
    ):
        stats = comparison_stats(comparisons, candidate, reference)
        lines.append("| {} | {} | {} | {}/3 |".format(
            name, fmt_pm(stats["ba_mean"], stats["ba_std"]),
            fmt_pm(stats["auc_mean"], stats["auc_std"]), stats["ba_wins"],
        ))
    lines += [
        "", "## 14. Projection-only Control", "",
        "D3 tests a common low-dimensional projection without a private branch. D5 is compared with D3 to test whether private preservation adds value.", "",
        "D3 vs D2: **{}**. D5 vs D3 mean ΔBA = {:.4f}, mean ΔAUC = {:.4f}.".format(
            decision["projection_status"], decision["stats"]["D5_vs_D3"]["ba_mean"], decision["stats"]["D5_vs_D3"]["auc_mean"]
        ), "",
        "## 15. ASD Sensitivity Analysis", "",
        "ASD sensitivity and NC specificity are reported together in the main table. Any BA change must therefore be read alongside the class-specific changes rather than interpreted from ACC alone.", "",
        "## 16. Verdict", "",
        "GLOBAL_ALIGNMENT_REPLICATED: **{}**".format(decision["global_status"]), "",
        "D4_DECOMPOSITION_WORKING: **{}**".format("YES" if decision["decomposition_rules"]["D4"]["working"] else "NO"), "",
        "D5_DECOMPOSITION_WORKING: **{}**".format("YES" if decision["decomposition_rules"]["D5"]["working"] else "NO"), "",
        "PROJECTION_ONLY_EFFECT: **{}**".format(decision["projection_status"]), "",
        "DISENTANGLED_ALIGNMENT_SUPPORTED: **{}**".format(decision["disentangled_status"]), "",
        "## 17. Scientific Interpretation", "",
        "Observed facts are the OOF diagnostic metrics, representation audit values, alignment metrics and paired bootstrap intervals above. Interpretation is limited to the fixed rules. A private representation with high modality probe accuracy is evidence of modality identity retention; it is not an ASD biomarker.", "",
    ]
    if decision["disentangled_status"] == "YES":
        next_step = "Transfer the fixed D4/D5 method to SPromptGL and test it on the stronger baseline."
    elif decision["disentangled_status"] == "PARTIAL":
        next_step = "Validate the fixed decomposition across another dataset or model before changing any hyperparameter."
    elif decision["disentangled_status"] == "NO":
        next_step = "Use global or interaction fusion for these four heterogeneous modalities and stop adding shared-private alignment terms."
    else:
        next_step = "Improve the representation decomposition design before drawing a diagnosis conclusion."
    lines += ["", "## 18. Next Step", "", next_step, "",
              "## Reproducibility", "",
              "The folder contains local ABIDE data copies, independent source files, 18 config-seed logs, and final CSV/report artifacts. Execution uses `/root/venvs/111/bin/python` on 32 CPU cores, 128 GB RAM and two 96 GB GPUs. The source environment `~/venvs/mmgl` was absent."]
    with open(os.path.join(results_dir, "report.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate MMGL Experiment 3")
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
    decision = decisions(summary, seed_rows, comparisons)
    write_report(args.results_dir, summary, seed_rows, comparisons, decision)
    print("aggregated fold_rows={} oof_rows={} seed_rows={} configs={}".format(
        len(all_fold_rows), len(all_oof_rows), len(seed_rows), len(summary)
    ))
    print("GLOBAL_ALIGNMENT_REPLICATED={} D4_DECOMPOSITION_WORKING={} D5_DECOMPOSITION_WORKING={} PROJECTION_ONLY_EFFECT={} DISENTANGLED_ALIGNMENT_SUPPORTED={}".format(
        decision["global_status"],
        "YES" if decision["decomposition_rules"]["D4"]["working"] else "NO",
        "YES" if decision["decomposition_rules"]["D5"]["working"] else "NO",
        decision["projection_status"], decision["disentangled_status"],
    ))


if __name__ == "__main__":
    main()
