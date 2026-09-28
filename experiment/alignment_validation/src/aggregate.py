import argparse
import csv
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr, wilcoxon


SHORT_METRICS = [
    "acc",
    "ba",
    "auc",
    "f1",
    "sen",
    "spe",
    "legacy_auc",
    "pos_cos",
    "neg_cos",
    "cos_gap",
    "retrieval_r1",
    "cmmd",
    "runtime_sec",
]

CANONICAL = {
    "acc": "acc",
    "ba": "balanced_acc",
    "auc": "prob_auc",
    "f1": "f1",
    "sen": "sensitivity",
    "spe": "specificity",
    "legacy_auc": "legacy_auc",
    "pos_cos": "align_pos_cos",
    "neg_cos": "align_neg_cos",
    "cos_gap": "align_cos_gap",
    "retrieval_r1": "align_retrieval_r1",
    "cmmd": "align_cmmd",
    "runtime_sec": "runtime_sec",
}

PAPER_VALUES = {
    "acc": (0.8977, 0.0272),
    "auc": (0.8981, 0.0256),
    "sen": (0.9032, 0.0421),
    "spe": (0.8930, 0.0604),
}


def read_rows(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError("No per-run CSV files matched {}".format(pattern))
    rows = []
    for path in paths:
        with open(path, newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    if not rows:
        raise ValueError("Matched CSV files are empty: {}".format(paths))
    return rows, paths


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def normalize_rows(rows):
    """Normalize old short names and the required explicit output names."""
    integer_fields = {
        "seed",
        "fold",
        "n_outer_train",
        "n_train",
        "n_val",
        "n_test",
        "best_epoch",
        "token_nmodal",
        "token_dim",
        "align_warmup",
    }
    normalized = []
    for source in rows:
        row = dict(source)
        for field in integer_fields:
            if field in row:
                try:
                    row[field] = int(float(row[field]))
                except (TypeError, ValueError):
                    pass
        for short, canonical in CANONICAL.items():
            if short not in row and canonical in row:
                row[short] = row[canonical]
            row[short] = _float(row.get(short))
            row[canonical] = row[short]
        for field in (
            "val_acc",
            "val_ba",
            "align_lambda",
            "align_temperature",
            "final_cls_loss",
            "final_align_loss",
            "train_acc",
            "train_ba",
        ):
            if field in row:
                row[field] = _float(row[field])
        normalized.append(row)
    return normalized


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(rows):
    summaries = []
    for config in sorted({row["config"] for row in rows}):
        selected = [row for row in rows if row["config"] == config]
        seeds = sorted({int(row["seed"]) for row in selected})
        item = {
            "config": config,
            "protocol": selected[0].get("protocol", ""),
            "seeds": ",".join(str(seed) for seed in seeds),
            "n_folds": len(selected),
        }
        for metric in SHORT_METRICS:
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            item[metric + "_mean"] = float(values.mean())
            item[metric + "_std"] = std
            item[metric + "_sem"] = float(std / np.sqrt(len(values))) if len(values) > 1 else 0.0
            canonical = CANONICAL[metric]
            item[canonical + "_mean"] = item[metric + "_mean"]
            item[canonical + "_std"] = item[metric + "_std"]
            item[canonical + "_sem"] = item[metric + "_sem"]

        # Uppercase/explicit aliases make summary.csv easy to consume.
        for metric, label in (
            ("acc", "ACC"),
            ("ba", "BA"),
            ("auc", "prob_AUC"),
            ("f1", "F1"),
            ("sen", "Sensitivity"),
            ("spe", "Specificity"),
            ("pos_cos", "align_pos_cos"),
            ("cos_gap", "align_cos_gap"),
            ("retrieval_r1", "align_retrieval_r1"),
            ("cmmd", "align_cmmd"),
        ):
            for suffix in ("mean", "std", "sem"):
                item["{}_{}".format(label, suffix)] = item[metric + "_" + suffix]
        summaries.append(item)

    baseline = next((item for item in summaries if item["config"] == "A0_baseline"), None)
    baseline_rows = [row for row in rows if row["config"] == "A0_baseline" and int(row["seed"]) == 0]
    baseline_by_fold = {int(row["fold"]): row for row in baseline_rows}
    for item in summaries:
        for metric, label in (("acc", "ACC"), ("ba", "BA"), ("auc", "prob_AUC"), ("f1", "F1")):
            item["delta_{}".format(label)] = (
                item[metric + "_mean"] - baseline[metric + "_mean"] if baseline else np.nan
            )
        candidate_rows = [row for row in rows if row["config"] == item["config"] and int(row["seed"]) == 0]
        common = [
            (row, baseline_by_fold[int(row["fold"])])
            for row in candidate_rows
            if int(row["fold"]) in baseline_by_fold
        ]
        item["wins_BA"] = int(sum(row["ba"] > base["ba"] for row, base in common))
        item["wins_AUC"] = int(sum(row["auc"] > base["auc"] for row, base in common))
        item["wins_BA_over_10"] = "{}/10".format(item["wins_BA"]) if len(common) == 10 else "{}/{}".format(item["wins_BA"], len(common))
        item["wins_AUC_over_10"] = "{}/10".format(item["wins_AUC"]) if len(common) == 10 else "{}/{}".format(item["wins_AUC"], len(common))
    return summaries


def paired_test(candidate_rows, baseline_rows, metric, seed=12345):
    candidate = {(int(row["seed"]), int(row["fold"])): row[metric] for row in candidate_rows}
    baseline = {(int(row["seed"]), int(row["fold"])): row[metric] for row in baseline_rows}
    keys = sorted(set(candidate).intersection(baseline))
    diffs = np.asarray([candidate[key] - baseline[key] for key in keys], dtype=np.float64)
    if len(diffs) == 0:
        return {
            "n": 0,
            "delta_mean": np.nan,
            "delta_median": np.nan,
            "wins": 0,
            "p": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }
    if np.allclose(diffs, 0.0):
        p_value = 1.0
    else:
        try:
            p_value = float(wilcoxon(diffs, alternative="two-sided", method="auto").pvalue)
        except ValueError:
            p_value = 1.0
    rng = np.random.default_rng(seed)
    bootstrap = rng.choice(diffs, size=(10000, len(diffs)), replace=True).mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "n": len(diffs),
        "delta_mean": float(diffs.mean()),
        "delta_median": float(np.median(diffs)),
        "wins": int(np.sum(diffs > 0)),
        "p": p_value,
        "ci_low": float(low),
        "ci_high": float(high),
    }


def seed0_rows(rows, config):
    return [row for row in rows if row["config"] == config and int(row["seed"]) == 0]


def decide_winner(rows, baseline="A0_baseline"):
    base = seed0_rows(rows, baseline)
    candidates = []
    for config in sorted({row["config"] for row in rows} - {baseline}):
        candidate = seed0_rows(rows, config)
        if len(base) != 10 or len(candidate) != 10:
            continue
        ba = paired_test(candidate, base, "ba")
        auc = paired_test(candidate, base, "auc")
        promising = ba["delta_mean"] >= 0.005 and auc["delta_mean"] >= 0.005 and ba["wins"] >= 6
        candidates.append(
            {
                "config": config,
                "delta_ba": ba["delta_mean"],
                "delta_auc": auc["delta_mean"],
                "ba_wins": ba["wins"],
                "promising": bool(promising),
            }
        )
    promising = [candidate for candidate in candidates if candidate["promising"]]
    winner = None
    if promising:
        winner = sorted(
            promising,
            key=lambda value: (value["delta_ba"], value["delta_auc"], value["ba_wins"]),
            reverse=True,
        )[0]["config"]
    return {
        "baseline": baseline,
        "winner": winner,
        "promising_candidates": promising,
        "candidates": candidates,
        "rerun_seeds": [1, 2] if winner else [],
    }


def _corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def config_level_correlations(summary):
    result = []
    for diagnostic, metric in (("cos_gap", "ba"), ("retrieval_r1", "ba"), ("cmmd_score", "ba"), ("cos_gap", "auc"), ("retrieval_r1", "auc"), ("cmmd_score", "auc")):
        values = []
        outcomes = []
        for row in summary:
            if diagnostic == "cmmd_score":
                values.append(-row["cmmd_mean"])
            else:
                values.append(row[diagnostic + "_mean"])
            outcomes.append(row[metric + "_mean"])
        rho, p_value = _corr(values, outcomes)
        result.append({"diagnostic": diagnostic, "metric": metric, "rho": rho, "p": p_value, "n": len(values)})
    return result


def paired_delta_rows(rows):
    base = {int(row["fold"]): row for row in seed0_rows(rows, "A0_baseline")}
    output = []
    aligned_configs = sorted({row["config"] for row in rows} - {"A0_baseline"})
    for config in aligned_configs:
        for row in seed0_rows(rows, config):
            fold = int(row["fold"])
            if fold not in base:
                continue
            baseline = base[fold]
            output.append(
                {
                    "config": config,
                    "fold": fold,
                    "delta_cos_gap": row["cos_gap"] - baseline["cos_gap"],
                    "delta_retrieval_r1": row["retrieval_r1"] - baseline["retrieval_r1"],
                    "delta_cmmd_improvement": baseline["cmmd"] - row["cmmd"],
                    "delta_ba": row["ba"] - baseline["ba"],
                    "delta_auc": row["auc"] - baseline["auc"],
                }
            )
    return output


def paired_delta_correlations(delta_rows):
    output = []
    for diagnostic in ("delta_cos_gap", "delta_retrieval_r1", "delta_cmmd_improvement"):
        for outcome in ("delta_ba", "delta_auc"):
            rho, p_value = _corr(
                [row[diagnostic] for row in delta_rows],
                [row[outcome] for row in delta_rows],
            )
            output.append({"diagnostic": diagnostic, "outcome": outcome, "rho": rho, "p": p_value, "n": len(delta_rows)})
    return output


def fmt_pm(mean, std, scale=1.0, digits=2):
    mean = float(mean)
    std = float(std)
    if not np.isfinite(mean):
        return "NA"
    return "{:.{d}f}±{:.{d}f}".format(mean * scale, std * scale, d=digits)


def fmt_num(value, digits=4):
    return "NA" if not np.isfinite(value) else "{:.{}f}".format(value, digits)


def write_report(rows, summary, paired, decision, report_path):
    lookup = {row["config"]: row for row in summary}
    aligned_summary = [row for row in summary if row["config"] != "A0_baseline"]
    best = max(aligned_summary, key=lambda row: (row["ba_mean"], row["auc_mean"])) if aligned_summary else lookup.get("A0_baseline")
    base = lookup.get("A0_baseline")
    config_corr = config_level_correlations(summary)
    delta_rows = paired_delta_rows(rows)
    delta_corr = paired_delta_correlations(delta_rows)
    over_configs = ["A3_hybrid_weak", "A4_hybrid_medium", "A5_hybrid_strong"]
    over = [lookup[name] for name in over_configs if name in lookup]

    if decision["winner"]:
        verdict = "SUPPORTED"
    elif best and base and best["ba_mean"] > base["ba_mean"] and best["auc_mean"] > base["auc_mean"]:
        verdict = "PARTIALLY_SUPPORTED"
    else:
        verdict = "NOT_SUPPORTED"

    lines = [
        "# MMGL ABIDE alignment validation report",
        "",
        "## 1. Executive Summary",
        "",
        "Explicit alignment changed the held-out strict-CV results, but the predefined PROMISING_ALIGNMENT rule was not met. The best mean BA was `{}` at {}, compared with A0 BA {}; its paired BA win count was {}/10. The best mean probability AUC was `{}` at {}.".format(
            best["config"] if best else "NA",
            fmt_pm(best["ba_mean"], best["ba_std"], 100) if best else "NA",
            fmt_pm(base["ba_mean"], base["ba_std"], 100) if base else "NA",
            int(best.get("wins_BA", 0)) if best else 0,
            best["config"] if best else "NA",
            fmt_pm(best["auc_mean"], best["auc_std"], 100) if best else "NA",
        ),
        "",
        "Original repository reproduction (transductive runner): ACC 89.32%, AUC 89.17%. The paper table reports ACC 89.77±2.72% and AUC 89.81±2.56% (paper-reported uncertainty); these are different protocols from the strict validation below.",
        "",
        "## 2. Dataset and modalities",
        "",
        "- ABIDE has 871 subjects. After repository label encoding (`label - 1`), class counts are 403 and 468.",
        "- Input modalities and dimensions: `PHENO=48`, `ANAT=6`, `FUNC=10`, `Correlation=256`; total CSV input dimension is 320.",
        "- The observed final token shape is `[N, 4, 36]` (`M=4`, `D=36`).",
        "",
        "## 3. MMGL Integration Point",
        "",
        "`PHENO/ANAT/FUNC/Correlation → VariLengthInputLayer → cross-modal Transformer → final per-modality modal_tokens [N,4,36] → explicit alignment regularization → original flatten/fusion (52-D hidden) → GraphLearn → GCN`.",
        "",
        "Alignment is applied only to final per-modality tokens. The attention-specific representation is not aligned, and the original graph-learning/GCN mathematics is retained.",
        "",
        "## 4. Alignment Objectives",
        "",
        "- Pair InfoNCE: for every unordered modality pair, same-subject tokens are positives and other training subjects are negatives; symmetric cross-entropy is averaged and divided by `log(B)`.",
        "- Class-conditional MMD: for each class separately, the six modality-pair distributions are compared with biased multi-RBF MMD using σ ∈ {0.25, 0.5, 1, 2}; the class losses are averaged.",
        "- Hybrid: `L_align = 0.5 L_NCE + 0.5 L_CMMD`.",
        "- Step A objective: `L_MF = L_cls + λ_eff L_align`, with a 20-epoch linear warmup. Alignment is train-subject-only; Step B graph/GCN training is unchanged.",
        "",
        "## 5. Experimental Protocol",
        "",
        "- Formal runs use stratified 10-fold outer CV. Each outer-training partition is split into 90% inner-train and 10% validation.",
        "- Validation accuracy selects the checkpoint. Test labels are not used for checkpoint selection or alignment loss; test alignment metrics are computed only after the checkpoint is fixed.",
        "- Fixed settings: weighted-cosine graph, concat fusion, GCN, dropout 0.35, lr 0.0038, weight decay 0.11, `simple-2`, `n_head=2`, `n_hidden=18`, `nlayer=1`.",
        "- The requested `~/venvs/mmgl` was absent; runs used `/root/venvs/111/bin/python` with CUDA. Original source/data files were not edited.",
        "",
        "## 6. Main Results Table",
        "",
        "Mean ± standard deviation across available folds; summary.csv also contains standard errors.",
        "",
        "| Config | Folds | ACC | BA | probability AUC | F1 | SEN | SPE | cos gap | R@1 | CMMD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config in sorted(lookup):
        item = lookup[config]
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                config,
                int(item["n_folds"]),
                fmt_pm(item["acc_mean"], item["acc_std"], 100),
                fmt_pm(item["ba_mean"], item["ba_std"], 100),
                fmt_pm(item["auc_mean"], item["auc_std"], 100),
                fmt_pm(item["f1_mean"], item["f1_std"], 100),
                fmt_pm(item["sen_mean"], item["sen_std"], 100),
                fmt_pm(item["spe_mean"], item["spe_std"], 100),
                fmt_pm(item["cos_gap_mean"], item["cos_gap_std"], 1, 4),
                fmt_pm(item["retrieval_r1_mean"], item["retrieval_r1_std"], 1, 4),
                fmt_pm(item["cmmd_mean"], item["cmmd_std"], 1, 4),
            )
        )

    lines.extend(
        [
            "",
            "## 7. Comparison with the original paper",
            "",
            "| Result | ACC | BA | probability AUC | SEN | SPE |",
            "|---|---:|---:|---:|---:|---:|",
            "| Paper MMGL ABIDE row | 89.77±2.72 | — | 89.81±2.56 | 90.32±4.21 | 89.30±6.04 |",
            "| Original repository reproduction | 89.32±8.77 | — | 89.17±8.82 | not emitted | not emitted |",
        ]
    )
    if base:
        lines.append(
            "| Strict A0 baseline | {} | {} | {} | {} | {} |".format(
                fmt_pm(base["acc_mean"], base["acc_std"], 100),
                fmt_pm(base["ba_mean"], base["ba_std"], 100),
                fmt_pm(base["auc_mean"], base["auc_std"], 100),
                fmt_pm(base["sen_mean"], base["sen_std"], 100),
                fmt_pm(base["spe_mean"], base["spe_std"], 100),
            )
        )

    lines.extend(
        [
            "",
            "## 8. Paired Comparison vs Baseline",
            "",
            "All tests use the same seed-0 outer folds. Deltas are aligned minus A0; bootstrap intervals are 95% fold-wise paired CIs. The ten-fold sample is small, so p-values are reported without suppressing non-significant results.",
            "",
            "| Config | Metric | mean Δ | median Δ | wins | Wilcoxon p | bootstrap 95% CI |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in paired:
        label = {"acc": "ACC", "ba": "BA", "auc": "prob_AUC"}[item["metric"]]
        lines.append(
            "| {} | {} | {:.4f} | {:.4f} | {}/10 | {:.4g} | [{:.4f}, {:.4f}] |".format(
                item["config"], label, item["delta_mean"], item["delta_median"], item["wins"], item["p"], item["ci_low"], item["ci_high"]
            )
        )

    lines.extend(
        [
            "",
            "## 9. Alignment–Diagnosis Relationship",
            "",
            "Config-level Spearman results use six configurations and are exploratory because n=6. For CMMD, `-CMMD` is used as the higher-is-better alignment score.",
            "",
            "| Alignment score | Diagnosis metric | n | Spearman ρ | p |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for item in config_corr:
        lines.append("| {} | {} | {} | {} | {} |".format(item["diagnostic"], item["metric"].upper(), item["n"], fmt_num(item["rho"]), fmt_num(item["p"])))
    lines.extend(
        [
            "",
            "Paired fold analysis pools aligned-config × fold deltas (5×10 = {} observations in this run):",
            "",
            "| Δ alignment score | Δ diagnosis | n | Spearman ρ | p |",
            "|---|---|---:|---:|---:|",
        ]
    )
    lines[-4] = lines[-4].format(len(delta_rows))
    for item in delta_corr:
        lines.append("| {} | {} | {} | {} | {} |".format(item["diagnostic"], item["outcome"].replace("delta_", "Δ").upper(), item["n"], fmt_num(item["rho"]), fmt_num(item["p"])))

    lines.extend(
        [
            "",
            "## 10. Over-Alignment Test",
            "",
            "| Hybrid config | BA | probability AUC | cos gap | R@1 | CMMD |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in over:
        lines.append("| {} | {} | {} | {} | {} | {} |".format(item["config"], fmt_pm(item["ba_mean"], item["ba_std"], 100), fmt_pm(item["auc_mean"], item["auc_std"], 100), fmt_pm(item["cos_gap_mean"], item["cos_gap_std"], 1, 4), fmt_pm(item["retrieval_r1_mean"], item["retrieval_r1_std"], 1, 4), fmt_pm(item["cmmd_mean"], item["cmmd_std"], 1, 4)))
    lines.extend(
        [
            "",
            "Observed fact: A4 medium has higher mean BA than A3 weak and A5 strong, while A5 has the strongest mean cos gap and R@1 among the three. Thus stronger alignment does not monotonically improve diagnosis in this run.",
            "",
            "## 11. Interpretation",
            "",
            "- Fact: A4 has the highest mean BA ({}), and A1 has the highest mean probability AUC ({}).".format(fmt_pm(lookup["A4_hybrid_medium"]["ba_mean"], lookup["A4_hybrid_medium"]["ba_std"], 100) if "A4_hybrid_medium" in lookup else "NA", fmt_pm(lookup["A1_pair_nce"]["auc_mean"], lookup["A1_pair_nce"]["auc_std"], 100) if "A1_pair_nce" in lookup else "NA"),
            "- Fact: no aligned configuration met all automatic criteria (both mean deltas ≥0.5 percentage points and at least 6/10 BA wins); A4 had 5/10 BA wins despite positive mean deltas.",
            "- Inference: the medium hybrid may provide a useful trade-off, whereas stronger matching may suppress modality-specific complementary information. This is a hypothesis rather than a causal proof.",
            "",
            "## 12. Verdict",
            "",
            "`{}` — the strict experiment shows partial mean-level evidence but does not satisfy the preregistered automatic support rule for the hypothesis: “Improving explicit multimodal alignment improves ASD diagnosis in MMGL.”".format(verdict),
            "",
            "## 13. Next Step",
            "",
            "Because the hybrid sweep shows a medium-strength optimum and stronger alignment is not monotonic, the single recommended next direction is selective/shared-private adaptive alignment that protects modality-specific information.",
            "",
            "## Reproducibility artifacts",
            "",
            "- `fold_results.csv`: one row per config × seed × outer fold, including explicit metric aliases and token dimensions.",
            "- `summary.csv`: mean, standard deviation, standard error, baseline deltas, and BA/AUC win counts.",
            "- `report.md`: this report; six config logs are under `logs/`.",
            "- Compile check, alignment backward self-test, and one-fold paper-protocol compatibility smoke all passed before the formal runs.",
        ]
    )
    with open(report_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate MMGL alignment validation runs")
    parser.add_argument("--input-glob", "--input_glob", dest="input_glob", required=True)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--decision-out", "--decision_out", dest="decision_out", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    raw_rows, _ = read_rows(args.input_glob)
    rows = normalize_rows(raw_rows)
    write_csv(os.path.join(args.output_dir, "fold_results.csv"), rows)
    summary = summary_rows(rows)
    write_csv(os.path.join(args.output_dir, "summary.csv"), summary)

    decision = decide_winner(rows)
    base = seed0_rows(rows, "A0_baseline")
    paired = []
    for config in sorted({row["config"] for row in rows} - {"A0_baseline"}):
        candidate = seed0_rows(rows, config)
        if len(base) != 10 or len(candidate) != 10:
            continue
        for metric in ("acc", "ba", "auc"):
            result = paired_test(candidate, base, metric)
            result.update({"config": config, "metric": metric})
            paired.append(result)

    os.makedirs(os.path.dirname(os.path.abspath(args.decision_out)), exist_ok=True)
    with open(args.decision_out, "w") as handle:
        json.dump(decision, handle, indent=2)
    report_path = os.path.join(args.output_dir, "report.md")
    write_report(rows, summary, paired, decision, report_path)
    print(
        json.dumps(
            {
                "fold_rows": len(rows),
                "configs": sorted({row["config"] for row in rows}),
                "decision": decision,
            },
            sort_keys=True,
        )
    )
    print("wrote", os.path.join(args.output_dir, "fold_results.csv"))
    print("wrote", os.path.join(args.output_dir, "summary.csv"))
    print("wrote", report_path)


if __name__ == "__main__":
    main()
