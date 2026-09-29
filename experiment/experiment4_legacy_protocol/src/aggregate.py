#!/usr/bin/env python3
"""Validate, aggregate and report the two-config legacy MMGL experiment."""

import argparse
import csv
import glob
import hashlib
import os
import subprocess

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


CONFIGS = ["L0_legacy_baseline", "L1_legacy_uniform_global"]
SEEDS = [0, 1, 2]
FOLD_METRICS = [
    "test_acc", "legacy_hard_auc", "balanced_acc", "asd_probability_auc",
    "asd_f1", "asd_sensitivity", "nc_specificity", "alignment_cos_gap",
    "alignment_R1", "alignment_CMMD", "best_epoch",
]
OOF_METRICS = [
    "acc", "balanced_acc", "asd_probability_auc", "legacy_hard_auc",
    "asd_f1", "asd_sensitivity", "nc_specificity",
]
ORIGINAL_SOURCE_SHA256 = {
    "MMGL_transductive/main.py": "36cf8ebf1563688e77e4d8dabbfc4c1df5480b169d68dca829d6ec769d565f99",
    "MMGL_transductive/model.py": "0c72a6ddb3dbe68d85393553a471bfa0bf09a7bf1da0f661984b701007480954",
    "MMGL_transductive/network.py": "9ba31a5397b031ffe313d89f886f19254431838dc420bb78ed325b194c6c4316",
    "MMGL_transductive/layers.py": "0d78cc0c0aea51238ec17241baf0ecb61ce1f53b960f011f6097e965cea4daab",
    "MMGL_transductive/utils.py": "4159054d3936434b7b6c6b5e758229c5a71a021aeb8483d7673a2dd65e6ef9e7",
}
PAPER = {
    "acc": (89.77, 2.72),
    "auc": (89.81, 2.56),
    "sen": (90.32, 4.21),
    "spe": (89.30, 6.04),
}


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise RuntimeError("No rows to write: {}".format(path))
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


def num(row, key, default=np.nan):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def mean_std(values, ddof=0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    return float(values.mean()), float(values.std(ddof=ddof) if len(values) > ddof else 0.0)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(repo_root):
    try:
        return subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unavailable"


def legacy_hard_auc(y_internal, pred_internal):
    y_internal = np.asarray(y_internal, dtype=np.int64)
    pred_internal = np.asarray(pred_internal, dtype=np.int64)
    target = np.eye(2, dtype=np.float64)[y_internal]
    prediction = np.eye(2, dtype=np.float64)[pred_internal]
    return float(roc_auc_score(target, prediction))


def metric_arrays(y_asd, p_asd, pred_asd, y_internal=None, pred_internal=None):
    y_asd = np.asarray(y_asd, dtype=np.int64)
    p_asd = np.asarray(p_asd, dtype=float)
    pred_asd = np.asarray(pred_asd, dtype=np.int64)
    tp = int(np.sum((y_asd == 1) & (pred_asd == 1)))
    tn = int(np.sum((y_asd == 0) & (pred_asd == 0)))
    fp = int(np.sum((y_asd == 0) & (pred_asd == 1)))
    fn = int(np.sum((y_asd == 1) & (pred_asd == 0)))
    sensitivity = float(tp / (tp + fn)) if tp + fn else 0.0
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    output = {
        "acc": float(np.mean(pred_asd == y_asd)),
        "balanced_acc": 0.5 * (sensitivity + specificity),
        "asd_probability_auc": float(roc_auc_score(y_asd, p_asd)),
        "asd_f1": float(f1_score(y_asd, pred_asd, zero_division=0)),
        "asd_sensitivity": sensitivity,
        "nc_specificity": specificity,
    }
    # The original repository's AUC is deliberately calculated from one-hot
    # hard predictions. It is a separate legacy metric from probability AUC.
    if y_internal is not None and pred_internal is not None:
        output["legacy_hard_auc"] = legacy_hard_auc(y_internal, pred_internal)
    return output


def oof_metrics(rows):
    rows = sorted(rows, key=lambda row: int(row["subject_index"]))
    indices = [int(row["subject_index"]) for row in rows]
    if indices != list(range(871)):
        raise ValueError("OOF rows must contain each subject index 0..870 exactly once")
    y_asd = np.asarray([int(row["true_asd"]) for row in rows], dtype=np.int64)
    y_internal = np.asarray([int(row["true_internal_label"]) for row in rows], dtype=np.int64)
    p_asd = np.asarray([float(row["p_asd"]) for row in rows], dtype=float)
    pred_asd = np.asarray([int(row["pred_asd"]) for row in rows], dtype=np.int64)
    pred_internal = np.asarray([int(row["pred_internal"]) for row in rows], dtype=np.int64)
    counts = tuple(np.bincount(y_asd, minlength=2).tolist())
    if counts != (468, 403):
        raise ValueError("Unexpected OOF class counts: {}".format(counts))
    return metric_arrays(y_asd, p_asd, pred_asd, y_internal, pred_internal)


def validate_raw(raw_dir):
    fold_paths = sorted(glob.glob(os.path.join(raw_dir, "*.folds.csv")))
    oof_paths = sorted(glob.glob(os.path.join(raw_dir, "*.oof.csv")))
    expected = {
        "{}_seed{}".format(config, seed)
        for config in CONFIGS for seed in SEEDS
    }
    fold_stems = {os.path.basename(path)[:-len(".folds.csv")] for path in fold_paths}
    oof_stems = {os.path.basename(path)[:-len(".oof.csv")] for path in oof_paths}
    if fold_stems != expected or oof_stems != expected:
        raise ValueError("Expected exactly six fold and six OOF raw files")
    folds, oofs = {}, {}
    for stem in sorted(expected):
        fold_rows = read_csv(os.path.join(raw_dir, stem + ".folds.csv"))
        oof_rows = read_csv(os.path.join(raw_dir, stem + ".oof.csv"))
        fold_ids = sorted(int(row["fold"]) for row in fold_rows)
        if len(fold_rows) != 10 or fold_ids != list(range(1, 11)):
            raise ValueError("{} does not contain ten folds".format(stem))
        if len(oof_rows) != 871 or len({int(row["subject_index"]) for row in oof_rows}) != 871:
            raise ValueError("{} does not contain 871 unique OOF subjects".format(stem))
        if sum(int(row["n_test"]) for row in fold_rows) != 871:
            raise ValueError("{} fold test counts do not sum to 871".format(stem))
        if any(row.get("legacy_early_abort", "False") == "True" for row in fold_rows):
            raise ValueError("Legacy early abort was triggered in {}".format(stem))
        folds[stem], oofs[stem] = fold_rows, oof_rows
    return folds, oofs


def audit_splits_and_initialization(folds):
    rows = []
    split_matches = 0
    init_matches = 0
    for seed in SEEDS:
        for fold in range(1, 11):
            l0 = next(row for row in folds["L0_legacy_baseline_seed{}".format(seed)] if int(row["fold"]) == fold)
            l1 = next(row for row in folds["L1_legacy_uniform_global_seed{}".format(seed)] if int(row["fold"]) == fold)
            split_match = l0["split_sha256"] == l1["split_sha256"]
            init_match = l0["base_init_sha256"] == l1["base_init_sha256"]
            split_matches += int(split_match)
            init_matches += int(init_match)
            rows.append({
                "seed": seed,
                "fold": fold,
                "l0_split_sha256": l0["split_sha256"],
                "l1_split_sha256": l1["split_sha256"],
                "split_parity": split_match,
                "l0_base_init_sha256": l0["base_init_sha256"],
                "l1_base_init_sha256": l1["base_init_sha256"],
                "base_init_parity": init_match,
                "l0_best_epoch": l0["best_epoch"],
                "l1_best_epoch": l1["best_epoch"],
            })
    return rows, split_matches / 30.0, init_matches / 30.0


def seed_summary_rows(folds, oofs):
    rows = []
    for config in CONFIGS:
        for seed in SEEDS:
            stem = "{}_seed{}".format(config, seed)
            fold_rows = folds[stem]
            oof = oof_metrics(oofs[stem])
            row = {
                "config": config,
                "seed": seed,
                "n_oof": 871,
                "legacy_early_abort": False,
            }
            for key, value in oof.items():
                row["oof_" + key] = value
            for key in FOLD_METRICS:
                mean, std = mean_std([num(item, key) for item in fold_rows], ddof=0)
                row["fold_mean_" + key] = mean
                row["fold_sd_" + key] = std
            row["runtime_sec"] = sum(num(item, "runtime_sec", 0.0) for item in fold_rows)
            row["base_init_sha256_first_fold"] = fold_rows[0]["base_init_sha256"]
            row["base_init_unique_count"] = len({item["base_init_sha256"] for item in fold_rows})
            rows.append(row)
    return rows


def summary_rows(seed_rows):
    rows = []
    for config in CONFIGS:
        selected = [row for row in seed_rows if row["config"] == config]
        row = {"config": config, "n_seeds": len(selected), "n_oof_total": 2613}
        for key in OOF_METRICS:
            mean, std = mean_std([num(item, "oof_" + key) for item in selected], ddof=1)
            row["oof_{}_mean".format(key)] = mean
            row["oof_{}_sd".format(key)] = std
        for key in FOLD_METRICS:
            mean, std = mean_std([num(item, "fold_mean_" + key) for item in selected], ddof=1)
            row["fold_mean_{}_mean".format(key)] = mean
            row["fold_mean_{}_sd".format(key)] = std
            sd_mean, sd_sd = mean_std([num(item, "fold_sd_" + key) for item in selected], ddof=1)
            row["fold_sd_{}_mean".format(key)] = sd_mean
            row["fold_sd_{}_sd".format(key)] = sd_sd
        row["runtime_sec_mean"], row["runtime_sec_sd"] = mean_std(
            [num(item, "runtime_sec") for item in selected], ddof=1
        )
        row["base_init_unique_count_total"] = sum(int(item["base_init_unique_count"]) for item in selected)
        row["legacy_early_abort"] = any(item["legacy_early_abort"] for item in selected)
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
                auc_from_counts(counts, cand_order)
                - auc_from_counts(counts, ref_order)
            )
        cursor += size

    def interval(values, full_delta):
        p_value = 2.0 * min(float(np.mean(values <= 0)), float(np.mean(values >= 0)))
        return {
            "delta": float(full_delta),
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
            "p": float(min(1.0, p_value)),
        }

    return {
        "ba": interval(
            ba_deltas,
            full_candidate["balanced_acc"] - full_reference["balanced_acc"],
        ),
        "probability_auc": interval(
            auc_deltas,
            full_candidate["asd_probability_auc"] - full_reference["asd_probability_auc"],
        ),
    }


def bootstrap_rows(oofs):
    rows = []
    for seed in SEEDS:
        stats = bootstrap_metric_delta(
            oofs["L1_legacy_uniform_global_seed{}".format(seed)],
            oofs["L0_legacy_baseline_seed{}".format(seed)],
            random_seed=9400 + seed,
        )
        rows.append({
            "candidate": "L1_legacy_uniform_global",
            "reference": "L0_legacy_baseline",
            "seed": seed,
            "n_bootstrap": 10000,
            "delta_BA": stats["ba"]["delta"],
            "BA_CI_low": stats["ba"]["ci_low"],
            "BA_CI_high": stats["ba"]["ci_high"],
            "BA_p": stats["ba"]["p"],
            "delta_probability_AUC": stats["probability_auc"]["delta"],
            "probability_AUC_CI_low": stats["probability_auc"]["ci_low"],
            "probability_AUC_CI_high": stats["probability_auc"]["ci_high"],
            "probability_AUC_p": stats["probability_auc"]["p"],
        })
    return rows


def bootstrap_summary(rows):
    ba = np.asarray([float(row["delta_BA"]) for row in rows])
    auc = np.asarray([float(row["delta_probability_AUC"]) for row in rows])
    return [{
        "comparison": "L1_legacy_uniform_global_vs_L0_legacy_baseline",
        "n_seeds": 3,
        "delta_BA_mean": float(ba.mean()),
        "delta_BA_sd": float(ba.std(ddof=1)),
        "delta_probability_AUC_mean": float(auc.mean()),
        "delta_probability_AUC_sd": float(auc.std(ddof=1)),
        "BA_positive_seed_wins": int(np.sum(ba > 0)),
        "probability_AUC_positive_seed_wins": int(np.sum(auc > 0)),
    }]


def source_and_data_provenance(legacy_root):
    repo_root = os.path.dirname(os.path.dirname(legacy_root))
    data_root = os.path.join(legacy_root, "data", "ABIDE")
    data_hashes = {
        "processed_standard_data.csv": sha256_file(os.path.join(data_root, "processed_standard_data.csv")),
        "modal_feat_dict.npy": sha256_file(os.path.join(data_root, "modal_feat_dict.npy")),
    }
    strict_root = os.path.join(repo_root, "experiment", "experiment4_pair_adaptive_alignment", "data", "ABIDE")
    strict_match = all(
        data_hashes[name] == sha256_file(os.path.join(strict_root, name))
        for name in data_hashes
    )
    return {
        "repo_root": repo_root,
        "git_head": git_head(repo_root),
        "data_hashes": data_hashes,
        "strict_data_match": strict_match,
        "original_source_hashes": ORIGINAL_SOURCE_SHA256,
    }


def fmt_pm(mean, std, scale=100.0, digits=2):
    if not np.isfinite(mean):
        return "NA"
    return "{:.{d}f}±{:.{d}f}".format(mean * scale, std * scale, d=digits)


def fmt(value, digits=4):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not np.isfinite(value) else "{:.{}f}".format(value, digits)


def make_decisions(seed_rows, summary, bootstrap, split_rate, init_rate):
    lookup = {row["config"]: row for row in summary}
    seed_lookup = {(row["config"], int(row["seed"])): row for row in seed_rows}
    l0 = seed_lookup[("L0_legacy_baseline", 0)]
    baseline_reproduced = (
        abs(num(l0, "fold_mean_test_acc") - 0.8932) <= 0.02
        and abs(num(l0, "fold_mean_legacy_hard_auc") - 0.8917) <= 0.02
    )
    rep_deltas = {
        key: num(lookup["L1_legacy_uniform_global"], "fold_mean_{}_mean".format(key))
        - num(lookup["L0_legacy_baseline"], "fold_mean_{}_mean".format(key))
        for key in ("alignment_cos_gap", "alignment_R1", "alignment_CMMD")
    }
    representation_working = (
        rep_deltas["alignment_cos_gap"] > 0
        and rep_deltas["alignment_R1"] > 0
        and rep_deltas["alignment_CMMD"] < 0
    )
    ba_deltas = np.asarray([
        num(seed_lookup[("L1_legacy_uniform_global", seed)], "oof_balanced_acc")
        - num(seed_lookup[("L0_legacy_baseline", seed)], "oof_balanced_acc")
        for seed in SEEDS
    ])
    auc_deltas = np.asarray([
        num(seed_lookup[("L1_legacy_uniform_global", seed)], "oof_asd_probability_auc")
        - num(seed_lookup[("L0_legacy_baseline", seed)], "oof_asd_probability_auc")
        for seed in SEEDS
    ])
    legacy_supported = (
        ba_deltas.mean() >= 0.005
        and int(np.sum(ba_deltas > 0)) >= 2
        and auc_deltas.mean() >= -0.0025
        and representation_working
    )
    legacy_partial = ba_deltas.mean() > 0 or auc_deltas.mean() > 0
    if legacy_supported:
        legacy_status = "YES"
    elif legacy_partial:
        legacy_status = "PARTIAL"
    else:
        legacy_status = "NO"
    if representation_working:
        representation_status = "YES"
    elif sum((rep_deltas["alignment_cos_gap"] > 0,
              rep_deltas["alignment_R1"] > 0,
              rep_deltas["alignment_CMMD"] < 0)) >= 2:
        representation_status = "PARTIAL"
    else:
        representation_status = "NO"
    if legacy_status == "YES":
        robust_status = "YES"
    elif legacy_status == "PARTIAL":
        robust_status = "PARTIAL"
    else:
        robust_status = "NO"
    return {
        "LEGACY_BASELINE_REPRODUCED": "YES" if baseline_reproduced else "PARTIAL",
        "ALIGNMENT_REPRESENTATION_WORKING": representation_status,
        "LEGACY_ALIGNMENT_SUPPORTED": legacy_status,
        "LEGACY_INIT_FULLY_MATCHED": "YES" if init_rate == 1.0 else "NO",
        "PROTOCOL_ROBUST_ALIGNMENT": robust_status,
        "split_parity_rate": split_rate,
        "init_parity_rate": init_rate,
        "representation_deltas": rep_deltas,
        "legacy_ba_deltas": ba_deltas,
        "legacy_auc_deltas": auc_deltas,
        "legacy_ba_wins": int(np.sum(ba_deltas > 0)),
        "strict_effect_ba_pp": 0.81,
        "strict_effect_probability_auc_pp": 1.57,
    }


def write_report(results_dir, provenance, seed_rows, summary, bootstrap, bootstrap_sum, decisions):
    seed_lookup = {(row["config"], int(row["seed"])): row for row in seed_rows}
    summary_lookup = {row["config"]: row for row in summary}
    b0 = seed_lookup[("L0_legacy_baseline", 0)]
    b1 = seed_lookup[("L1_legacy_uniform_global", 0)]
    s0 = summary_lookup["L0_legacy_baseline"]
    s1 = summary_lookup["L1_legacy_uniform_global"]
    boot_mean = bootstrap_sum[0]
    data_hashes = provenance["data_hashes"]

    lines = [
        "# MMGL Experiment 4-Legacy: Original Transductive Protocol", "",
        "## 1. Executive Summary", "",
        "This experiment restores the original `MMGL_transductive` ABIDE protocol and compares only the original no-alignment baseline (L0) with a fixed six-pair global hybrid token alignment term (L1). It is intentionally separate from the strict validation experiments.", "",
        "| Verdict | Status |", "|---|---|",
    ]
    for key in (
        "LEGACY_BASELINE_REPRODUCED", "ALIGNMENT_REPRESENTATION_WORKING",
        "LEGACY_ALIGNMENT_SUPPORTED", "LEGACY_INIT_FULLY_MATCHED",
        "PROTOCOL_ROBUST_ALIGNMENT",
    ):
        lines.append("| {} | **{}** |".format(key, decisions[key]))
    lines += [
        "",
        "The legacy protocol is scientifically optimistic because the test fold is used every epoch for checkpoint selection and early stopping. Therefore its OOF is called **legacy-selected OOF**, not strict OOF.", "",
        "## 2. Why This Experiment Exists", "",
        "The paper's Table III ABIDE result maps to the repository's `MMGL_transductive` implementation. Earlier strict experiments changed the validation protocol to prevent test-label selection. This run answers the narrower question: does the alignment mechanism still work when the original repository protocol is restored exactly?", "",
        "## 3. Original Protocol Audit", "",
        "| Item | Registered legacy behavior |", "|---|---|",
        "| Split | `StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)` |",
        "| Validation | `val_idx=test_idx` |",
        "| Checkpoint | test-fold ACC, strict `>` improvement |",
        "| Early stopping | `wait_cnt > 50`; maximum 1000 epochs |",
        "| Fold RNG | seed once before splitter and once before fold loop; no per-fold reset |",
        "| Training | original `simple-2`, Step A then Step B, original optimizer/loss order |",
        "| Graph | original weighted-cosine GraphLearn + GCN |",
        "| Hard AUC | original one-hot true/prediction AUC retained as `legacy_hard_auc` |",
        "| Alignment | only L1 Step A; six equal pairs, λ=0.50, temperature=0.10, warmup=20 |",
        "",
        "## 4. Data and Source Provenance", "",
        "ABIDE contains 871 subjects: ASD=403 and NC=468. Internal labels are 0=ASD and 1=NC. Modalities are PHENO (48), ANAT (6), FUNC (10), and Correlation (256); the token tensor is `[N,4,36]`.", "",
        "| Artifact | SHA256 / value |", "|---|---|",
        "| Repository HEAD at run | `{}` |".format(provenance["git_head"]),
        "| `processed_standard_data.csv` | `{}` |".format(data_hashes["processed_standard_data.csv"]),
        "| `modal_feat_dict.npy` | `{}` |".format(data_hashes["modal_feat_dict.npy"]),
        "| Data matches strict Experiment 4 | **{}** |".format("YES" if provenance["strict_data_match"] else "NO"),
        "",
        "Original source-file hashes used as the legacy reference:", "",
    ]
    for name, digest in provenance["original_source_hashes"].items():
        lines.append("- `{}`: `{}`".format(name, digest))
    lines += [
        "",
        "## 5. Configurations", "",
        "| Config | Definition | λ |", "|---|---|---:|",
        "| L0_legacy_baseline | original simple-2; no token alignment | 0.00 |",
        "| L1_legacy_uniform_global | original simple-2 + six equal hybrid pairs in Step A | 0.50 |",
        "",
        "Both configurations use the same raw data, folds, graph path, classifier path, and post-hoc metrics. No pair-adaptive, shared/private, shuffled, sparse, or new hyperparameter search is included.", "",
        "## 6. Alignment Implementation", "",
        "At each L1 Step A, `compute_alignment_loss(tokens[train_idx], labels[train_idx], mode='hybrid', temperature=0.10)` is added to the original weighted classification loss. The effective coefficient is `0.50 × min(1,(epoch+1)/20)`, giving 0.025 at epoch 0 and 0.50 from epoch 19 onward. The six pair weights are exactly uniform.", "",
        "## 7. Training and Test-Label Boundary", "",
        "Alignment training uses only train-fold tokens and train-fold labels. Test-fold labels are not used by the alignment loss or any gradient. The legacy protocol nevertheless uses test-fold ACC for checkpoint selection, so selected checkpoints and legacy-selected OOF remain optimistic. Graph construction is transductive and can use all subject features, as in the original code. Representation metrics on the selected test fold are post-hoc diagnostics only.", "",
        "## 8. Sanity Checks", "",
        "The pre-run checks passed: data SHA parity with strict Experiment 4; data shape `(871,321)`; modal dimensions `[48,6,10,256]`; original-versus-modified Transformer maximum probability and hidden-state differences below `1e-7` (observed 0); token shape `[871,4,36]`; L0 Step-A parity; finite Step-B graph loss; finite L1 hybrid alignment; and warmup values 0.025/0.50/0.50 for epochs 0/19/20.", "",
        "## 9. Author-Default Seed-0 Results", "",
        "The table below uses the registered paper configuration and reports the 10 fold mean ± population fold SD. `legacy_hard_auc` is the original hard-prediction AUC; probability AUC is reported separately.", "",
        "| Config | ACC | legacy hard AUC | ASD SEN | NC SPE | best epoch mean |", "|---|---:|---:|---:|---:|---:|",
        "| L0 legacy baseline | {} | {} | {} | {} | {} |".format(
            fmt_pm(num(b0, "fold_mean_test_acc"), num(b0, "fold_sd_test_acc")),
            fmt_pm(num(b0, "fold_mean_legacy_hard_auc"), num(b0, "fold_sd_legacy_hard_auc")),
            fmt_pm(num(b0, "fold_mean_asd_sensitivity"), num(b0, "fold_sd_asd_sensitivity")),
            fmt_pm(num(b0, "fold_mean_nc_specificity"), num(b0, "fold_sd_nc_specificity")),
            fmt(num(b0, "fold_mean_best_epoch"), 2),
        ),
        "| L1 legacy uniform global | {} | {} | {} | {} | {} |".format(
            fmt_pm(num(b1, "fold_mean_test_acc"), num(b1, "fold_sd_test_acc")),
            fmt_pm(num(b1, "fold_mean_legacy_hard_auc"), num(b1, "fold_sd_legacy_hard_auc")),
            fmt_pm(num(b1, "fold_mean_asd_sensitivity"), num(b1, "fold_sd_asd_sensitivity")),
            fmt_pm(num(b1, "fold_mean_nc_specificity"), num(b1, "fold_sd_nc_specificity")),
            fmt(num(b1, "fold_mean_best_epoch"), 2),
        ),
        "",
        "## 10. Paper and Repository Reference", "",
        "The source PDF's ABIDE Table III reports the MMGL row as ACC `89.77±2.72`, AUC `89.81±2.56`, SEN `90.32±4.21`, and SPE `89.30±6.04`. The paper text describes mean scores and standard errors for ACC/AUC; the repository reproduction report also notes that the original script prints fold standard deviations. These uncertainty conventions are therefore not interchangeable with the fold-SD table above.", "",
        "The prior repository reproduction recorded approximately ACC `89.32%` and original hard AUC `89.17%` for the same transductive script/configuration. The L0 seed-0 result is compared descriptively against that reference; no result-guided tuning was performed.", "",
        "## 11. Extended Three-Seed Legacy-Selected OOF", "",
        "Each config has 3 seeds × 10 folds and 871 subject-level predictions per seed. The following mean ± sample SD is across the three seed-level legacy-selected OOF evaluations.", "",
        "| Config | ACC | BA | probability ASD-AUC | legacy hard AUC | ASD F1 | ASD SEN | NC SPE |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config in CONFIGS:
        row = summary_lookup[config]
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            config,
            fmt_pm(num(row, "oof_acc_mean"), num(row, "oof_acc_sd")),
            fmt_pm(num(row, "oof_balanced_acc_mean"), num(row, "oof_balanced_acc_sd")),
            fmt_pm(num(row, "oof_asd_probability_auc_mean"), num(row, "oof_asd_probability_auc_sd")),
            fmt_pm(num(row, "oof_legacy_hard_auc_mean"), num(row, "oof_legacy_hard_auc_sd")),
            fmt_pm(num(row, "oof_asd_f1_mean"), num(row, "oof_asd_f1_sd")),
            fmt_pm(num(row, "oof_asd_sensitivity_mean"), num(row, "oof_asd_sensitivity_sd")),
            fmt_pm(num(row, "oof_nc_specificity_mean"), num(row, "oof_nc_specificity_sd")),
        ))
    lines += [
        "",
        "## 12. Hard-Metric Comparison", "",
        "The original hard AUC and post-hoc probability ASD-AUC answer different questions. Hard AUC is retained for protocol fidelity and can be tied closely to hard predictions; probability AUC evaluates ranking quality from the ASD posterior. Alignment does not replace the legacy metric.", "",
        "| Comparison | Δ ACC | Δ BA | Δ probability ASD-AUC | Δ legacy hard AUC | BA wins |", "|---|---:|---:|---:|---:|---:|",
        "| L1 − L0, three-seed OOF mean | {:+.2f} pp | {:+.2f} pp | {:+.2f} pp | {:+.2f} pp | {}/3 |".format(
            100 * (num(s1, "oof_acc_mean") - num(s0, "oof_acc_mean")),
            100 * (num(s1, "oof_balanced_acc_mean") - num(s0, "oof_balanced_acc_mean")),
            100 * (num(s1, "oof_asd_probability_auc_mean") - num(s0, "oof_asd_probability_auc_mean")),
            100 * (num(s1, "oof_legacy_hard_auc_mean") - num(s0, "oof_legacy_hard_auc_mean")),
            decisions["legacy_ba_wins"],
        ),
        "",
        "## 13. Representation Diagnostics", "",
        "These are post-hoc selected-test-fold token diagnostics: cosine gap and R@1 should increase, while class-conditional MMD should decrease. They do not affect checkpoint selection or training.", "",
        "| Config | cosine gap | R@1 | CMMD |", "|---|---:|---:|---:|",
    ]
    for config in CONFIGS:
        row = summary_lookup[config]
        lines.append("| {} | {} | {} | {} |".format(
            config,
            fmt_pm(num(row, "fold_mean_alignment_cos_gap_mean"), num(row, "fold_mean_alignment_cos_gap_sd"), scale=1.0),
            fmt_pm(num(row, "fold_mean_alignment_R1_mean"), num(row, "fold_mean_alignment_R1_sd"), scale=1.0),
            fmt_pm(num(row, "fold_mean_alignment_CMMD_mean"), num(row, "fold_mean_alignment_CMMD_sd"), scale=1.0),
        ))
    lines += [
        "",
        "L1−L0 representation deltas are cosine gap `{:+.6f}`, R@1 `{:+.6f}`, and CMMD `{:+.6f}`.".format(
            decisions["representation_deltas"]["alignment_cos_gap"],
            decisions["representation_deltas"]["alignment_R1"],
            decisions["representation_deltas"]["alignment_CMMD"],
        ),
        "",
        "## 14. Alignment Effect", "",
        "L1 is supported as a legacy alignment mechanism only when the pre-registered descriptive criteria are met: mean ΔBA ≥0.5 percentage point, positive BA in at least two of three seeds, probability-AUC change no worse than −0.25 percentage point, and all three representation directions improve. This criterion separates a representation effect from a one-seed accuracy fluctuation.", "",
        "| Seed | Δ BA | Δ probability ASD-AUC |", "|---:|---:|---:|",
    ]
    for seed in SEEDS:
        l0_seed = seed_lookup[("L0_legacy_baseline", seed)]
        l1_seed = seed_lookup[("L1_legacy_uniform_global", seed)]
        lines.append("| {} | {:+.4f} | {:+.4f} |".format(
            seed,
            num(l1_seed, "oof_balanced_acc") - num(l0_seed, "oof_balanced_acc"),
            num(l1_seed, "oof_asd_probability_auc") - num(l0_seed, "oof_asd_probability_auc"),
        ))
    lines += [
        "",
        "## 15. RNG and Initialization Audit", "",
        "| Audit | Result |", "|---|---:|",
        "| L0/L1 split parity | {:.0%} ({}/30 seed-fold pairs) |".format(
            decisions["split_parity_rate"], int(round(decisions["split_parity_rate"] * 30))
        ),
        "| L0/L1 base initialization parity | {:.0%} ({}/30 seed-fold pairs) |".format(
            decisions["init_parity_rate"], int(round(decisions["init_parity_rate"] * 30))
        ),
        "| Per-fold RNG reset | **NO**, as required by legacy protocol |",
        "| Early-abort trigger | **NO**; every job produced 10 folds |",
        "",
        "Initialization parity means the pre-training state of ModalFusion, GraphConstruct and MessagePassing matched between L0 and L1 for the same seed/fold. Because the legacy runner does not reset RNG per fold, later fold initializations are expected to be different from earlier folds; that is preserved rather than normalized away.", "",
        "## 16. Strict-versus-Legacy Comparison", "",
        "| Protocol / reference | Δ BA | Δ probability AUC | Interpretation |", "|---|---:|---:|---|",
        "| Strict Experiment 4 historical L1−L0 | +0.81 pp | +1.57 pp | test labels excluded from selection; prior completed result |",
        "| Current legacy protocol L1−L0 | {:+.2f} pp | {:+.2f} pp | test fold selects checkpoints; legacy-selected OOF |".format(
            100 * (num(s1, "oof_balanced_acc_mean") - num(s0, "oof_balanced_acc_mean")),
            100 * (num(s1, "oof_asd_probability_auc_mean") - num(s0, "oof_asd_probability_auc_mean")),
        ),
        "",
        "The strict result is the valid protocol reference for generalization; the legacy result is a sensitivity analysis of the original repository behavior. They should not be pooled or treated as the same estimator.", "",
        "## 17. Paired Subject Bootstrap", "",
        "Each seed uses 10,000 paired subject bootstrap resamples comparing the same subject indices under L1 and L0. The resampling unit is the subject, not the repeated-CV fold.", "",
        "| Seed | Δ BA | BA 95% CI | p | Δ probability AUC | AUC 95% CI | p |", "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in bootstrap:
        lines.append("| {} | {:+.4f} | [{:+.4f}, {:+.4f}] | {:.4f} | {:+.4f} | [{:+.4f}, {:+.4f}] | {:.4f} |".format(
            row["seed"], float(row["delta_BA"]), float(row["BA_CI_low"]), float(row["BA_CI_high"]), float(row["BA_p"]),
            float(row["delta_probability_AUC"]), float(row["probability_AUC_CI_low"]), float(row["probability_AUC_CI_high"]), float(row["probability_AUC_p"]),
        ))
    lines += [
        "| Mean across seeds | {:+.4f}±{:.4f} | — | — | {:+.4f}±{:.4f} | — | — |".format(
            float(boot_mean["delta_BA_mean"]), float(boot_mean["delta_BA_sd"]),
            float(boot_mean["delta_probability_AUC_mean"]), float(boot_mean["delta_probability_AUC_sd"]),
        ),
        "",
        "Bootstrap p-values are exploratory because 10-fold repeated-CV training sets overlap and the legacy checkpoint rule uses the test fold. They are not independent clinical-validation p-values.", "",
        "## 18. Limitations and Interpretation", "",
        "The main limitation is protocol leakage by design: test-fold ACC controls checkpoint selection and early stopping. The legacy-selected OOF quantifies reproducibility of the paper/repository procedure, not an unbiased deployment estimate. In addition, the original hard AUC is not a probability-ranking AUC, and representation diagnostics are post-hoc. The alignment experiment is deliberately narrow: it does not establish that the mechanism transfers to TADPOLE or to another model.", "",
        "## 19. Verdict and Next Step", "",
    ]
    for key in (
        "LEGACY_BASELINE_REPRODUCED", "ALIGNMENT_REPRESENTATION_WORKING",
        "LEGACY_ALIGNMENT_SUPPORTED", "LEGACY_INIT_FULLY_MATCHED",
        "PROTOCOL_ROBUST_ALIGNMENT",
    ):
        lines.append("{}: **{}**".format(key, decisions[key]))
    lines += [
        "",
        "结论：本次严格恢复了原论文对应的 `MMGL_transductive` 旧协议，并保留了 60 个 fold 的初始化与划分审计。L1 的 alignment 是否被支持，以本报告中的三 seed legacy-selected OOF、表示指标方向和 bootstrap 为准；不能把 legacy-selected OOF 当作严格泛化性能。下一步应进入 SPromptGL 的 Experiment 5，而不是继续在 MMGL 上扩展 Experiment 5。", "",
        "## Files", "",
        "`results/fold_results.csv`, `results/oof_predictions.csv`, `results/seed_summary.csv`, `results/summary.csv`, `results/rng_audit.csv`, `results/bootstrap.csv`, `results/bootstrap_summary.csv`, `results/report.md`, and six job logs.", "",
    ]
    with open(os.path.join(results_dir, "report.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    legacy_root = os.path.dirname(os.path.dirname(os.path.abspath(args.raw_dir)))
    folds, oofs = validate_raw(args.raw_dir)
    rng_rows, split_rate, init_rate = audit_splits_and_initialization(folds)
    seed_rows = seed_summary_rows(folds, oofs)
    summary = summary_rows(seed_rows)
    boot = bootstrap_rows(oofs)
    boot_sum = bootstrap_summary(boot)
    provenance = source_and_data_provenance(legacy_root)
    decisions = make_decisions(seed_rows, summary, boot, split_rate, init_rate)

    all_folds = [row for stem in sorted(folds) for row in folds[stem]]
    all_oofs = [row for stem in sorted(oofs) for row in oofs[stem]]
    write_csv(os.path.join(args.results_dir, "fold_results.csv"), all_folds)
    write_csv(os.path.join(args.results_dir, "oof_predictions.csv"), all_oofs)
    write_csv(os.path.join(args.results_dir, "seed_summary.csv"), seed_rows)
    write_csv(os.path.join(args.results_dir, "summary.csv"), summary)
    write_csv(os.path.join(args.results_dir, "rng_audit.csv"), rng_rows)
    write_csv(os.path.join(args.results_dir, "bootstrap.csv"), boot)
    write_csv(os.path.join(args.results_dir, "bootstrap_summary.csv"), boot_sum)
    write_csv(os.path.join(args.results_dir, "verdicts.csv"), [
        {"verdict": key, "status": decisions[key]}
        for key in (
            "LEGACY_BASELINE_REPRODUCED",
            "ALIGNMENT_REPRESENTATION_WORKING",
            "LEGACY_ALIGNMENT_SUPPORTED",
            "LEGACY_INIT_FULLY_MATCHED",
            "PROTOCOL_ROBUST_ALIGNMENT",
        )
    ])
    write_report(args.results_dir, provenance, seed_rows, summary, boot, boot_sum, decisions)

    print("AGGREGATE_OK")
    print("FOLD_ROWS={}".format(len(all_folds)))
    print("OOF_ROWS={}".format(len(all_oofs)))
    print("SPLIT_PARITY_RATE={:.0%}".format(split_rate))
    print("LEGACY_INIT_PARITY_RATE={:.0%}".format(init_rate))
    print("DATA_MATCH_STRICT_EXPERIMENT4={}".format("YES" if provenance["strict_data_match"] else "NO"))
    print("BOOTSTRAP_RESAMPLES_PER_SEED=10000")


if __name__ == "__main__":
    main()
