import argparse
import csv
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from alignment import PAIR_NAMES
from metrics import evaluate_prob
from model import EvalHelper
from utils import set_rng_seed


MODALITY_NAMES = ["PHENO", "ANAT", "FUNC", "Correlation"]
CONFIGS = {
    "E0_baseline": ("none", "none", 0.0),
    "E1_uniform_global": ("uniform", "hybrid", 0.50),
    "E2_diagnosis_adaptive": ("adaptive", "hybrid", 0.50),
    "E3_shuffled_adaptive": ("shuffled", "hybrid", 0.50),
    "E4_adaptive_top3": ("top3", "hybrid", 0.50),
    "E5_balanced_sparse_control": ("sparse", "hybrid", 0.50),
}


def make_hyperparameters(args):
    return SimpleNamespace(
        datname="ABIDE", nclass=2, nmodal=4,
        nepoch=args.nepoch, early=args.early,
        lr=0.0038, reg=0.11, dropout=0.35,
        nlayer=1, n_hidden=18, n_head=2, n_iter=10,
        th=0.9, GC_mode="weighted-cosine", MP_mode="GCN",
        MF_mode="concat", alpha=0.5,
        theta_smooth=1.0, theta_degree=0.5, theta_sparsity=0.0,
        mode="simple-2", seed=args.seed,
        align_strategy=args.align_strategy, align_mode=args.align_mode,
        align_lambda=args.align_lambda, align_temperature=0.10,
        align_warmup=20, weight_update_interval=5,
    )


def load_abide(datadir):
    data_dir = os.path.join(datadir, "ABIDE")
    data = pd.read_csv(os.path.join(data_dir, "processed_standard_data.csv")).values
    modal_feat_dict = np.load(
        os.path.join(data_dir, "modal_feat_dict.npy"), allow_pickle=True
    ).item()
    with open(os.path.join(data_dir, "subject_IDs.txt")) as handle:
        subject_ids = [line.strip() for line in handle if line.strip()]
    input_data = data[:, :-1].astype(np.float32)
    original_labels = data[:, -1].astype(np.int64)
    labels = original_labels - 1
    input_dims = [len(modal_feat_dict[name]) for name in MODALITY_NAMES]
    if input_data.shape != (871, 320):
        raise ValueError("Unexpected ABIDE input shape: {}".format(input_data.shape))
    if len(subject_ids) != len(input_data):
        raise ValueError("subject_IDs length does not match CSV rows")
    if sum(input_dims) != input_data.shape[1]:
        raise ValueError("Modal dimensions do not match CSV columns")
    if sorted(np.unique(original_labels).tolist()) != [1, 2]:
        raise ValueError("Expected original ABIDE labels 1 and 2")
    if np.bincount(labels, minlength=2).tolist() != [403, 468]:
        raise ValueError("Unexpected internal ABIDE class counts")
    return input_data, labels, original_labels, subject_ids, input_dims


def make_split(input_data, labels, seed, fold_id):
    outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    outer_train, outer_test = list(outer.split(input_data, labels))[fold_id]
    inner = StratifiedShuffleSplit(
        n_splits=1, test_size=0.10, random_state=seed + 1000 + fold_id
    )
    relative_train, relative_val = next(
        inner.split(input_data[outer_train], labels[outer_train])
    )
    return (
        outer_train[relative_train], outer_train[relative_val],
        outer_test, outer_train,
    )


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def row_from_result(args, fold_id, split_sizes, agent, best_epoch, elapsed, result):
    outer_train_size, train_size, val_size, test_size = split_sizes
    metrics = result["test_metrics"]
    raw = result["raw_alignment"]
    row = {
        "config": args.config, "seed": args.seed, "fold": fold_id + 1,
        "n_outer_train": outer_train_size, "n_train": train_size,
        "n_val": val_size, "n_test": test_size, "best_epoch": best_epoch,
        "acc": metrics["acc"], "balanced_acc": metrics["balanced_acc"],
        "asd_auc": metrics["asd_auc"], "asd_f1": metrics["asd_f1"],
        "asd_sensitivity": metrics["asd_sensitivity"],
        "nc_specificity": metrics["nc_specificity"],
        "raw_global_cos_gap": raw["cos_gap"],
        "raw_global_R1": raw["retrieval_r1"],
        "raw_global_CMMD": raw["cmmd"],
        "pair_weight_std": result["pair_weight_std"],
        "pair_weight_entropy": result["pair_weight_entropy"],
        "top1_pair": PAIR_NAMES[result["top1_pair"]] if result["top1_pair"] >= 0 else "NA",
        "top3_pairs": ",".join(PAIR_NAMES[i] for i in result["top3_pairs"]) if result["top3_pairs"] else "NA",
        "final_cls_loss": result["final_cls_loss"],
        "final_align_loss": result["final_align_loss"],
        "align_strategy": args.align_strategy, "align_mode": args.align_mode,
        "align_lambda": args.align_lambda, "align_warmup": 20,
        "weight_update_interval": 5,
        "token_nmodal": agent.token_shape[1], "token_dim": agent.token_shape[2],
        "base_init_sha256": agent.base_init_sha256,
        "runtime_sec": elapsed,
    }
    for pair_id, score in enumerate(result["pair_scores"]):
        row["score_P{}".format(pair_id)] = score
    for pair_id, weight in enumerate(result["pair_weights"]):
        row["weight_P{}".format(pair_id)] = weight
    for pair in result["pair_metrics"]:
        prefix = "P{}".format(pair["pair_id"])
        row[prefix + "_cos_gap"] = pair["cos_gap"]
        row[prefix + "_R1"] = pair["retrieval_r1"]
        row[prefix + "_CMMD"] = pair["cmmd"]
    return row


def make_oof_rows(args, fold_id, test_index, subject_ids, original_labels, result):
    probs = result["probs"][test_index]
    pred_internal = probs.argmax(axis=1)
    rows = []
    for local_index, subject_index in enumerate(test_index):
        true_internal = int(original_labels[subject_index] - 1)
        rows.append({
            "config": args.config, "seed": args.seed, "fold": fold_id + 1,
            "subject_index": int(subject_index), "subject_id": subject_ids[subject_index],
            "true_original_label": int(original_labels[subject_index]),
            "true_asd": int(true_internal == 0),
            "p_asd": float(probs[local_index, 0]),
            "pred_asd": int(pred_internal[local_index] == 0),
        })
    return rows


def run_fold(args, input_data, labels, original_labels, subject_ids, input_dims, fold_id):
    train_index, val_index, test_index, outer_train = make_split(
        input_data, labels, args.seed, fold_id
    )
    fold_seed = args.seed * 100000 + fold_id * 100 + 17
    train_seed = args.seed * 100000 + fold_id * 100 + 73
    set_rng_seed(fold_seed)
    hyperpm = make_hyperparameters(args)
    agent = EvalHelper(
        input_dims, input_data, labels, hyperpm,
        train_index, val_index, test_index, MODALITY_NAMES,
    )
    set_rng_seed(train_seed)

    start = time.perf_counter()
    best_val_acc, best_epoch, wait_count, best_states = -np.inf, 0, 0, None
    for epoch in range(args.nepoch):
        agent.run_epoch(mode="simple-2", epoch=epoch)
        val_prob, _, _ = agent.forward_all(return_modal_tokens=False)
        val_metrics = evaluate_prob(val_prob, agent.targ, agent.val_idx)
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch + 1
            wait_count = 0
            best_states = agent.state_dicts()
        else:
            wait_count += 1
            if wait_count > args.early:
                break
    if best_states is None:
        raise RuntimeError("No checkpoint produced for fold {}".format(fold_id + 1))
    agent.load_state_dicts(best_states)
    result = agent.best_metrics()
    elapsed = time.perf_counter() - start
    row = row_from_result(
        args, fold_id,
        (len(outer_train), len(train_index), len(val_index), len(test_index)),
        agent, best_epoch, elapsed, result,
    )
    current_oof = make_oof_rows(
        args, fold_id, test_index, subject_ids, original_labels, result
    )
    print(
        "fold={:02d} epoch={} ACC={:.4f} BA={:.4f} ASD-AUC={:.4f} runtime={:.1f}s".format(
            fold_id + 1, best_epoch, row["acc"], row["balanced_acc"],
            row["asd_auc"], elapsed
        ), flush=True
    )
    del agent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row, current_oof


def main():
    parser = argparse.ArgumentParser(description="MMGL Experiment 4")
    parser.add_argument("--datadir", default="data/")
    parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--align_strategy", choices=["none", "uniform", "adaptive", "shuffled", "top3", "sparse"], required=True)
    parser.add_argument("--align_mode", choices=["none", "hybrid"], required=True)
    parser.add_argument("--align_lambda", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--nepoch", type=int, default=1000)
    parser.add_argument("--early", type=int, default=50)
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--fold_results", required=True)
    parser.add_argument("--oof_results", required=True)
    args = parser.parse_args()
    if (args.align_strategy, args.align_mode, args.align_lambda) != CONFIGS[args.config]:
        raise ValueError("config arguments do not match {}".format(args.config))
    if not (-1 <= args.fold <= 9):
        raise ValueError("fold must be -1 or 0..9")

    input_data, labels, original_labels, subject_ids, input_dims = load_abide(args.datadir)
    print(json.dumps({
        "config": args.config, "seed": args.seed,
        "data_shape": list(input_data.shape),
        "class_counts_internal": np.bincount(labels, minlength=2).tolist(),
        "modal_names": MODALITY_NAMES, "modal_dims": input_dims,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }, sort_keys=True), flush=True)
    folds = [args.fold] if args.fold >= 0 else list(range(10))
    fold_rows, all_oof = [], []
    for fold_id in folds:
        row, current_oof = run_fold(
            args, input_data, labels, original_labels, subject_ids, input_dims, fold_id
        )
        fold_rows.append(row)
        all_oof.extend(current_oof)
    write_csv(args.fold_results, fold_rows)
    write_csv(args.oof_results, all_oof)
    print("wrote fold_rows={} oof_rows={}".format(len(fold_rows), len(all_oof)), flush=True)


if __name__ == "__main__":
    main()
