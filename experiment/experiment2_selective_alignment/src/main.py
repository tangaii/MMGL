import argparse
import csv
import json
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from model import EvalHelper, evaluate_prob


MODALITY_NAMES = ["PHENO", "ANAT", "FUNC", "Correlation"]


def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_hyperparameters(args):
    return SimpleNamespace(
        datname="ABIDE",
        nclass=2,
        nmodal=4,
        nepoch=args.nepoch,
        early=args.early,
        lr=0.0038,
        reg=0.11,
        dropout=0.35,
        nlayer=1,
        n_hidden=18,
        n_head=2,
        n_iter=10,
        th=0.9,
        GC_mode="weighted-cosine",
        MP_mode="GCN",
        MF_mode="concat",
        alpha=0.5,
        theta_smooth=1.0,
        theta_degree=0.5,
        theta_sparsity=0.0,
        mode="simple-2",
        seed=args.seed,
        align_strategy=args.align_strategy,
        align_mode=args.align_mode,
        align_lambda=args.align_lambda,
        align_temperature=args.align_temperature,
        align_warmup=args.align_warmup,
        lambda_orth=args.lambda_orth,
        lambda_balance=args.lambda_balance,
        target_shared_ratio=args.target_shared_ratio,
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
    train_index = outer_train[relative_train]
    val_index = outer_train[relative_val]
    return train_index, val_index, outer_test, outer_train


def _row_with_common_fields(args, fold_id, split_sizes, agent, best_epoch, elapsed, result):
    outer_train_size, train_size, val_size, test_size = split_sizes
    test_metrics = result["test_metrics"]
    raw = result["raw_alignment"]
    shared = result["shared_alignment"]
    row = {
        "config": args.config,
        "seed": args.seed,
        "fold": fold_id + 1,
        "n_outer_train": outer_train_size,
        "n_train": train_size,
        "n_val": val_size,
        "n_test": test_size,
        "best_epoch": best_epoch,
        "acc": test_metrics["acc"],
        "balanced_acc": test_metrics["balanced_acc"],
        "asd_auc": test_metrics["asd_auc"],
        "asd_f1": test_metrics["asd_f1"],
        "asd_sensitivity": test_metrics["asd_sensitivity"],
        "nc_specificity": test_metrics["nc_specificity"],
        "raw_pos_cos": raw["pos_cos"],
        "raw_neg_cos": raw["neg_cos"],
        "raw_cos_gap": raw["cos_gap"],
        "raw_retrieval_r1": raw["retrieval_r1"],
        "raw_cmmd": raw["cmmd"],
        "shared_pos_cos": shared["pos_cos"],
        "shared_neg_cos": shared["neg_cos"],
        "shared_cos_gap": shared["cos_gap"],
        "shared_retrieval_r1": shared["retrieval_r1"],
        "shared_cmmd": shared["cmmd"],
        "final_cls_loss": result["final_cls_loss"],
        "final_align_loss": result["final_align_loss"],
        "final_orth_loss": result["final_orth_loss"],
        "final_balance_loss": result["final_balance_loss"],
        "align_strategy": args.align_strategy,
        "align_mode": args.align_mode,
        "align_lambda": args.align_lambda,
        "align_temperature": args.align_temperature,
        "align_warmup": args.align_warmup,
        "lambda_orth": args.lambda_orth,
        "lambda_balance": args.lambda_balance,
        "target_shared_ratio": args.target_shared_ratio,
        "token_nmodal": agent.token_shape[1],
        "token_dim": agent.token_shape[2],
        "runtime_sec": elapsed,
    }
    for key, value in result["gate_stats"].items():
        row[key] = value
    return row


def _oof_rows(args, fold_id, test_index, subject_ids, original_labels, result):
    probs = result["probs"][test_index]
    pred_internal = probs.argmax(axis=1)
    rows = []
    for local_index, subject_index in enumerate(test_index):
        true_internal = int(original_labels[subject_index] - 1)
        true_asd = int(true_internal == 0)
        pred_asd = int(pred_internal[local_index] == 0)
        rows.append(
            {
                "config": args.config,
                "seed": args.seed,
                "fold": fold_id + 1,
                "subject_index": int(subject_index),
                "subject_id": subject_ids[subject_index],
                "true_original_label": int(original_labels[subject_index]),
                "true_asd": true_asd,
                "p_asd": float(probs[local_index, 0]),
                "pred_asd": pred_asd,
            }
        )
    return rows


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_fold(args, input_data, labels, original_labels, subject_ids, input_dims, fold_id):
    train_index, val_index, test_index, outer_train = make_split(
        input_data, labels, args.seed, fold_id
    )

    # Reset before each fold's complete module initialization.
    fold_seed = args.seed * 100000 + fold_id * 100 + 17
    set_rng_seed(fold_seed)
    hyperpm = make_hyperparameters(args)
    agent = EvalHelper(
        input_dims,
        input_data,
        labels,
        hyperpm,
        train_index,
        val_index,
        test_index,
        MODALITY_NAMES,
    )

    # SelectiveAligner has now been initialized. Start training from the same
    # RNG point across configurations for this seed/fold.
    train_seed = args.seed * 100000 + fold_id * 100 + 73
    set_rng_seed(train_seed)

    start = time.perf_counter()
    best_val_acc = -np.inf
    best_epoch = 0
    wait_count = 0
    best_states = None
    for epoch in range(args.nepoch):
        losses = agent.run_epoch(mode="simple-2", epoch=epoch)
        val_prob, _, _ = agent.forward_all(return_modal_tokens=False)
        val_metrics = evaluate_prob(val_prob, agent.targ, agent.val_idx)
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch + 1
            wait_count = 0
            best_states = agent.state_dicts()
            print(
                "fold={:02d} epoch={:04d} val_acc={:.4f} val_ba={:.4f} "
                "cls={:.5f} align={:.5f} orth={:.5f} balance={:.5f}".format(
                    fold_id + 1,
                    epoch + 1,
                    val_metrics["acc"],
                    val_metrics["balanced_acc"],
                    losses["cls"],
                    losses["align"],
                    losses["orth"],
                    losses["balance"],
                )
            )
        else:
            wait_count += 1
            if wait_count > args.early:
                break
        if (epoch + 1) % 25 == 0:
            print(
                "fold={:02d} epoch={:04d} current_val_acc={:.4f} wait={}".format(
                    fold_id + 1, epoch + 1, val_metrics["acc"], wait_count
                )
            )

    if best_states is None:
        raise RuntimeError("No checkpoint produced for fold {}".format(fold_id + 1))
    agent.load_state_dicts(best_states)
    result = agent.best_metrics()
    elapsed = time.perf_counter() - start
    row = _row_with_common_fields(
        args,
        fold_id,
        (len(outer_train), len(train_index), len(val_index), len(test_index)),
        agent,
        best_epoch,
        elapsed,
        result,
    )
    oof_rows = _oof_rows(
        args, fold_id, test_index, subject_ids, original_labels, result
    )
    print(
        "fold={:02d} done epoch={} ACC={:.4f} BA={:.4f} ASD-AUC={:.4f} "
        "runtime={:.1f}s".format(
            fold_id + 1,
            best_epoch,
            row["acc"],
            row["balanced_acc"],
            row["asd_auc"],
            elapsed,
        )
    )
    del agent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row, oof_rows


def main():
    parser = argparse.ArgumentParser(description="MMGL experiment 2 selective alignment")
    parser.add_argument("--datadir", default="data/")
    parser.add_argument("--config", required=True)
    parser.add_argument("--align_strategy", "--align-strategy", dest="align_strategy", choices=["none", "global", "selective"], required=True)
    parser.add_argument("--align_mode", "--align-mode", dest="align_mode", choices=["none", "pair_nce", "cmmd", "hybrid"], required=True)
    parser.add_argument("--align_lambda", "--align-lambda", dest="align_lambda", type=float, required=True)
    parser.add_argument("--align_temperature", "--align-temperature", dest="align_temperature", type=float, default=0.10)
    parser.add_argument("--align_warmup", "--align-warmup", dest="align_warmup", type=int, default=20)
    parser.add_argument("--lambda_orth", "--lambda-orth", dest="lambda_orth", type=float, default=0.02)
    parser.add_argument("--lambda_balance", "--lambda-balance", dest="lambda_balance", type=float, default=0.05)
    parser.add_argument("--target_shared_ratio", "--target-shared-ratio", dest="target_shared_ratio", type=float, default=0.50)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--nepoch", type=int, default=1000)
    parser.add_argument("--early", type=int, default=50)
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--fold_results", "--fold-results", dest="fold_results", required=True)
    parser.add_argument("--oof_results", "--oof-results", dest="oof_results", required=True)
    args = parser.parse_args()

    if args.align_strategy == "none" and args.align_mode != "none":
        raise ValueError("none strategy requires none alignment mode")
    if args.align_strategy == "selective" and args.align_mode != "hybrid":
        raise ValueError("selective strategy requires hybrid alignment mode")
    if args.align_strategy != "none" and args.align_lambda <= 0:
        raise ValueError("aligned strategies require positive align_lambda")
    if not (-1 <= args.fold <= 9):
        raise ValueError("fold must be -1 or 0..9")

    input_data, labels, original_labels, subject_ids, input_dims = load_abide(args.datadir)
    print(
        json.dumps(
            {
                "config": args.config,
                "seed": args.seed,
                "data_shape": list(input_data.shape),
                "class_counts_internal": np.bincount(labels).tolist(),
                "modal_names": MODALITY_NAMES,
                "modal_dims": input_dims,
                "align_strategy": args.align_strategy,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            },
            sort_keys=True,
        )
    )

    folds = [args.fold] if args.fold >= 0 else list(range(10))
    all_rows = []
    all_oof = []
    for fold_id in folds:
        row, oof_rows = run_fold(
            args,
            input_data,
            labels,
            original_labels,
            subject_ids,
            input_dims,
            fold_id,
        )
        all_rows.append(row)
        all_oof.extend(oof_rows)

    write_csv(args.fold_results, all_rows)
    write_csv(args.oof_results, all_oof)
    print("wrote {} fold rows to {}".format(len(all_rows), args.fold_results))
    print("wrote {} OOF rows to {}".format(len(all_oof), args.oof_results))


if __name__ == "__main__":
    main()
