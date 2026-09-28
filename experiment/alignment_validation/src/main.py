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

from alignment import compute_alignment_metrics
from model import EvalHelper, evaluate_metrics


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
    # These are the ABIDE settings used by the original MMGL transductive
    # runner, except for the explicitly requested strict split/alignment
    # controls.
    return SimpleNamespace(
        datname=args.datname,
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
        align_mode=args.align_mode,
        align_lambda=args.align_lambda,
        align_temperature=args.align_temperature,
        align_warmup=args.align_warmup,
    )


def load_abide(datadir, datname):
    data_dir = os.path.join(datadir, datname)
    modal_path = os.path.join(data_dir, "modal_feat_dict.npy")
    csv_path = os.path.join(data_dir, "processed_standard_data.csv")
    modal_feat_dict = np.load(modal_path, allow_pickle=True).item()
    data = pd.read_csv(csv_path).values
    input_data = data[:, :-1].astype(np.float32)
    label = data[:, -1].astype(np.int64) - 1
    modal_order = ["PHENO", "ANAT", "FUNC", "Correlation"]
    missing = [key for key in modal_order if key not in modal_feat_dict]
    if missing:
        raise ValueError("Missing ABIDE modalities: {}".format(missing))
    input_data_dims = [len(modal_feat_dict[key]) for key in modal_order]
    if sum(input_data_dims) != input_data.shape[1]:
        raise ValueError(
            "Modal dimensions {} sum to {}, but CSV has {} feature columns".format(
                input_data_dims, sum(input_data_dims), input_data.shape[1]
            )
        )
    if not np.isfinite(input_data).all():
        raise ValueError("Input data contains non-finite values")
    return input_data, label, input_data_dims, modal_order


def make_splits(input_data, label, seed, fold_id, protocol):
    outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    outer_splits = list(outer.split(input_data, label))
    outer_train, outer_test = outer_splits[fold_id]
    if protocol == "paper":
        # This reproduces the published code's transductive validation choice
        # (the outer test fold is also used for checkpoint selection).
        return outer_train, outer_test, outer_test, outer_train, outer_test

    inner = StratifiedShuffleSplit(
        n_splits=1, test_size=0.10, random_state=seed + 1000 + fold_id
    )
    rel_train, rel_val = next(inner.split(input_data[outer_train], label[outer_train]))
    train_index = outer_train[rel_train]
    val_index = outer_train[rel_val]
    return train_index, val_index, outer_test, outer_train, outer_test


def _write_rows(path, rows):
    if not rows:
        return
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_fold(args, input_data, label, input_data_dims, fold_id):
    train_index, val_index, test_index, outer_train, outer_test = make_splits(
        input_data, label, args.seed, fold_id, args.protocol
    )
    hyperpm = make_hyperparameters(args)
    agent = EvalHelper(
        input_data_dims,
        input_data,
        label,
        hyperpm,
        train_index,
        val_index,
        test_index,
    )

    start = time.perf_counter()
    best_val_acc = -np.inf
    best_epoch = 0
    wait_count = 0
    best_states = None
    best_val_metrics = None

    for epoch in range(args.nepoch):
        losses = agent.run_epoch(mode="simple-2", epoch=epoch)
        val_prob, _, _ = agent.forward_all(return_modal_tokens=False)
        val_metrics = evaluate_metrics(val_prob, agent.targ, agent.val_idx)
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_val_metrics = val_metrics
            best_epoch = epoch + 1
            best_states = agent.state_dicts()
            wait_count = 0
            print(
                "fold={:02d} epoch={:04d} best_val_acc={:.4f} val_ba={:.4f} "
                "cls={:.5f} align={:.5f} lambda={:.5f}".format(
                    fold_id + 1,
                    epoch + 1,
                    val_metrics["acc"],
                    val_metrics["balanced_acc"],
                    losses["cls"],
                    losses["align"],
                    losses["lambda"],
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
        raise RuntimeError("No checkpoint was produced for fold {}".format(fold_id + 1))
    agent.load_state_dicts(best_states)
    metrics, train_metrics, modal_tokens, final_cls_loss, final_align_loss = (
        agent.best_metrics()
    )
    test_tokens = modal_tokens[agent.tst_idx]
    align_metrics = compute_alignment_metrics(test_tokens, agent.targ[agent.tst_idx])
    elapsed = time.perf_counter() - start

    row = {
        "config": args.config,
        "protocol": args.protocol,
        "seed": args.seed,
        "fold": fold_id + 1,
        "n_outer_train": len(outer_train),
        "n_train": len(train_index),
        "n_val": len(val_index),
        "n_test": len(test_index),
        "best_epoch": best_epoch,
        "val_acc": best_val_metrics["acc"],
        "val_ba": best_val_metrics["balanced_acc"],
        "acc": metrics["acc"],
        "ba": metrics["balanced_acc"],
        "auc": metrics["prob_auc"],
        "f1": metrics["f1"],
        "sen": metrics["sensitivity"],
        "spe": metrics["specificity"],
        "legacy_auc": metrics["legacy_auc"],
        "pos_cos": align_metrics["pos_cos"],
        "neg_cos": align_metrics["neg_cos"],
        "cos_gap": align_metrics["cos_gap"],
        "retrieval_r1": align_metrics["retrieval_r1"],
        "cmmd": align_metrics["cmmd"],
        "train_acc": train_metrics["acc"],
        "train_ba": train_metrics["balanced_acc"],
        "final_cls_loss": float(final_cls_loss),
        "final_align_loss": float(final_align_loss),
        "align_mode": args.align_mode,
        "align_lambda": args.align_lambda,
        "align_temperature": args.align_temperature,
        "align_warmup": args.align_warmup,
        "token_nmodal": int(agent.token_shape[1]),
        "token_dim": int(agent.token_shape[2]),
        "runtime_sec": elapsed,
    }
    # Keep the human-readable names required by the experiment specification
    # in addition to the short names used internally by the aggregator.
    row.update(
        {
            "balanced_acc": row["ba"],
            "prob_auc": row["auc"],
            "sensitivity": row["sen"],
            "specificity": row["spe"],
            "align_pos_cos": row["pos_cos"],
            "align_neg_cos": row["neg_cos"],
            "align_cos_gap": row["cos_gap"],
            "align_retrieval_r1": row["retrieval_r1"],
            "align_cmmd": row["cmmd"],
        }
    )
    print(
        "fold={:02d} done best_epoch={} test_acc={:.4f} test_ba={:.4f} "
        "test_auc={:.4f} sen={:.4f} spe={:.4f} runtime={:.1f}s".format(
            fold_id + 1,
            best_epoch,
            row["acc"],
            row["ba"],
            row["auc"],
            row["sen"],
            row["spe"],
            elapsed,
        )
    )
    del agent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def main():
    parser = argparse.ArgumentParser(description="MMGL ABIDE alignment validation")
    parser.add_argument("--datadir", default="data/")
    parser.add_argument("--datname", default="ABIDE")
    parser.add_argument("--protocol", choices=["strict", "paper"], default="strict")
    parser.add_argument("--config", required=True)
    parser.add_argument("--align_mode", "--align-mode", dest="align_mode", choices=["none", "pair_nce", "cmmd", "hybrid"], default="none")
    parser.add_argument("--align_lambda", "--align-lambda", dest="align_lambda", type=float, default=0.0)
    parser.add_argument("--align_temperature", "--align-temperature", dest="align_temperature", type=float, default=0.10)
    parser.add_argument("--align_warmup", "--align-warmup", dest="align_warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nepoch", type=int, default=1000)
    parser.add_argument("--early", type=int, default=50)
    parser.add_argument("--fold", type=int, default=-1, help="0-based fold, or -1 for all 10")
    parser.add_argument("--fold-results", required=True)
    args = parser.parse_args()

    if args.align_mode == "none":
        args.align_lambda = 0.0
    if args.datname != "ABIDE":
        raise ValueError("This validation runner is intentionally scoped to ABIDE")
    if not (0 <= args.fold < 10 or args.fold == -1):
        raise ValueError("--fold must be -1 or in [0, 9]")

    set_rng_seed(args.seed)
    input_data, label, input_data_dims, modal_order = load_abide(
        args.datadir, args.datname
    )
    print(
        json.dumps(
            {
                "data_shape": list(input_data.shape),
                "label_counts": np.bincount(label).tolist(),
                "modal_order": modal_order,
                "modal_dims": input_data_dims,
                "protocol": args.protocol,
                "config": args.config,
                "seed": args.seed,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            },
            sort_keys=True,
        )
    )
    folds = [args.fold] if args.fold >= 0 else list(range(10))
    rows = [
        run_fold(args, input_data, label, input_data_dims, fold_id)
        for fold_id in folds
    ]
    _write_rows(args.fold_results, rows)
    print("wrote {} rows to {}".format(len(rows), args.fold_results))


if __name__ == "__main__":
    main()
