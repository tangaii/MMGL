#!/usr/bin/env python3
"""MMGL legacy-protocol runner.

This file intentionally keeps the original transductive training protocol:
the test fold is also used as ``val_idx`` for checkpoint selection, and RNG
state is continuous across folds within one process.  The only experimental
addition is an optional train-fold-only token alignment term in Step A.
"""

import argparse
import csv
import hashlib
import os
import random
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch

from model import EvalHelper


def set_rng_seed(seed):
    """The original repository seed routine, called only at process setup."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256_bytes(parts):
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def split_fingerprint(train_index, test_index):
    train_index = np.asarray(train_index, dtype=np.int64)
    test_index = np.asarray(test_index, dtype=np.int64)
    return sha256_bytes([
        b"train\0", train_index.tobytes(),
        b"test\0", test_index.tobytes(),
        b"val_equals_test\0", b"1",
    ])


def module_fingerprint(agent):
    """Hash only the initial parameter/state tensors; no random calls."""
    digest = hashlib.sha256()
    for module_name, module in (
        ("ModalFusion", agent.ModalFusion),
        ("GraphConstruct", agent.GraphConstruct),
        ("MessagePassing", agent.MessagePassing),
    ):
        for tensor_name, tensor in module.state_dict().items():
            tensor_cpu = tensor.detach().cpu().contiguous()
            digest.update((module_name + "\0" + tensor_name + "\0").encode("utf-8"))
            digest.update(str(tensor_cpu.dtype).encode("utf-8"))
            digest.update(str(tuple(tensor_cpu.shape)).encode("utf-8"))
            digest.update(tensor_cpu.numpy().tobytes())
    return digest.hexdigest()


def load_abide(datadir, datname):
    path = os.path.join(datadir, datname)
    modal_feat_dict = np.load(
        os.path.join(path, "modal_feat_dict.npy"), allow_pickle=True
    ).item()
    data = pd.read_csv(os.path.join(path, "processed_standard_data.csv")).values
    input_data_dims = [len(modal_feat_dict[key]) for key in modal_feat_dict.keys()]
    input_data = data[:, :-1].astype(np.float32, copy=False)
    labels_original = data[:, -1].astype(np.int64, copy=False)
    labels_internal = labels_original - 1
    return input_data_dims, input_data, labels_original, labels_internal


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        raise RuntimeError("No rows were produced for {}".format(path))
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser():
    parser = argparse.ArgumentParser()
    # Original repository defaults are retained, with the ABIDE paper
    # configuration supplied explicitly by run_all.sh.
    parser.add_argument("--datadir", type=str, default="./data/")
    parser.add_argument("--datname", type=str, default="ABIDE")
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--nepoch", type=int, default=1000)
    parser.add_argument("--early", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0038)
    parser.add_argument("--reg", type=float, default=0.11)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--nlayer", type=int, default=1)
    parser.add_argument("--n_hidden", type=int, default=18)
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--n_iter", type=int, default=10)
    parser.add_argument("--nmodal", type=int, default=4)
    parser.add_argument("--th", type=float, default=0.9)
    parser.add_argument("--GC_mode", type=str, default="weighted-cosine")
    parser.add_argument("--MP_mode", type=str, default="GCN")
    parser.add_argument("--MF_mode", type=str, default=" ")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--theta_smooth", type=float, default=1.0)
    parser.add_argument("--theta_degree", type=float, default=0.5)
    parser.add_argument("--theta_sparsity", type=float, default=0.0)
    parser.add_argument("--nclass", type=int, default=2)
    parser.add_argument("--mode", type=str, default="simple-2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--align_mode", type=str, choices=("none", "hybrid"), default="none")
    parser.add_argument("--align_lambda", type=float, default=0.0)
    parser.add_argument("--align_temperature", type=float, default=0.10)
    parser.add_argument("--align_warmup", type=int, default=20)
    parser.add_argument("--single_fold", type=int, default=None)
    parser.add_argument("--fold_results", type=str, required=True)
    parser.add_argument("--oof_results", type=str, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.config == "L0_legacy_baseline":
        expected_mode, expected_lambda = "none", 0.0
    elif args.config == "L1_legacy_uniform_global":
        expected_mode, expected_lambda = "hybrid", 0.50
    else:
        raise ValueError("Unknown legacy config: {}".format(args.config))
    if args.align_mode != expected_mode or abs(args.align_lambda - expected_lambda) > 1e-12:
        raise ValueError("Config/alignment mismatch for {}".format(args.config))

    print("LEGACY_PROTOCOL=True")
    print("CONFIG={}".format(args.config))
    print("SEED={}".format(args.seed))
    print("DATASET={}/{}".format(args.datadir.rstrip("/"), args.datname))
    print("SPLIT=StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)")
    print("VAL_EQUALS_TEST=True")
    print("CHECKPOINT_SELECTION=legacy_test_fold_acc")
    print("EARLY_STOP=wait_cnt>50")
    print("NEPOCH=1000")
    print("RNG_RESEED_PER_FOLD=False")
    print("ALIGN_MODE={}".format(args.align_mode))
    print("ALIGN_LAMBDA={:.2f}".format(args.align_lambda))
    print("ALIGN_TEMPERATURE={:.2f}".format(args.align_temperature))
    print("ALIGN_WARMUP={}".format(args.align_warmup))
    print("ALIGNMENT_LABEL_SCOPE=train_idx_only")

    # Match the original train_and_eval order: seed once before setup,
    # construct the splitter, then seed once before the fold loop.
    set_rng_seed(args.seed)
    input_data_dims, input_data, labels_original, labels_internal = load_abide(
        args.datadir, args.datname
    )
    args.nclass = 2
    args.nmodal = 4
    print("DATA_SHAPE={}".format((input_data.shape[0], input_data.shape[1] + 1)))
    print("MODAL_DIMS={}".format(input_data_dims))
    print("ORIGINAL_LABEL_COUNTS={}".format({
        int(k): int(v) for k, v in zip(*np.unique(labels_original, return_counts=True))
    }))
    print("INTERNAL_LABEL_COUNTS={}".format({
        int(k): int(v) for k, v in zip(*np.unique(labels_internal, return_counts=True))
    }))

    skf = StratifiedKFold(
        n_splits=10, random_state=args.seed, shuffle=True
    )
    set_rng_seed(args.seed)

    fold_rows = []
    oof_rows = []
    test_acc_history = []
    start_job = time.time()
    processed_folds = 0

    for fold_zero, (train_index, test_index) in enumerate(
        skf.split(input_data, labels_internal)
    ):
        if args.single_fold is not None and fold_zero != args.single_fold:
            continue
        fold_number = fold_zero + 1
        processed_folds += 1
        split_sha = split_fingerprint(train_index, test_index)
        print("\n=== FOLD {}/10 ===".format(fold_number))
        print("N_TRAIN={} N_TEST={}".format(len(train_index), len(test_index)))
        print("SPLIT_SHA256={}".format(split_sha))

        agent = EvalHelper(
            input_data_dims, input_data, labels_internal, args,
            train_index, test_index,
        )
        base_init_sha = module_fingerprint(agent)
        print("BASE_INIT_SHA256={}".format(base_init_sha))

        tm = time.time()
        best_val_acc = 0.0
        wait_cnt = 0
        best_epoch = -1
        epochs_run = 0
        model_sav = tempfile.TemporaryFile()

        for epoch in range(args.nepoch):
            print("%3d/%d" % (epoch, args.nepoch), end=" ")
            agent.run_epoch(mode=args.mode, end=" ", epoch=epoch)
            _, cur_val_acc = agent.print_trn_acc(args.mode)
            epochs_run = epoch + 1
            # This strict > comparison and test-fold selection are original.
            if cur_val_acc > best_val_acc:
                wait_cnt = 0
                best_val_acc = cur_val_acc
                best_epoch = epoch
                model_sav.close()
                model_sav = tempfile.TemporaryFile()
                dict_list = [
                    agent.ModalFusion.state_dict(),
                    agent.GraphConstruct.state_dict(),
                    agent.MessagePassing.state_dict(),
                ]
                torch.save(dict_list, model_sav)
            else:
                wait_cnt += 1
                if wait_cnt > args.early:
                    break
        elapsed = time.time() - tm
        print("time: %.4f sec." % elapsed)

        model_sav.seek(0)
        try:
            dict_list = torch.load(model_sav, weights_only=False)
        except TypeError:
            dict_list = torch.load(model_sav)
        agent.ModalFusion.load_state_dict(dict_list[0])
        agent.GraphConstruct.load_state_dict(dict_list[1])
        agent.MessagePassing.load_state_dict(dict_list[2])

        evaluated = agent.evaluate_test_metrics()
        metrics = evaluated["fold_metrics"]
        representation = evaluated["representation"]
        test_acc_history.append(metrics["acc"])
        legacy_early_abort = bool(
            len(test_acc_history) == 5
            and float(np.mean(test_acc_history)) < 0.6
        )
        fold_row = {
            "config": args.config,
            "seed": args.seed,
            "fold": fold_number,
            "n_train": len(train_index),
            "n_test": len(test_index),
            "val_equals_test": True,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "test_acc": metrics["acc"],
            "legacy_hard_auc": metrics["legacy_hard_auc"],
            "balanced_acc": metrics["balanced_acc"],
            "asd_probability_auc": metrics["asd_probability_auc"],
            "asd_f1": metrics["asd_f1"],
            "asd_sensitivity": metrics["asd_sensitivity"],
            "nc_specificity": metrics["nc_specificity"],
            "alignment_pos_cos": representation["pos_cos"],
            "alignment_neg_cos": representation["neg_cos"],
            "alignment_cos_gap": representation["cos_gap"],
            "alignment_R1": representation["retrieval_r1"],
            "alignment_CMMD": representation["cmmd"],
            "final_stepA_cls_loss": evaluated["final_stepA_cls_loss"],
            "final_align_loss": evaluated["final_align_loss"],
            "final_align_nce": evaluated["final_align_parts"]["nce"],
            "final_align_cmmd": evaluated["final_align_parts"]["cmmd"],
            "base_init_sha256": base_init_sha,
            "split_sha256": split_sha,
            "epochs_run": epochs_run,
            "runtime_sec": elapsed,
            "legacy_early_abort": legacy_early_abort,
            "align_mode": args.align_mode,
            "align_lambda": args.align_lambda,
            "align_temperature": args.align_temperature,
            "align_warmup": args.align_warmup,
        }
        fold_rows.append(fold_row)
        for row_number, subject_index in enumerate(test_index):
            oof_rows.append({
                "config": args.config,
                "seed": args.seed,
                "fold": fold_number,
                "subject_index": int(subject_index),
                "true_original_label": int(labels_original[subject_index]),
                "true_internal_label": int(labels_internal[subject_index]),
                "true_asd": int(labels_internal[subject_index] == 0),
                "p_asd": float(evaluated["p_asd"][row_number]),
                "pred_internal": int(evaluated["pred_internal"][row_number]),
                "pred_asd": int(evaluated["pred_asd"][row_number]),
                "legacy_selected_epoch": best_epoch,
            })
        print(
            "FOLD_RESULT acc={:.6f} hard_auc={:.6f} prob_auc={:.6f} "
            "BA={:.6f} ASD_SEN={:.6f} NC_SPE={:.6f} best_epoch={} "
            "cos_gap={:.6f} R1={:.6f} CMMD={:.6f}".format(
                metrics["acc"], metrics["legacy_hard_auc"],
                metrics["asd_probability_auc"], metrics["balanced_acc"],
                metrics["asd_sensitivity"], metrics["nc_specificity"],
                best_epoch, representation["cos_gap"],
                representation["retrieval_r1"], representation["cmmd"],
            )
        )
        if legacy_early_abort:
            print("LEGACY_EARLY_ABORT_TRIGGERED=True")
            break

    if processed_folds == 0:
        raise RuntimeError("No fold was selected")
    write_csv(args.fold_results, fold_rows)
    write_csv(args.oof_results, oof_rows)
    print("\nJOB_DONE config={} seed={} folds={} oof_rows={} runtime_sec={:.3f}".format(
        args.config, args.seed, len(fold_rows), len(oof_rows), time.time() - start_job
    ))
    print("FOLD_RESULTS={}".format(os.path.abspath(args.fold_results)))
    print("OOF_RESULTS={}".format(os.path.abspath(args.oof_results)))


if __name__ == "__main__":
    main()
