"""Metrics for the original MMGL legacy evaluation protocol."""

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def legacy_hard_auc(labels_internal, pred_internal, nclass=2):
    """Copy the original repository's one-hot hard-prediction AUC."""
    labels_internal = np.asarray(labels_internal, dtype=np.int64)
    pred_internal = np.asarray(pred_internal, dtype=np.int64)
    target = np.eye(nclass, dtype=np.float64)[labels_internal]
    prediction = np.eye(nclass, dtype=np.float64)[pred_internal]
    return float(roc_auc_score(target, prediction))


def _safe_auc(labels, score):
    return float(roc_auc_score(labels, score)) if len(np.unique(labels)) == 2 else float("nan")


def probability_metrics(labels_internal, log_prob, pred_internal=None):
    """Report ASD-positive metrics from log probabilities, post hoc only."""
    labels_internal = np.asarray(labels_internal, dtype=np.int64)
    if hasattr(log_prob, "detach"):
        log_prob = log_prob.detach().cpu().numpy()
    log_prob = np.asarray(log_prob, dtype=np.float64)
    probabilities = np.exp(log_prob)
    p_asd = probabilities[:, 0]
    if pred_internal is None:
        pred_internal = np.argmax(log_prob, axis=1)
    pred_internal = np.asarray(pred_internal, dtype=np.int64)
    y_asd = (labels_internal == 0).astype(np.int64)
    pred_asd = (pred_internal == 0).astype(np.int64)
    tp = int(np.sum((y_asd == 1) & (pred_asd == 1)))
    tn = int(np.sum((y_asd == 0) & (pred_asd == 0)))
    fp = int(np.sum((y_asd == 0) & (pred_asd == 1)))
    fn = int(np.sum((y_asd == 1) & (pred_asd == 0)))
    sensitivity = float(tp / (tp + fn)) if tp + fn else 0.0
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    return {
        "acc": float(np.mean(pred_internal == labels_internal)),
        "balanced_acc": 0.5 * (sensitivity + specificity),
        "asd_probability_auc": _safe_auc(y_asd, p_asd),
        "asd_f1": float(f1_score(y_asd, pred_asd, zero_division=0)),
        "asd_sensitivity": sensitivity,
        "nc_specificity": specificity,
        "p_asd": p_asd,
        "pred_asd": pred_asd,
    }


def fold_metrics(labels_internal, log_prob, pred_internal=None):
    labels_internal = np.asarray(labels_internal, dtype=np.int64)
    if hasattr(log_prob, "detach"):
        log_prob_cpu = log_prob.detach().cpu()
    else:
        log_prob_cpu = log_prob
    if pred_internal is None:
        pred_internal = np.asarray(log_prob_cpu.argmax(dim=1).numpy(), dtype=np.int64)
    else:
        pred_internal = np.asarray(pred_internal, dtype=np.int64)
    output = probability_metrics(labels_internal, log_prob_cpu, pred_internal)
    output["legacy_hard_auc"] = legacy_hard_auc(labels_internal, pred_internal)
    output["pred_internal"] = pred_internal
    return output
