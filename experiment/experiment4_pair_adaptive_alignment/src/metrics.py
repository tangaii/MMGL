import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def metrics_from_arrays(labels_internal, probs, pred_internal=None):
    """Metrics with ASD (internal label 0) as the positive class."""
    labels_internal = np.asarray(labels_internal, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    if pred_internal is None:
        pred_internal = probs.argmax(axis=1)
    pred_internal = np.asarray(pred_internal, dtype=np.int64)
    y_asd = (labels_internal == 0).astype(np.int64)
    p_asd = probs[:, 0]
    pred_asd = (pred_internal == 0).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_asd, pred_asd, labels=[0, 1]).ravel()
    return {
        "acc": float(accuracy_score(y_asd, pred_asd)),
        "balanced_acc": float(balanced_accuracy_score(y_asd, pred_asd)),
        "asd_auc": float(roc_auc_score(y_asd, p_asd)) if len(np.unique(y_asd)) == 2 else float("nan"),
        "asd_f1": float(f1_score(y_asd, pred_asd, zero_division=0)),
        "asd_sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "nc_specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


@torch.no_grad()
def evaluate_prob(prob, labels, idx):
    probs = prob[idx].exp().detach().cpu().numpy()
    labels_np = labels[idx].detach().cpu().numpy()
    return metrics_from_arrays(labels_np, probs, probs.argmax(axis=1))
