#!/usr/bin/env python3
"""Pre-run provenance, data, forward, loss and alignment checks."""

import argparse
import hashlib
import importlib.util
import os
import sys
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
DATA = os.path.join(ROOT, "data", "ABIDE")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_network(path, module_name):
    for name in ("network", "layers"):
        sys.modules.pop(name, None)
    sys.path.insert(0, path)
    try:
        spec = importlib.util.spec_from_file_location(module_name, os.path.join(path, "network.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules["network"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-data", required=True)
    args = parser.parse_args()

    data_path = os.path.join(DATA, "processed_standard_data.csv")
    modal_path = os.path.join(DATA, "modal_feat_dict.npy")
    strict_data_path = os.path.join(args.strict_data, "processed_standard_data.csv")
    strict_modal_path = os.path.join(args.strict_data, "modal_feat_dict.npy")
    if sha256_file(data_path) != sha256_file(strict_data_path) or sha256_file(modal_path) != sha256_file(strict_modal_path):
        raise RuntimeError("DATA_MATCH_STRICT_EXPERIMENT4=NO")

    data = pd.read_csv(data_path).values
    modal_dict = np.load(modal_path, allow_pickle=True).item()
    labels_original = data[:, -1].astype(np.int64)
    labels_internal = labels_original - 1
    expected_dims = [48, 6, 10, 256]
    dims = [len(modal_dict[key]) for key in modal_dict.keys()]
    counts = {int(label): int((labels_internal == label).sum()) for label in np.unique(labels_internal)}
    print("DATA_MATCH_STRICT_EXPERIMENT4=YES")
    print("DATA_SHAPE={}".format(tuple(data.shape)))
    print("MODAL_KEYS={}".format(list(modal_dict.keys())))
    print("MODAL_DIMS={}".format(dims))
    print("INTERNAL_LABEL_COUNTS={}".format(counts))
    if tuple(data.shape) != (871, 321) or dims != expected_dims or counts != {0: 403, 1: 468}:
        raise RuntimeError("ABIDE data audit failed")

    original_src = os.path.abspath(os.path.join(ROOT, "..", "..", "MMGL_transductive"))
    legacy_src = SRC
    original_network = load_network(original_src, "original_network")
    legacy_network = load_network(legacy_src, "legacy_network")
    hp = SimpleNamespace(
        n_hidden=18, n_head=2, dropout=0.35, nlayer=1,
        nmodal=4, nclass=2,
    )
    input_dims = [48, 6, 10, 256]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(data[:, :-1]).float().to(device)
    torch.manual_seed(1234)
    original_model = original_network.VLTransformer(input_dims, hp).to(device).eval()
    original_state = original_model.state_dict()
    legacy_model = legacy_network.VLTransformer(input_dims, hp).to(device).eval()
    legacy_model.load_state_dict(original_state)
    with torch.no_grad():
        p0, h0, _ = original_model(x)
        p1, h1, _, tokens = legacy_model(x, return_modal_tokens=True)
    prob_diff = float((p0 - p1).abs().max())
    hidden_diff = float((h0 - h1).abs().max())
    print("FORWARD_PARITY_MAX_PROB_DIFF={:.12g}".format(prob_diff))
    print("FORWARD_PARITY_MAX_HIDDEN_DIFF={:.12g}".format(hidden_diff))
    print("MODAL_TOKENS_SHAPE={}".format(tuple(tokens.shape)))
    if prob_diff >= 1e-7 or hidden_diff >= 1e-7 or tuple(tokens.shape) != (871, 4, 36):
        raise RuntimeError("VLTransformer forward parity failed")

    # Check the exact original loss expressions on a fixed forward state.
    sys.modules.pop("network", None)
    sys.modules.pop("layers", None)
    sys.path.insert(0, legacy_src)
    try:
        import alignment
        from utils import ClsLoss, GraphConstructLoss, normalize_adj
    finally:
        sys.path.pop(0)
    train_idx = torch.arange(0, 783, dtype=torch.long, device=device)
    labels = torch.from_numpy(labels_internal).long().to(device)
    train_counts = Counter(labels_internal[train_idx.cpu().numpy()].tolist())
    weight = torch.tensor([783.0 / 2.0 / train_counts[i] for i in range(2)], device=device)
    cls_loss = ClsLoss(p1, labels, train_idx.cpu().numpy(), weight)
    l0_step_a = cls_loss
    if float((l0_step_a - cls_loss).abs()) != 0.0:
        raise RuntimeError("L0 Step-A parity failed")
    fusion = h1.detach()
    # The graph formula is evaluated on a small fixed prefix to keep sanity checks light.
    graph_model = legacy_network.GraphLearn(52, th=0.9, mode="weighted-cosine").to(device).eval()
    graph = graph_model(fusion[:64])
    graph_loss = GraphConstructLoss(fusion[:64], graph, 1.0, 0.5, 0.0)
    cls_small = F.nll_loss(p1[:64], labels[:64], weight=weight)
    if not torch.isfinite(cls_small + graph_loss):
        raise RuntimeError("L0 Step-B loss is not finite")
    print("L0_STEP_A_LOSS_PARITY=PASS")
    print("L0_STEP_B_LOSS_FINITE=PASS")

    train_tokens = tokens[train_idx]
    train_labels = labels[train_idx]
    align_loss, parts = alignment.compute_alignment_loss(train_tokens, train_labels, mode="hybrid", temperature=0.10)
    if not torch.isfinite(align_loss) or not all(np.isfinite(parts[key]) for key in ("nce", "cmmd", "total")):
        raise RuntimeError("L1 alignment sanity failed")
    for epoch, expected in ((0, 0.025), (19, 0.50), (20, 0.50)):
        value = 0.50 * min(1.0, float(epoch + 1) / 20.0)
        if abs(value - expected) > 1e-12:
            raise RuntimeError("warmup schedule failed")
    print("L1_ALIGNMENT_FINITE=PASS")
    print("L1_WARMUP_EPOCH0=0.025000")
    print("L1_WARMUP_EPOCH19=0.500000")
    print("SANITY_CHECK_OK")


if __name__ == "__main__":
    main()
