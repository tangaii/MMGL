import math
from itertools import combinations

import torch
import torch.nn.functional as F


def normalize_tokens(tokens, eps=1e-8):
    return F.normalize(tokens, p=2, dim=-1, eps=eps)


def _off_diagonal_mean(matrix):
    n = matrix.size(0)
    if n <= 1:
        return matrix.new_tensor(0.0)
    return (matrix.sum() - torch.diagonal(matrix).sum()) / float(n * (n - 1))


def paired_info_nce(tokens, temperature=0.10):
    z = normalize_tokens(tokens)
    batch_size, modal_num, _ = z.shape
    if batch_size < 2 or modal_num < 2:
        return z.sum() * 0.0

    targets = torch.arange(batch_size, device=z.device)
    normalizer = max(math.log(float(batch_size)), 1.0)
    losses = []
    for m1, m2 in combinations(range(modal_num), 2):
        logits = torch.matmul(z[:, m1, :], z[:, m2, :].transpose(0, 1))
        logits = logits / temperature
        loss_12 = F.cross_entropy(logits, targets)
        loss_21 = F.cross_entropy(logits.transpose(0, 1), targets)
        losses.append(0.5 * (loss_12 + loss_21) / normalizer)
    return torch.stack(losses).mean()


def _multi_rbf_kernel(x, y, sigmas=(0.25, 0.5, 1.0, 2.0)):
    dist2 = torch.cdist(x, y, p=2).pow(2)
    kernels = []
    for sigma in sigmas:
        kernels.append(torch.exp(-dist2 / (2.0 * sigma * sigma)))
    return torch.stack(kernels, dim=0).mean(dim=0)


def _mmd2_biased(x, y):
    k_xx = _multi_rbf_kernel(x, x)
    k_yy = _multi_rbf_kernel(y, y)
    k_xy = _multi_rbf_kernel(x, y)
    return torch.clamp(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean(), min=0.0)


def class_conditional_mmd(tokens, labels):
    z = normalize_tokens(tokens)
    batch_size, modal_num, _ = z.shape
    if batch_size < 2 or modal_num < 2:
        return z.sum() * 0.0

    losses = []
    for cls in torch.unique(labels):
        mask = labels == cls
        if int(mask.sum().item()) < 2:
            continue
        z_cls = z[mask]
        for m1, m2 in combinations(range(modal_num), 2):
            losses.append(_mmd2_biased(z_cls[:, m1, :], z_cls[:, m2, :]))

    if not losses:
        return z.sum() * 0.0
    return torch.stack(losses).mean()


def compute_alignment_loss(tokens, labels, mode="none", temperature=0.10):
    zero = tokens.sum() * 0.0
    if mode == "none":
        return zero, {"nce": 0.0, "cmmd": 0.0, "total": 0.0}

    if mode == "pair_nce":
        nce = paired_info_nce(tokens, temperature)
        return nce, {"nce": float(nce.detach()), "cmmd": 0.0, "total": float(nce.detach())}

    if mode == "cmmd":
        cmmd = class_conditional_mmd(tokens, labels)
        return cmmd, {"nce": 0.0, "cmmd": float(cmmd.detach()), "total": float(cmmd.detach())}

    if mode == "hybrid":
        nce = paired_info_nce(tokens, temperature)
        cmmd = class_conditional_mmd(tokens, labels)
        total = 0.5 * nce + 0.5 * cmmd
        return total, {
            "nce": float(nce.detach()),
            "cmmd": float(cmmd.detach()),
            "total": float(total.detach()),
        }

    raise ValueError("Unknown alignment mode: {}".format(mode))


@torch.no_grad()
def compute_alignment_metrics(tokens, labels):
    z = normalize_tokens(tokens)
    batch_size, modal_num, _ = z.shape
    pos_cos, neg_cos, retrieval = [], [], []
    if batch_size >= 2 and modal_num >= 2:
        target = torch.arange(batch_size, device=z.device)
        for m1, m2 in combinations(range(modal_num), 2):
            sim = torch.matmul(z[:, m1, :], z[:, m2, :].transpose(0, 1))
            pos_cos.append(torch.diagonal(sim).mean())
            neg_cos.append(_off_diagonal_mean(sim))
            r12 = (sim.argmax(dim=1) == target).float().mean()
            r21 = (sim.argmax(dim=0) == target).float().mean()
            retrieval.append(0.5 * (r12 + r21))

    if pos_cos:
        pos = torch.stack(pos_cos).mean()
        neg = torch.stack(neg_cos).mean()
        r1 = torch.stack(retrieval).mean()
    else:
        pos = z.new_tensor(0.0)
        neg = z.new_tensor(0.0)
        r1 = z.new_tensor(0.0)

    return {
        "pos_cos": float(pos),
        "neg_cos": float(neg),
        "cos_gap": float(pos - neg),
        "retrieval_r1": float(r1),
        "cmmd": float(class_conditional_mmd(z, labels)),
    }
