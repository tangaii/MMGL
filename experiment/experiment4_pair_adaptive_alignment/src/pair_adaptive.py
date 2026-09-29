import torch
import torch.nn.functional as F

from alignment import PAIR_INDICES


SHUFFLE_PERM = [1, 2, 3, 4, 5, 0]


def gradient_agreement_scores(
    cls_loss,
    pair_losses,
    modal_tokens,
    train_idx,
    pair_indices=PAIR_INDICES,
    eps=1e-12,
):
    """Return detached cosine agreement scores for the six modality pairs."""
    g_cls_all = torch.autograd.grad(
        cls_loss,
        modal_tokens,
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )[0].detach()

    scores = []
    for (m1, m2), pair_loss in zip(pair_indices, pair_losses):
        g_pair_all = torch.autograd.grad(
            pair_loss,
            modal_tokens,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0].detach()
        g_cls_pair = torch.cat([
            g_cls_all[train_idx, m1, :].reshape(-1),
            g_cls_all[train_idx, m2, :].reshape(-1),
        ])
        g_align_pair = torch.cat([
            g_pair_all[train_idx, m1, :].reshape(-1),
            g_pair_all[train_idx, m2, :].reshape(-1),
        ])
        norm_cls = torch.linalg.vector_norm(g_cls_pair)
        norm_align = torch.linalg.vector_norm(g_align_pair)
        if float(norm_cls) < eps or float(norm_align) < eps:
            score = modal_tokens.new_tensor(0.0)
        else:
            score = F.cosine_similarity(
                g_cls_pair.unsqueeze(0), g_align_pair.unsqueeze(0), dim=1
            )[0]
        scores.append(torch.clamp(score, -1.0, 1.0))
    return torch.stack(scores).detach()


def normalized_weights(weights):
    weights = weights.detach()
    return weights / weights.mean().clamp_min(1e-8)


def balanced_sparse_weights(epoch, device=None, dtype=torch.float32):
    weights = torch.zeros(6, device=device, dtype=dtype)
    active = [0, 2, 4] if epoch % 2 == 0 else [1, 3, 5]
    weights[active] = 2.0
    return weights


def normalized_weight_entropy(weights):
    weights = weights.detach().float()
    if torch.any(weights < 0) or float(weights.sum()) <= 0:
        return float("nan")
    p = weights / weights.sum()
    positive = p[p > 0]
    return float(-(positive * positive.log()).sum() / torch.log(weights.new_tensor(6.0)))


class PairWeightController:
    def __init__(self, n_pairs=6, ema_beta=0.90, device=None):
        self.n_pairs = n_pairs
        self.ema_beta = ema_beta
        self.ema_weights = torch.ones(n_pairs, device=device)
        self.last_scores = torch.zeros(n_pairs, device=device)
        self.initialized = False
        self.last_update_epoch = -1

    def update(self, scores, epoch):
        scores = scores.detach().to(self.ema_weights.device).float().clamp(-1.0, 1.0)
        if scores.numel() != self.n_pairs:
            raise ValueError("score count mismatch")
        proposed = normalized_weights(1.0 + 0.75 * scores)
        if not self.initialized:
            self.ema_weights = proposed
            self.initialized = True
        else:
            self.ema_weights = normalized_weights(
                self.ema_beta * self.ema_weights
                + (1.0 - self.ema_beta) * proposed
            )
        self.last_scores = scores.clone().detach()
        self.last_update_epoch = int(epoch)

    def get_adaptive_weights(self):
        if not self.initialized:
            return torch.ones(self.n_pairs, device=self.ema_weights.device)
        return normalized_weights(self.ema_weights)

    def get_shuffled_weights(self):
        adaptive = self.get_adaptive_weights()
        return normalized_weights(adaptive[SHUFFLE_PERM])

    def get_topk_weights(self, k=3):
        adaptive = self.get_adaptive_weights()
        indices = torch.topk(adaptive, k=k).indices
        selected = torch.zeros_like(adaptive)
        selected[indices] = adaptive[indices]
        return normalized_weights(selected)

    def state_dict(self):
        return {
            "n_pairs": self.n_pairs,
            "ema_beta": self.ema_beta,
            "ema_weights": self.ema_weights.detach().cpu().clone(),
            "last_scores": self.last_scores.detach().cpu().clone(),
            "initialized": self.initialized,
            "last_update_epoch": self.last_update_epoch,
        }

    def load_state_dict(self, state):
        device = self.ema_weights.device
        self.n_pairs = int(state["n_pairs"])
        self.ema_beta = float(state["ema_beta"])
        self.ema_weights = state["ema_weights"].to(device=device).float().clone()
        self.last_scores = state["last_scores"].to(device=device).float().clone()
        self.initialized = bool(state["initialized"])
        self.last_update_epoch = int(state["last_update_epoch"])
