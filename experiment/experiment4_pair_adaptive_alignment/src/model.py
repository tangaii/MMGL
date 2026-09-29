import copy
import hashlib

import numpy as np
import torch

from alignment import (
    compute_alignment_loss,
    compute_alignment_metrics,
    compute_pair_alignment_metrics,
    compute_pairwise_hybrid_losses,
    weighted_pair_alignment_loss,
)
from metrics import evaluate_prob
from network import GAT, GCN, GraphLearn, VLTransformer, VLTransformer_Gate
from pair_adaptive import (
    PairWeightController,
    balanced_sparse_weights,
    gradient_agreement_scores,
    normalized_weight_entropy,
)
from utils import ClsLoss, GraphConstructLoss, my_weight_init, normalize_adj


def module_fingerprint(modules):
    digest = hashlib.sha256()
    for module in modules:
        for name, value in module.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class EvalHelper:
    def __init__(
        self,
        input_data_dims,
        feat,
        label,
        hyperpm,
        train_index,
        val_index,
        test_index,
        modality_names,
    ):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hyperpm = hyperpm
        self.GC_mode = hyperpm.GC_mode
        self.MP_mode = hyperpm.MP_mode
        self.MF_mode = hyperpm.MF_mode
        self.align_strategy = hyperpm.align_strategy
        self.align_mode = hyperpm.align_mode
        self.align_lambda = hyperpm.align_lambda
        self.align_temperature = hyperpm.align_temperature
        self.align_warmup = hyperpm.align_warmup
        self.modality_names = list(modality_names)

        self.d_v = hyperpm.n_hidden
        self.modal_num = hyperpm.nmodal
        self.n_class = hyperpm.nclass
        self.dropout = hyperpm.dropout
        self.alpha = hyperpm.alpha
        self.n_head = hyperpm.n_head
        self.th = hyperpm.th

        self.feat = torch.from_numpy(feat).float().to(self.dev)
        self.targ = torch.from_numpy(label).long().to(self.dev)
        self.trn_idx = torch.as_tensor(train_index, dtype=torch.long, device=self.dev)
        self.val_idx = torch.as_tensor(val_index, dtype=torch.long, device=self.dev)
        self.tst_idx = torch.as_tensor(test_index, dtype=torch.long, device=self.dev)

        train_y_np = self.targ[self.trn_idx].detach().cpu().numpy()
        counts = np.bincount(train_y_np, minlength=self.n_class).astype(np.float64)
        class_weight = len(train_y_np) / (self.n_class * np.maximum(counts, 1.0))
        self.weight = torch.tensor(class_weight, dtype=torch.float32, device=self.dev)

        self.out_dim = self.d_v * self.n_head + self.modal_num ** 2

        # This initialization order is shared by all six configurations.
        if self.MF_mode == "sum":
            self.ModalFusion = VLTransformer_Gate(input_data_dims, hyperpm).to(self.dev)
        else:
            self.ModalFusion = VLTransformer(input_data_dims, hyperpm).to(self.dev)
        self.ModalFusion.apply(my_weight_init)
        self.GraphConstruct = GraphLearn(
            self.out_dim, th=self.th, mode=self.GC_mode
        ).to(self.dev)
        if self.MP_mode == "GCN":
            self.MessagePassing = GCN(
                self.out_dim, self.out_dim // 2, self.n_class, self.dropout
            ).to(self.dev)
        elif self.MP_mode == "GAT":
            self.MessagePassing = GAT(
                self.out_dim, self.out_dim // 2, self.n_class, self.dropout,
                self.alpha, nheads=2,
            ).to(self.dev)
        else:
            raise ValueError("Unsupported MP_mode: {}".format(self.MP_mode))

        self.base_init_sha256 = module_fingerprint([
            self.ModalFusion, self.GraphConstruct, self.MessagePassing
        ])

        self.optimizer_MF = torch.optim.Adam(
            self.ModalFusion.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg
        )
        self.optimizer_GC = torch.optim.Adam(
            self.GraphConstruct.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg
        )
        self.optimizer_MP = torch.optim.Adam(
            self.MessagePassing.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg
        )

        self.pair_controller = None
        if self.align_strategy in ("adaptive", "shuffled", "top3"):
            self.pair_controller = PairWeightController(device=self.dev)
        self.current_pair_weights = torch.ones(6, device=self.dev)
        self.current_pair_scores = torch.full((6,), float("nan"), device=self.dev)
        self.token_shape = None
        self.last_losses = {
            "cls": 0.0, "align": 0.0, "graph": 0.0,
            "lambda_align": 0.0, "pair_weight_std": 0.0,
        }

    def _warmup_scale(self, epoch):
        if self.align_warmup <= 0:
            return 1.0
        return min(1.0, float(epoch + 1) / float(self.align_warmup))

    def _lambda_eff(self, epoch):
        if self.align_strategy == "none":
            return 0.0
        return self.align_lambda * self._warmup_scale(epoch)

    def _effective_pair_weights(self, epoch):
        if self.align_strategy == "uniform":
            return torch.ones(6, device=self.dev)
        if self.align_strategy == "sparse":
            return balanced_sparse_weights(epoch, device=self.dev)
        if self.pair_controller is None:
            raise RuntimeError("adaptive configuration has no pair controller")
        if self.align_strategy == "adaptive":
            return self.pair_controller.get_adaptive_weights()
        if self.align_strategy == "shuffled":
            return self.pair_controller.get_shuffled_weights()
        if self.align_strategy == "top3":
            return self.pair_controller.get_topk_weights(k=3)
        raise ValueError("Unknown alignment strategy: {}".format(self.align_strategy))

    def _update_pair_controller(self, cls_loss, pair_losses, modal_tokens, epoch):
        if self.pair_controller is None:
            return
        if epoch % self.hyperpm.weight_update_interval != 0:
            return
        scores = gradient_agreement_scores(
            cls_loss, pair_losses, modal_tokens, self.trn_idx
        )
        self.pair_controller.update(scores, epoch)
        self.current_pair_scores = scores.detach().clone()

    def run_epoch(self, mode="simple-2", epoch=0):
        if mode != "simple-2":
            raise ValueError("This runner only supports simple-2")

        # Step A: original MMGL modal prediction plus pair alignment.
        self.ModalFusion.train()
        self.GraphConstruct.eval()
        self.MessagePassing.eval()
        self.optimizer_MF.zero_grad()
        self.optimizer_GC.zero_grad()
        self.optimizer_MP.zero_grad()

        prob, _, _, modal_tokens = self.ModalFusion(
            self.feat, return_modal_tokens=True
        )
        if modal_tokens.dim() != 3:
            raise RuntimeError("Expected [N,M,D] modal tokens")
        batch_size, modal_count, token_dim = modal_tokens.shape
        if (batch_size, modal_count, token_dim) != (self.feat.size(0), 4, 36):
            raise RuntimeError("Unexpected modal token shape: {}".format(tuple(modal_tokens.shape)))
        self.token_shape = (int(batch_size), int(modal_count), int(token_dim))
        cls_loss = ClsLoss(prob, self.targ, self.trn_idx, self.weight)

        if self.align_strategy == "none":
            pair_losses = []
            align_loss = modal_tokens.sum() * 0.0
            weights = torch.zeros(6, device=self.dev)
        elif self.align_strategy == "uniform":
            # Preserve the Experiment 3 global hybrid algebra bit-for-bit as
            # the integrity control. The pair-wise weighted implementation is
            # checked against this reference in the mandatory sanity check.
            train_tokens = modal_tokens[self.trn_idx]
            train_labels = self.targ[self.trn_idx]
            align_loss, _ = compute_alignment_loss(
                train_tokens,
                train_labels,
                mode="hybrid",
                temperature=self.align_temperature,
            )
            pair_losses = []
            weights = torch.ones(6, device=self.dev)
        else:
            train_tokens = modal_tokens[self.trn_idx]
            train_labels = self.targ[self.trn_idx]
            pair_losses, _ = compute_pairwise_hybrid_losses(
                train_tokens, train_labels, temperature=self.align_temperature
            )
            self._update_pair_controller(cls_loss, pair_losses, modal_tokens, epoch)
            weights = self._effective_pair_weights(epoch)
            align_loss = weighted_pair_alignment_loss(pair_losses, weights)

        total_mf_loss = cls_loss + self._lambda_eff(epoch) * align_loss
        total_mf_loss.backward()
        self.optimizer_MF.step()
        self.current_pair_weights = weights.detach().clone()

        # Step B: original MMGL graph stage. The 52-D fusion features are detached.
        self.ModalFusion.eval()
        self.GraphConstruct.train()
        self.MessagePassing.train()
        self.optimizer_MF.zero_grad()
        self.optimizer_GC.zero_grad()
        self.optimizer_MP.zero_grad()

        _, embedding, _ = self.ModalFusion(self.feat)
        fusion_feat = embedding.detach()
        adj = self.GraphConstruct(fusion_feat)
        graph_loss = GraphConstructLoss(
            fusion_feat, adj,
            self.hyperpm.theta_smooth,
            self.hyperpm.theta_degree,
            self.hyperpm.theta_sparsity,
        )
        normalized_adj = normalize_adj(adj + torch.eye(adj.size(0), device=self.dev))
        graph_prob, _ = self.MessagePassing(fusion_feat, normalized_adj)
        graph_cls_loss = ClsLoss(graph_prob, self.targ, self.trn_idx, self.weight)
        (graph_cls_loss + graph_loss).backward()
        self.optimizer_GC.step()
        self.optimizer_MP.step()

        self.last_losses = {
            "cls": float(graph_cls_loss.detach()),
            "align": float(align_loss.detach()),
            "graph": float(graph_loss.detach()),
            "lambda_align": float(self._lambda_eff(epoch)),
            "pair_weight_std": float(weights.std(unbiased=False)),
        }
        return self.last_losses

    @torch.no_grad()
    def forward_all(self, return_modal_tokens=True):
        self.ModalFusion.eval()
        self.GraphConstruct.eval()
        self.MessagePassing.eval()
        output = self.ModalFusion(self.feat, return_modal_tokens=return_modal_tokens)
        if return_modal_tokens:
            _, fusion_feat, _, modal_tokens = output
        else:
            _, fusion_feat, _ = output
            modal_tokens = None
        adj = self.GraphConstruct(fusion_feat)
        normalized_adj = normalize_adj(adj + torch.eye(adj.size(0), device=self.dev))
        prob, _ = self.MessagePassing(fusion_feat, normalized_adj)
        return prob, fusion_feat, modal_tokens

    def state_dicts(self):
        controller_state = None
        if self.pair_controller is not None:
            controller_state = copy.deepcopy(self.pair_controller.state_dict())
        return {
            "ModalFusion": copy.deepcopy(self.ModalFusion.state_dict()),
            "GraphConstruct": copy.deepcopy(self.GraphConstruct.state_dict()),
            "MessagePassing": copy.deepcopy(self.MessagePassing.state_dict()),
            "pair_controller": controller_state,
            "pair_weights": self.current_pair_weights.detach().cpu().clone(),
            "pair_scores": self.current_pair_scores.detach().cpu().clone(),
        }

    def load_state_dicts(self, states):
        self.ModalFusion.load_state_dict(states["ModalFusion"])
        self.GraphConstruct.load_state_dict(states["GraphConstruct"])
        self.MessagePassing.load_state_dict(states["MessagePassing"])
        if self.pair_controller is not None and states["pair_controller"] is not None:
            self.pair_controller.load_state_dict(states["pair_controller"])
        self.current_pair_weights = states["pair_weights"].to(self.dev).float().clone()
        self.current_pair_scores = states["pair_scores"].to(self.dev).float().clone()

    def _audit_weights(self):
        if self.align_strategy == "none":
            return torch.full((6,), float("nan"), device=self.dev)
        return self.current_pair_weights.detach().clone()

    @torch.no_grad()
    def best_metrics(self):
        prob, _, modal_tokens = self.forward_all(return_modal_tokens=True)
        test_metrics = evaluate_prob(prob, self.targ, self.tst_idx)
        test_tokens = modal_tokens[self.tst_idx]
        test_labels = self.targ[self.tst_idx]
        pair_metrics = compute_pair_alignment_metrics(test_tokens, test_labels)
        raw_alignment = compute_alignment_metrics(test_tokens, test_labels)

        pair_weights = self._audit_weights()
        pair_scores = self.current_pair_scores.detach().clone()
        finite_weights = torch.isfinite(pair_weights)
        if bool(finite_weights.all()):
            pair_weight_std = float(pair_weights.std(unbiased=False))
            pair_weight_entropy = normalized_weight_entropy(pair_weights)
            top1_id = int(torch.argmax(pair_weights).item())
            top3_ids = torch.topk(pair_weights, k=3).indices.tolist()
        else:
            pair_weight_std = float("nan")
            pair_weight_entropy = float("nan")
            top1_id = -1
            top3_ids = []

        if self.align_strategy == "none":
            final_align = 0.0
        elif self.align_strategy == "uniform":
            train_tokens = modal_tokens[self.trn_idx]
            train_labels = self.targ[self.trn_idx]
            final_align, _ = compute_alignment_loss(
                train_tokens,
                train_labels,
                mode="hybrid",
                temperature=self.align_temperature,
            )
            final_align = float(final_align)
        else:
            train_tokens = modal_tokens[self.trn_idx]
            train_labels = self.targ[self.trn_idx]
            pair_losses, _ = compute_pairwise_hybrid_losses(
                train_tokens, train_labels, temperature=self.align_temperature
            )
            final_align = float(weighted_pair_alignment_loss(pair_losses, pair_weights))
        final_cls = ClsLoss(prob, self.targ, self.trn_idx, self.weight)
        score_values = pair_scores.detach().cpu().tolist()
        weight_values = pair_weights.detach().cpu().tolist()
        return {
            "test_metrics": test_metrics,
            "modal_tokens": modal_tokens,
            "raw_alignment": raw_alignment,
            "pair_metrics": pair_metrics,
            "pair_scores": score_values,
            "pair_weights": weight_values,
            "pair_weight_std": pair_weight_std,
            "pair_weight_entropy": pair_weight_entropy,
            "top1_pair": top1_id,
            "top3_pairs": top3_ids,
            "final_cls_loss": float(final_cls),
            "final_align_loss": final_align,
            "probs": prob.exp().detach().cpu().numpy(),
        }
