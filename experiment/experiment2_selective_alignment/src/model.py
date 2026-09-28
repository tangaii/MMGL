import copy

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from alignment import compute_alignment_loss, compute_alignment_metrics
from network import GAT, GCN, GraphLearn, VLTransformer, VLTransformer_Gate
from selective_alignment import (
    SelectiveSharedPrivateGate,
    gate_balance_loss,
    gate_statistics,
    shared_private_orthogonality,
)
from utils import ClsLoss, GraphConstructLoss, my_weight_init, normalize_adj


def _safe_auc(y_true, score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


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
    cm = confusion_matrix(y_asd, pred_asd, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    asd_sensitivity = float(tp / (tp + fn)) if tp + fn else 0.0
    nc_specificity = float(tn / (tn + fp)) if tn + fp else 0.0

    return {
        "acc": float(accuracy_score(y_asd, pred_asd)),
        "balanced_acc": float(balanced_accuracy_score(y_asd, pred_asd)),
        "asd_auc": _safe_auc(y_asd, p_asd),
        "asd_f1": float(f1_score(y_asd, pred_asd, zero_division=0)),
        "asd_sensitivity": asd_sensitivity,
        "nc_specificity": nc_specificity,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


@torch.no_grad()
def evaluate_prob(prob, labels, idx):
    selected = prob[idx].exp()
    labels_np = labels[idx].detach().cpu().numpy()
    probs_np = selected.detach().cpu().numpy()
    pred_np = probs_np.argmax(axis=1)
    return metrics_from_arrays(labels_np, probs_np, pred_np)


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
        self.lambda_orth = hyperpm.lambda_orth
        self.lambda_balance = hyperpm.lambda_balance
        self.target_shared_ratio = hyperpm.target_shared_ratio
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
        weight = len(train_y_np) / (self.n_class * np.maximum(counts, 1.0))
        self.weight = torch.tensor(weight, dtype=torch.float32, device=self.dev)

        self.out_dim = self.d_v * self.n_head + self.modal_num ** 2

        # Keep the base MMGL initialization order fixed across configurations.
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
                self.out_dim,
                self.out_dim // 2,
                self.n_class,
                self.dropout,
                self.alpha,
                nheads=2,
            ).to(self.dev)
        else:
            raise ValueError("Unsupported MP_mode: {}".format(self.MP_mode))

        self.SelectiveAligner = None
        if self.align_strategy == "selective":
            self.SelectiveAligner = SelectiveSharedPrivateGate(
                modal_num=self.modal_num,
                token_dim=self.n_head * self.d_v,
                hidden_dim=self.n_head * self.d_v,
                target_shared_ratio=self.target_shared_ratio,
            ).to(self.dev)

        mf_params = list(self.ModalFusion.parameters())
        if self.SelectiveAligner is not None:
            mf_params += list(self.SelectiveAligner.parameters())
        self.optimizer_MF = torch.optim.Adam(
            mf_params, lr=hyperpm.lr, weight_decay=hyperpm.reg
        )
        self.optimizer_GC = torch.optim.Adam(
            self.GraphConstruct.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg
        )
        self.optimizer_MP = torch.optim.Adam(
            self.MessagePassing.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg
        )
        self.last_losses = {
            "cls": 0.0,
            "align": 0.0,
            "orth": 0.0,
            "balance": 0.0,
            "graph": 0.0,
            "lambda_align": 0.0,
            "lambda_orth": 0.0,
            "lambda_balance": 0.0,
        }
        self.token_shape = None

    def _warmup_scale(self, epoch):
        if self.align_warmup <= 0:
            return 1.0
        return min(1.0, float(epoch + 1) / float(self.align_warmup))

    def _alignment_coefficients(self, epoch):
        if self.align_strategy == "none":
            return 0.0, 0.0, 0.0
        scale = self._warmup_scale(epoch)
        lambda_align = self.align_lambda * scale
        if self.align_strategy == "selective":
            return lambda_align, self.lambda_orth * scale, self.lambda_balance * scale
        return lambda_align, 0.0, 0.0

    def _regularization_terms(self, train_tokens, train_labels):
        zero = train_tokens.sum() * 0.0
        if self.align_strategy == "none":
            return zero, zero, zero, None, None, None

        if self.align_strategy == "global":
            align_loss, parts = compute_alignment_loss(
                train_tokens,
                train_labels,
                mode=self.align_mode,
                temperature=self.align_temperature,
            )
            return align_loss, zero, zero, None, None, parts

        shared, private, gates = self.SelectiveAligner(train_tokens)
        align_loss, parts = compute_alignment_loss(
            shared,
            train_labels,
            mode="hybrid",
            temperature=self.align_temperature,
        )
        orth_loss = shared_private_orthogonality(shared, private)
        balance_loss = gate_balance_loss(gates, self.target_shared_ratio)
        return align_loss, orth_loss, balance_loss, shared, private, parts

    def run_epoch(self, mode="simple-2", epoch=0):
        if mode != "simple-2":
            raise ValueError("This runner only supports simple-2")

        # Step A: update ModalFusion and, for selective runs, the gate only.
        self.ModalFusion.train()
        if self.SelectiveAligner is not None:
            self.SelectiveAligner.train()
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
        if batch_size != self.feat.size(0):
            raise RuntimeError("Modal token batch does not match input batch")
        self.token_shape = (int(batch_size), int(modal_count), int(token_dim))
        cls_loss = ClsLoss(prob, self.targ, self.trn_idx, self.weight)
        train_tokens = modal_tokens[self.trn_idx]
        train_labels = self.targ[self.trn_idx]
        align_loss, orth_loss, balance_loss, _, _, parts = self._regularization_terms(
            train_tokens, train_labels
        )
        lambda_align, lambda_orth, lambda_balance = self._alignment_coefficients(epoch)
        total_mf_loss = (
            cls_loss
            + lambda_align * align_loss
            + lambda_orth * orth_loss
            + lambda_balance * balance_loss
        )
        total_mf_loss.backward()
        self.optimizer_MF.step()

        # Step B is the original MMGL graph stage. The selective module is unused.
        self.ModalFusion.eval()
        if self.SelectiveAligner is not None:
            self.SelectiveAligner.eval()
        self.GraphConstruct.train()
        self.MessagePassing.train()
        self.optimizer_MF.zero_grad()
        self.optimizer_GC.zero_grad()
        self.optimizer_MP.zero_grad()

        _, embedding, _ = self.ModalFusion(self.feat)
        fusion_feat = embedding.detach()
        adj = self.GraphConstruct(fusion_feat)
        graph_loss = GraphConstructLoss(
            fusion_feat,
            adj,
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
            "orth": float(orth_loss.detach()),
            "balance": float(balance_loss.detach()),
            "graph": float(graph_loss.detach()),
            "lambda_align": float(lambda_align),
            "lambda_orth": float(lambda_orth),
            "lambda_balance": float(lambda_balance),
        }
        if parts is not None:
            self.last_losses.update(
                {"align_nce": parts["nce"], "align_cmmd": parts["cmmd"]}
            )
        return self.last_losses

    @torch.no_grad()
    def forward_all(self, return_modal_tokens=True):
        self.ModalFusion.eval()
        self.GraphConstruct.eval()
        self.MessagePassing.eval()
        if self.SelectiveAligner is not None:
            self.SelectiveAligner.eval()
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
        states = [
            copy.deepcopy(self.ModalFusion.state_dict()),
            copy.deepcopy(self.GraphConstruct.state_dict()),
            copy.deepcopy(self.MessagePassing.state_dict()),
        ]
        if self.SelectiveAligner is not None:
            states.append(copy.deepcopy(self.SelectiveAligner.state_dict()))
        return states

    def load_state_dicts(self, states):
        self.ModalFusion.load_state_dict(states[0])
        self.GraphConstruct.load_state_dict(states[1])
        self.MessagePassing.load_state_dict(states[2])
        if self.SelectiveAligner is not None:
            self.SelectiveAligner.load_state_dict(states[3])

    @torch.no_grad()
    def best_metrics(self):
        prob, fusion_feat, modal_tokens = self.forward_all(return_modal_tokens=True)
        test_metrics = evaluate_prob(prob, self.targ, self.tst_idx)
        train_metrics = evaluate_prob(prob, self.targ, self.trn_idx)
        test_tokens = modal_tokens[self.tst_idx]
        test_labels = self.targ[self.tst_idx]
        raw_alignment = compute_alignment_metrics(test_tokens, test_labels)

        shared_alignment = {
            "pos_cos": float("nan"),
            "neg_cos": float("nan"),
            "cos_gap": float("nan"),
            "retrieval_r1": float("nan"),
            "cmmd": float("nan"),
        }
        gate_stats = {}
        if self.SelectiveAligner is not None:
            shared_test, _, gate_values = self.SelectiveAligner(test_tokens)
            shared_alignment = compute_alignment_metrics(shared_test, test_labels)
            gate_stats = gate_statistics(gate_values, self.modality_names)

        train_tokens = modal_tokens[self.trn_idx]
        train_labels = self.targ[self.trn_idx]
        final_align, final_orth, final_balance, _, _, _ = self._regularization_terms(
            train_tokens, train_labels
        )
        final_cls = ClsLoss(prob, self.targ, self.trn_idx, self.weight)
        return {
            "test_metrics": test_metrics,
            "train_metrics": train_metrics,
            "modal_tokens": modal_tokens,
            "raw_alignment": raw_alignment,
            "shared_alignment": shared_alignment,
            "gate_stats": gate_stats,
            "final_cls_loss": float(final_cls.detach()),
            "final_align_loss": float(final_align.detach()),
            "final_orth_loss": float(final_orth.detach()),
            "final_balance_loss": float(final_balance.detach()),
            "probs": prob.exp().detach().cpu().numpy(),
        }
