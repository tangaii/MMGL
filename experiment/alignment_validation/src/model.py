import copy
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from alignment import compute_alignment_loss
from network import GAT, GCN, GraphLearn, VLTransformer, VLTransformer_Gate
from utils import ClsLoss, GraphConstructLoss, my_weight_init, normalize_adj, one_hot


def evaluate_metrics(prob, labels, idx):
    """Evaluate binary ASD/NC metrics from log probabilities."""
    log_probs = prob[idx]
    probs = log_probs.exp()
    y_true = labels[idx].detach().cpu().numpy().astype(np.int64)
    score = probs[:, 1].detach().cpu().numpy()
    y_pred = probs.argmax(dim=1).detach().cpu().numpy().astype(np.int64)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = float(tp / (tp + fn)) if tp + fn else 0.0
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    true_one_hot = np.eye(2, dtype=np.float64)[y_true]
    pred_one_hot = np.eye(2, dtype=np.float64)[y_pred]

    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "prob_auc": float(roc_auc_score(y_true, score)),
        "legacy_auc": float(roc_auc_score(true_one_hot, pred_one_hot)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


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
    ):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hyperpm = hyperpm
        self.GC_mode = hyperpm.GC_mode
        self.MP_mode = hyperpm.MP_mode
        self.MF_mode = hyperpm.MF_mode
        self.d_v = hyperpm.n_hidden
        self.modal_num = hyperpm.nmodal
        self.n_class = hyperpm.nclass
        self.dropout = hyperpm.dropout
        self.alpha = hyperpm.alpha
        self.n_head = hyperpm.n_head
        self.th = hyperpm.th
        self.align_mode = hyperpm.align_mode
        self.align_lambda = hyperpm.align_lambda
        self.align_temperature = hyperpm.align_temperature
        self.align_warmup = hyperpm.align_warmup

        self.feat = torch.from_numpy(feat).float().to(self.dev)
        self.targ = torch.from_numpy(label).long().to(self.dev)
        self.trn_idx = torch.as_tensor(train_index, dtype=torch.long, device=self.dev)
        self.val_idx = torch.as_tensor(val_index, dtype=torch.long, device=self.dev)
        self.tst_idx = torch.as_tensor(test_index, dtype=torch.long, device=self.dev)

        train_labels = self.targ[self.trn_idx].detach().cpu().numpy()
        counter = Counter(train_labels)
        weight = len(train_labels) / np.array(list(counter.values()), dtype=np.float64) / self.n_class
        self.weight = torch.from_numpy(weight).float().to(self.dev)

        self.out_dim = self.d_v * self.n_head + self.modal_num ** 2
        if self.MF_mode == "sum":
            self.ModalFusion = VLTransformer_Gate(input_data_dims, hyperpm).to(self.dev)
        else:
            self.ModalFusion = VLTransformer(input_data_dims, hyperpm).to(self.dev)
        self.GraphConstruct = GraphLearn(self.out_dim, th=self.th, mode=self.GC_mode).to(self.dev)
        if self.MP_mode == "GCN":
            self.MessagePassing = GCN(self.out_dim, self.out_dim // 2, self.n_class, self.dropout).to(self.dev)
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

        self.optimizer_MF = torch.optim.Adam(
            self.ModalFusion.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg
        )
        self.optimizer_GC = torch.optim.Adam(
            self.GraphConstruct.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg
        )
        self.optimizer_MP = torch.optim.Adam(
            self.MessagePassing.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg
        )
        self.ModalFusion.apply(my_weight_init)
        self.last_losses = {"cls": 0.0, "align": 0.0, "graph": 0.0, "lambda": 0.0}
        self.token_shape = None

    def _alignment_scale(self, epoch):
        if self.align_mode == "none":
            return 0.0
        if self.align_warmup > 0:
            return self.align_lambda * min(1.0, float(epoch + 1) / float(self.align_warmup))
        return self.align_lambda

    def run_epoch(self, mode="simple-2", epoch=0):
        if mode != "simple-2":
            raise ValueError("This validation runner only supports simple-2")

        # Step A: update ModalFusion with train labels and train-only alignment.
        self.ModalFusion.train()
        self.GraphConstruct.eval()
        self.MessagePassing.eval()
        self.optimizer_MF.zero_grad()
        self.optimizer_GC.zero_grad()
        self.optimizer_MP.zero_grad()

        prob, _, _, modal_tokens = self.ModalFusion(self.feat, return_modal_tokens=True)
        if modal_tokens.dim() != 3:
            raise RuntimeError(
                "Expected modal tokens with shape [N, M, D], got {}".format(
                    tuple(modal_tokens.shape)
                )
            )
        batch_size, modal_count, token_dim = modal_tokens.shape
        if batch_size != self.feat.size(0):
            raise RuntimeError(
                "Modal-token batch {} does not match input batch {}".format(
                    batch_size, self.feat.size(0)
                )
            )
        self.token_shape = (int(batch_size), int(modal_count), int(token_dim))
        cls_loss_a = ClsLoss(prob, self.targ, self.trn_idx, self.weight)
        train_tokens = modal_tokens[self.trn_idx]
        train_labels = self.targ[self.trn_idx]
        align_loss, _ = compute_alignment_loss(
            train_tokens,
            train_labels,
            mode=self.align_mode,
            temperature=self.align_temperature,
        )
        lambda_eff = self._alignment_scale(epoch)
        mf_loss = cls_loss_a + lambda_eff * align_loss
        mf_loss.backward()
        self.optimizer_MF.step()

        # Step B: preserve the original MMGL graph training stage.
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
            fusion_feat,
            adj,
            self.hyperpm.theta_smooth,
            self.hyperpm.theta_degree,
            self.hyperpm.theta_sparsity,
        )
        normalized_adj = normalize_adj(adj + torch.eye(adj.size(0), device=self.dev))
        prob, _ = self.MessagePassing(fusion_feat, normalized_adj)
        cls_loss_b = ClsLoss(prob, self.targ, self.trn_idx, self.weight)
        total_loss = cls_loss_b + graph_loss
        total_loss.backward()
        self.optimizer_GC.step()
        self.optimizer_MP.step()

        self.last_losses = {
            "cls": float(cls_loss_b.detach()),
            "align": float(align_loss.detach()),
            "graph": float(graph_loss.detach()),
            "lambda": float(lambda_eff),
        }
        return self.last_losses

    @torch.no_grad()
    def forward_all(self, return_modal_tokens=False):
        self.ModalFusion.eval()
        self.GraphConstruct.eval()
        self.MessagePassing.eval()
        if return_modal_tokens:
            _, fusion_feat, _, modal_tokens = self.ModalFusion(
                self.feat, return_modal_tokens=True
            )
        else:
            modal_tokens = None
            _, fusion_feat, _ = self.ModalFusion(self.feat)
        adj = self.GraphConstruct(fusion_feat)
        normalized_adj = normalize_adj(adj + torch.eye(adj.size(0), device=self.dev))
        prob, _ = self.MessagePassing(fusion_feat, normalized_adj)
        return prob, fusion_feat, modal_tokens

    def state_dicts(self):
        return [
            copy.deepcopy(self.ModalFusion.state_dict()),
            copy.deepcopy(self.GraphConstruct.state_dict()),
            copy.deepcopy(self.MessagePassing.state_dict()),
        ]

    def load_state_dicts(self, states):
        self.ModalFusion.load_state_dict(states[0])
        self.GraphConstruct.load_state_dict(states[1])
        self.MessagePassing.load_state_dict(states[2])

    @torch.no_grad()
    def best_metrics(self):
        prob, fusion_feat, modal_tokens = self.forward_all(return_modal_tokens=True)
        metrics = evaluate_metrics(prob, self.targ, self.tst_idx)
        train_metrics = evaluate_metrics(prob, self.targ, self.trn_idx)
        train_tokens = modal_tokens[self.trn_idx]
        train_labels = self.targ[self.trn_idx]
        final_align_loss, _ = compute_alignment_loss(
            train_tokens,
            train_labels,
            mode=self.align_mode,
            temperature=self.align_temperature,
        )
        adj = self.GraphConstruct(fusion_feat)
        normalized_adj = normalize_adj(adj + torch.eye(adj.size(0), device=self.dev))
        final_prob, _ = self.MessagePassing(fusion_feat, normalized_adj)
        final_cls_loss = ClsLoss(final_prob, self.targ, self.trn_idx, self.weight)
        return metrics, train_metrics, modal_tokens, final_cls_loss, final_align_loss
