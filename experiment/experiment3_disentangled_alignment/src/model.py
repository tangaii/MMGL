import copy
import hashlib

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
from disentangle import (
    SharedOnlyProjector,
    SharedPrivateDisentangler,
    cross_covariance_loss,
    normalized_reconstruction_loss,
    private_modality_loss,
    representation_statistics,
)
from probe import modality_probe_accuracy
from utils import ClsLoss, GraphConstructLoss, my_weight_init, normalize_adj, set_rng_seed


def module_fingerprint(modules):
    digest = hashlib.sha256()
    for module in modules:
        for name, value in module.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


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
        self.lambda_rec = hyperpm.lambda_rec
        self.lambda_xcov = hyperpm.lambda_xcov
        self.lambda_mod = hyperpm.lambda_mod
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

        self.base_init_sha256 = module_fingerprint([
            self.ModalFusion, self.GraphConstruct, self.MessagePassing
        ])
        self.SharedProjector = None
        self.Disentangler = None
        self.shared_init_sha256 = ""
        if self.align_strategy in ("shared_only", "disentangled"):
            set_rng_seed(hyperpm.aux_seed)
            if self.align_strategy == "shared_only":
                self.SharedProjector = SharedOnlyProjector(
                    token_dim=36, shared_dim=18, hidden_dim=36
                ).to(self.dev)
                self.shared_init_sha256 = module_fingerprint([self.SharedProjector.projector])
            else:
                self.Disentangler = SharedPrivateDisentangler(
                    modal_num=4, token_dim=36, shared_dim=18,
                    private_dim=18, hidden_dim=36,
                ).to(self.dev)
                self.shared_init_sha256 = module_fingerprint([self.Disentangler.shared_projector])

        auxiliary = self.SharedProjector if self.SharedProjector is not None else self.Disentangler
        if auxiliary is None:
            self.optimizer_MF = torch.optim.Adam(
                self.ModalFusion.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg
            )
        else:
            self.optimizer_MF = torch.optim.Adam([
                {"params": self.ModalFusion.parameters(), "lr": hyperpm.lr, "weight_decay": hyperpm.reg},
                {"params": auxiliary.parameters(), "lr": hyperpm.lr, "weight_decay": 1e-4},
            ])
        self.optimizer_GC = torch.optim.Adam(
            self.GraphConstruct.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg
        )
        self.optimizer_MP = torch.optim.Adam(
            self.MessagePassing.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg
        )
        self.last_losses = {
            "cls": 0.0,
            "align": 0.0,
            "rec": 0.0,
            "xcov": 0.0,
            "modality": 0.0,
            "graph": 0.0,
            "lambda_align": 0.0,
            "lambda_rec": 0.0,
            "lambda_xcov": 0.0,
            "lambda_mod": 0.0,
        }
        self.token_shape = None

    def _warmup_scale(self, epoch):
        if self.align_warmup <= 0:
            return 1.0
        return min(1.0, float(epoch + 1) / float(self.align_warmup))

    def _alignment_coefficients(self, epoch):
        if self.align_strategy == "none":
            return 0.0, 0.0, 0.0, 0.0
        scale = self._warmup_scale(epoch)
        lambda_align = self.align_lambda * scale
        if self.align_strategy == "disentangled":
            return lambda_align, self.lambda_rec * scale, self.lambda_xcov * scale, self.lambda_mod * scale
        return lambda_align, 0.0, 0.0, 0.0

    def _regularization_terms(self, train_tokens, train_labels):
        zero = train_tokens.sum() * 0.0
        losses = {"align": zero, "rec": zero, "xcov": zero, "modality": zero}
        if self.align_strategy == "none":
            return losses, None

        if self.align_strategy == "global":
            align_loss, parts = compute_alignment_loss(
                train_tokens,
                train_labels,
                mode=self.align_mode,
                temperature=self.align_temperature,
            )
            losses["align"] = align_loss
            return losses, parts

        if self.align_strategy == "shared_only":
            shared = self.SharedProjector(train_tokens)
        else:
            shared, private, reconstruction, modality_logits = self.Disentangler(train_tokens)
            losses["rec"] = normalized_reconstruction_loss(reconstruction, train_tokens)
            losses["xcov"] = cross_covariance_loss(shared, private)
            losses["modality"] = private_modality_loss(modality_logits)
        losses["align"], parts = compute_alignment_loss(
            shared, train_labels, mode="hybrid", temperature=self.align_temperature
        )
        return losses, parts

    def _set_auxiliary_training(self, training):
        for module in (self.SharedProjector, self.Disentangler):
            if module is not None:
                module.train(training)

    def run_epoch(self, mode="simple-2", epoch=0):
        if mode != "simple-2":
            raise ValueError("This runner only supports simple-2")

        # Step A: original prediction loss plus training-subject regularizers.
        self.ModalFusion.train()
        self._set_auxiliary_training(True)
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
        auxiliary_losses, parts = self._regularization_terms(train_tokens, train_labels)
        align_loss = auxiliary_losses["align"]
        rec_loss = auxiliary_losses["rec"]
        xcov_loss = auxiliary_losses["xcov"]
        modality_loss = auxiliary_losses["modality"]
        lambda_align, lambda_rec, lambda_xcov, lambda_mod = self._alignment_coefficients(epoch)
        total_mf_loss = (
            cls_loss
            + lambda_align * align_loss
            + lambda_rec * rec_loss
            + lambda_xcov * xcov_loss
            + lambda_mod * modality_loss
        )
        total_mf_loss.backward()
        self.optimizer_MF.step()

        # Step B: original MMGL graph stage with detached 52-D prediction features.
        self.ModalFusion.eval()
        self._set_auxiliary_training(False)
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
            "rec": float(rec_loss.detach()),
            "xcov": float(xcov_loss.detach()),
            "modality": float(modality_loss.detach()),
            "graph": float(graph_loss.detach()),
            "lambda_align": float(lambda_align),
            "lambda_rec": float(lambda_rec),
            "lambda_xcov": float(lambda_xcov),
            "lambda_mod": float(lambda_mod),
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
        self._set_auxiliary_training(False)
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
        auxiliary = self.SharedProjector if self.SharedProjector is not None else self.Disentangler
        if auxiliary is not None:
            states.append(copy.deepcopy(auxiliary.state_dict()))
        return states

    def load_state_dicts(self, states):
        self.ModalFusion.load_state_dict(states[0])
        self.GraphConstruct.load_state_dict(states[1])
        self.MessagePassing.load_state_dict(states[2])
        auxiliary = self.SharedProjector if self.SharedProjector is not None else self.Disentangler
        if auxiliary is not None:
            auxiliary.load_state_dict(states[3])

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
        train_tokens = modal_tokens[self.trn_idx]
        train_labels = self.targ[self.trn_idx]
        train_array = train_tokens.detach().cpu().numpy()
        test_array = test_tokens.detach().cpu().numpy()
        audit = {
            "raw_modality_probe_acc": modality_probe_accuracy(train_array, test_array, self.hyperpm.aux_seed),
            "shared_modality_probe_acc": float("nan"),
            "private_modality_probe_acc": float("nan"),
            "private_shared_probe_gap": float("nan"),
            "shared_mean_std": float("nan"),
            "private_mean_std": float("nan"),
            "reconstruction_nmse": float("nan"),
            "shared_private_xcov": float("nan"),
        }
        if self.align_strategy == "shared_only":
            shared_train = self.SharedProjector(train_tokens)
            shared_test = self.SharedProjector(test_tokens)
            audit["shared_mean_std"] = float(shared_test.std(dim=0, unbiased=False).mean())
        elif self.align_strategy == "disentangled":
            shared_train, private_train, _, _ = self.Disentangler(train_tokens)
            shared_test, private_test, reconstruction_test, _ = self.Disentangler(test_tokens)
            audit.update(representation_statistics(shared_test, private_test, reconstruction_test, test_tokens))
            audit["private_modality_probe_acc"] = modality_probe_accuracy(
                private_train.cpu().numpy(), private_test.cpu().numpy(), self.hyperpm.aux_seed
            )
        if self.align_strategy in ("shared_only", "disentangled"):
            shared_alignment = compute_alignment_metrics(shared_test, test_labels)
            audit["shared_modality_probe_acc"] = modality_probe_accuracy(
                shared_train.cpu().numpy(), shared_test.cpu().numpy(), self.hyperpm.aux_seed
            )
        if self.align_strategy == "disentangled":
            audit["private_shared_probe_gap"] = audit["private_modality_probe_acc"] - audit["shared_modality_probe_acc"]

        final_losses, _ = self._regularization_terms(train_tokens, train_labels)
        final_cls = ClsLoss(prob, self.targ, self.trn_idx, self.weight)
        return {
            "test_metrics": test_metrics,
            "train_metrics": train_metrics,
            "modal_tokens": modal_tokens,
            "raw_alignment": raw_alignment,
            "shared_alignment": shared_alignment,
            "representation_audit": audit,
            "final_cls_loss": float(final_cls.detach()),
            "final_align_loss": float(final_losses["align"]),
            "final_rec_loss": float(final_losses["rec"]),
            "final_xcov_loss": float(final_losses["xcov"]),
            "final_modality_loss": float(final_losses["modality"]),
            "probs": prob.exp().detach().cpu().numpy(),
        }
