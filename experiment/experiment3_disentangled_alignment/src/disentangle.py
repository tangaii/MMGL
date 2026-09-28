import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionMLP(nn.Module):
    """Xavier-initialized projection; neither output layer is zero initialized."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x)


class SharedOnlyProjector(nn.Module):
    """Projection-only control with one common projector for all modalities."""

    def __init__(self, token_dim=36, shared_dim=18, hidden_dim=36):
        super().__init__()
        self.projector = ProjectionMLP(token_dim, hidden_dim, shared_dim)

    def forward(self, tokens):
        batch, modalities, dim = tokens.shape
        shared = self.projector(tokens.reshape(batch * modalities, dim))
        return shared.reshape(batch, modalities, -1)


class SharedPrivateDisentangler(nn.Module):
    """Common shared coordinates, independent private projectors and decoders."""

    def __init__(self, modal_num=4, token_dim=36, shared_dim=18,
                 private_dim=18, hidden_dim=36):
        super().__init__()
        self.modal_num = modal_num
        self.token_dim = token_dim
        self.shared_dim = shared_dim
        self.private_dim = private_dim
        self.shared_projector = ProjectionMLP(token_dim, hidden_dim, shared_dim)
        self.private_projectors = nn.ModuleList([
            ProjectionMLP(token_dim, hidden_dim, private_dim)
            for _ in range(modal_num)
        ])
        self.decoders = nn.ModuleList()
        for _ in range(modal_num):
            decoder = nn.Sequential(
                nn.Linear(shared_dim + private_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, token_dim),
            )
            for module in decoder.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            self.decoders.append(decoder)
        self.private_modality_classifier = nn.Linear(private_dim, modal_num)
        nn.init.xavier_uniform_(self.private_modality_classifier.weight)
        nn.init.zeros_(self.private_modality_classifier.bias)

    def forward(self, tokens):
        if tokens.dim() != 3:
            raise ValueError("Expected tokens [B,M,D]")
        batch, modalities, dim = tokens.shape
        if modalities != self.modal_num:
            raise ValueError("modal_num mismatch")
        if dim != self.token_dim:
            raise ValueError("token_dim mismatch")
        shared = self.shared_projector(tokens.reshape(batch * modalities, dim))
        shared = shared.reshape(batch, modalities, self.shared_dim)
        private_list, reconstruction_list, logits_list = [], [], []
        for modality in range(modalities):
            private_m = self.private_projectors[modality](tokens[:, modality, :])
            reconstruction_m = self.decoders[modality](
                torch.cat([shared[:, modality, :], private_m], dim=-1)
            )
            logits_m = self.private_modality_classifier(private_m)
            private_list.append(private_m)
            reconstruction_list.append(reconstruction_m)
            logits_list.append(logits_m)
        return (
            shared,
            torch.stack(private_list, dim=1),
            torch.stack(reconstruction_list, dim=1),
            torch.stack(logits_list, dim=1),
        )


def normalized_reconstruction_loss(reconstruction, target, eps=1e-4):
    target_detached = target.detach()
    mse = F.mse_loss(reconstruction, target_detached)
    target_variance = target_detached.var(unbiased=False).clamp_min(eps)
    return mse / target_variance


def cross_covariance_loss(shared, private, eps=1e-4):
    batch, modalities, _ = shared.shape
    if private.shape[:2] != (batch, modalities):
        raise ValueError("shared/private shape mismatch")
    losses = []
    for modality in range(modalities):
        s = shared[:, modality, :]
        p = private[:, modality, :]
        s = s - s.mean(dim=0, keepdim=True)
        p = p - p.mean(dim=0, keepdim=True)
        s = s / torch.sqrt(s.var(dim=0, unbiased=False) + eps)
        p = p / torch.sqrt(p.var(dim=0, unbiased=False) + eps)
        cross_cov = s.transpose(0, 1) @ p / float(max(batch - 1, 1))
        losses.append(cross_cov.pow(2).mean())
    return torch.stack(losses).mean()


def private_modality_loss(modality_logits):
    batch, modalities, classes = modality_logits.shape
    if modalities != classes:
        raise ValueError("Expected modality classifier output dimension M")
    targets = torch.arange(modalities, device=modality_logits.device)
    targets = targets.view(1, modalities).expand(batch, modalities).reshape(-1)
    return F.cross_entropy(modality_logits.reshape(batch * modalities, classes), targets)


@torch.no_grad()
def representation_statistics(shared, private, reconstruction, target):
    return {
        "shared_mean_std": float(shared.std(dim=0, unbiased=False).mean()),
        "private_mean_std": float(private.std(dim=0, unbiased=False).mean()),
        "reconstruction_nmse": float(normalized_reconstruction_loss(reconstruction, target)),
        "shared_private_xcov": float(cross_covariance_loss(shared, private)),
    }
