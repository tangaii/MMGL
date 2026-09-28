import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSharedPrivateGate(nn.Module):
    """Decompose tokens for regularization while preserving the MMGL path."""

    def __init__(self, modal_num, token_dim, hidden_dim=None, target_shared_ratio=0.50):
        super().__init__()
        self.modal_num = modal_num
        self.token_dim = token_dim
        self.hidden_dim = token_dim if hidden_dim is None else hidden_dim
        self.target_shared_ratio = target_shared_ratio
        self.gates = nn.ModuleList()
        for _ in range(modal_num):
            gate = nn.Sequential(
                nn.Linear(token_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, token_dim),
            )
            nn.init.xavier_uniform_(gate[0].weight)
            nn.init.zeros_(gate[0].bias)
            nn.init.zeros_(gate[2].weight)
            nn.init.zeros_(gate[2].bias)
            self.gates.append(gate)

    def forward(self, tokens):
        if tokens.dim() != 3:
            raise ValueError("tokens must be [B,M,D]")
        batch_size, modal_num, token_dim = tokens.shape
        if modal_num != self.modal_num:
            raise ValueError("modal count mismatch")
        if token_dim != self.token_dim:
            raise ValueError("token dim mismatch")
        gate_values = [torch.sigmoid(self.gates[m](tokens[:, m, :])) for m in range(modal_num)]
        gate_values = torch.stack(gate_values, dim=1)
        shared = gate_values * tokens
        private = (1.0 - gate_values) * tokens
        return shared, private, gate_values


def shared_private_orthogonality(shared, private, eps=1e-8):
    shared_norm = F.normalize(shared, p=2, dim=-1, eps=eps)
    private_norm = F.normalize(private, p=2, dim=-1, eps=eps)
    cosine = (shared_norm * private_norm).sum(dim=-1)
    return cosine.pow(2).mean()


def gate_balance_loss(gate_values, target_ratio=0.50):
    modality_mean = gate_values.mean(dim=(0, 2))
    target = torch.full_like(modality_mean, float(target_ratio))
    return (modality_mean - target).pow(2).mean()


@torch.no_grad()
def gate_statistics(gate_values, modality_names):
    stats = {}
    for m, name in enumerate(modality_names):
        values = gate_values[:, m, :]
        stats["gate_{}_mean".format(name)] = float(values.mean().item())
        stats["gate_{}_std".format(name)] = float(values.std().item())
        stats["gate_{}_gt05".format(name)] = float((values > 0.5).float().mean().item())
    stats["gate_global_mean"] = float(gate_values.mean().item())
    stats["gate_global_std"] = float(gate_values.std().item())
    return stats
