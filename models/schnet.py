"""
SchNet: Invariant Graph Neural Network for potential energy prediction.

Architecture follows:
  Schütt et al. "SchNet: A continuous-filter convolutional neural network for
  modeling quantum interactions." NeurIPS 2017.

Key invariance property: energy depends only on pairwise distances d_ij = ||r_i - r_j||,
making it invariant to global rotations and translations by construction.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter


class GaussianSmearing(nn.Module):
    """Expands scalar distances into a fixed Radial Basis Function (RBF) feature vector."""

    def __init__(self, d_min: float = 0.0, d_max: float = 5.0, num_rbf: int = 50):
        super().__init__()
        centers = torch.linspace(d_min, d_max, num_rbf)
        self.register_buffer("centers", centers)
        self.gamma = 2.0 / (d_max - d_min) * num_rbf  # controls bandwidth

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        # d: [E] → output: [E, num_rbf]
        return torch.exp(-self.gamma * (d.unsqueeze(-1) - self.centers) ** 2)


class CFConv(MessagePassing):
    """
    Continuous-Filter Convolution layer (the core SchNet message-passing block).

    For each edge (i←j): weight = MLP(RBF(d_ij)), message = weight * h_j.
    Node update: h_i ← h_i + Σ_j message_ij  (followed by a linear + shift-net).
    """

    def __init__(self, hidden_dim: int, num_rbf: int):
        super().__init__(aggr="add")
        self.filter_net = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_rbf: torch.Tensor) -> torch.Tensor:
        # h: [N, hidden_dim], edge_rbf: [E, num_rbf]
        W = self.filter_net(edge_rbf)           # [E, hidden_dim]
        return self.node_net(self.propagate(edge_index, h=h, W=W))

    def message(self, h_j: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        return h_j * W


class SchNet(nn.Module):
    """
    Full SchNet model: atom embeddings → iterative CFConv message passing → pooled energy.

    Returns a scalar energy per graph (batch support via torch_geometric batching).
    Forces are NOT computed here; call compute_forces() after a forward pass.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_rbf: int = 50,
        cutoff: float = 5.0,
        max_z: int = 100,
    ):
        super().__init__()
        self.embedding = nn.Embedding(max_z, hidden_dim)
        self.smearing = GaussianSmearing(d_min=0.0, d_max=cutoff, num_rbf=num_rbf)

        self.conv_layers = nn.ModuleList(
            [CFConv(hidden_dim, num_rbf) for _ in range(num_layers)]
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """
        z:          [N] atomic numbers
        pos:        [N, 3] Cartesian coordinates (must have requires_grad=True for force computation)
        edge_index: [2, E] COO edge connectivity
        batch:      [N] graph assignment index (None → single graph)

        Returns:    [B] predicted potential energies (one per graph in the batch)
        """
        if batch is None:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)

        # --- Node features: learned atomic embeddings ---
        h = self.embedding(z)                               # [N, hidden_dim]

        # --- Edge features: rotationally invariant distances ---
        row, col = edge_index
        d_vec = pos[row] - pos[col]                         # [E, 3]
        d = d_vec.norm(dim=-1, keepdim=False)               # [E]  — scalar, invariant
        edge_rbf = self.smearing(d)                         # [E, num_rbf]

        # --- Message-passing iterations ---
        for conv, ln in zip(self.conv_layers, self.layer_norms):
            h = ln(h + conv(h, edge_index, edge_rbf))      # residual + layer norm

        # --- Per-atom energies → global sum pooling ---
        atom_energies = self.output_net(h).squeeze(-1)      # [N]
        energy = scatter(atom_energies, batch, dim=0, reduce="sum")  # [B]

        return energy


def compute_forces(energy: torch.Tensor, pos: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """
    Compute conservative forces as F_i = -∂E/∂r_i via autograd.

    energy:       [B] or scalar result from SchNet.forward()
    pos:          [N, 3] — must have been created with requires_grad=True
    create_graph: True during training (loss.backward() needs higher-order graph);
                  False during evaluation (no second backward needed, ~2× faster).
    """
    grad = torch.autograd.grad(
        energy.sum(),
        pos,
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    return -grad  # [N, 3]
