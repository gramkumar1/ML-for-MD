import torch


def radius_graph_pure(pos: torch.Tensor, r: float, loop: bool = False) -> torch.Tensor:
    """
    Pure-PyTorch radius graph (no torch-cluster dependency).

    Builds COO edge_index [2, E] for all pairs (i, j) where ||pos_i - pos_j|| < r.
    O(N²) — fine for small molecules like ethanol (9 atoms).
    """
    # Pairwise squared distances via broadcasting
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)   # [N, N, 3]
    dist2 = (diff ** 2).sum(dim=-1)               # [N, N]

    mask = dist2 < r ** 2
    if not loop:
        mask.fill_diagonal_(False)

    edge_index = mask.nonzero(as_tuple=False).t().contiguous()  # [2, E]
    return edge_index
