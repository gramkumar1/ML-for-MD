"""
MD17 ethanol dataset loader.

Downloads via torch_geometric on first run, builds radius-cutoff graphs,
subtracts the mean energy so the model learns ~1 kcal/mol residuals rather
than the ~-97,000 kcal/mol absolute offset (which would swamp the force loss).
Forces are derivatives of energy, so they need no adjustment.
"""

import os
import torch
from torch_geometric.datasets import MD17
from utils import radius_graph_pure

CUTOFF = 5.0
_DATA_ROOT = os.path.join(os.path.dirname(__file__), "md17")
_CACHE     = os.path.join(os.path.dirname(__file__), "md17_ethanol_processed.pt")


def load_and_process(max_samples: int = 5000) -> list:
    print("Fetching MD17 ethanol via torch_geometric (downloads on first run)...")
    raw = MD17(root=_DATA_ROOT, name="ethanol")
    n = min(max_samples, len(raw))

    dataset = []
    for i in range(n):
        s = raw[i].clone()
        s.edge_index = radius_graph_pure(s.pos, r=CUTOFF, loop=False)
        s.forces = s.force
        del s.force
        dataset.append(s)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n} frames processed")

    E_mean = torch.stack([d.energy for d in dataset]).mean()
    for d in dataset:
        d.energy = d.energy - E_mean
    print(f"Energy mean subtracted: {E_mean.item():.2f} kcal/mol")

    torch.save(dataset, _CACHE)
    print(f"Cached {len(dataset)} graphs → {_CACHE}")
    return dataset


def get_dataset(max_samples: int = 5000) -> list:
    if os.path.exists(_CACHE):
        return torch.load(_CACHE, weights_only=False)
    return load_and_process(max_samples)
