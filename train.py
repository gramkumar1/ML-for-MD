"""
Joint training loop for SchNet on MD17 ethanol.

Loss = rho * MSE(E_pred, E_true) + (1 - rho) * MSE(F_pred, F_true)

Forces are derived analytically via autograd, not predicted by a separate branch,
so the model is guaranteed to produce conservative force fields.
"""

import argparse
import os
import random

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from data.dataset import get_dataset
from models.schnet import SchNet, compute_forces


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)


def split_dataset(dataset, train_frac=0.8, val_frac=0.1):
    n = len(dataset)
    idx = list(range(n))
    random.shuffle(idx)
    t = int(n * train_frac)
    v = int(n * (train_frac + val_frac))
    return [dataset[i] for i in idx[:t]], [dataset[i] for i in idx[t:v]], [dataset[i] for i in idx[v:]]


def energy_force_loss(pred_E, true_E, pred_F, true_F, rho=0.01):
    e_loss = nn.functional.mse_loss(pred_E.squeeze(), true_E.squeeze())
    f_loss = nn.functional.mse_loss(pred_F, true_F)
    return rho * e_loss + (1.0 - rho) * f_loss, e_loss, f_loss


def evaluate(model, loader, device, rho):
    model.eval()
    total, n = 0.0, 0
    f_mae_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pos = batch.pos.detach().requires_grad_(True)

            with torch.enable_grad():
                E_pred = model(batch.z, pos, batch.edge_index, batch.batch)
                F_pred = compute_forces(E_pred, pos, create_graph=False)

            loss, _, _ = energy_force_loss(E_pred, batch.energy, F_pred, batch.forces, rho)
            f_mae = (F_pred - batch.forces).abs().mean().item()
            total += loss.item() * batch.num_graphs
            f_mae_sum += f_mae * batch.num_graphs
            n += batch.num_graphs

    return total / n, f_mae_sum / n


def train(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = get_dataset(max_samples=args.max_samples)
    train_ds, val_ds, test_ds = split_dataset(dataset, train_frac=0.8, val_frac=0.1)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = SchNet(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_rbf=50,
        cutoff=5.0,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5, min_lr=1e-6)

    best_val = float("inf")
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            # pos must track gradients for autograd force computation
            pos = batch.pos.detach().requires_grad_(True)

            optimizer.zero_grad()
            E_pred = model(batch.z, pos, batch.edge_index, batch.batch)
            F_pred = compute_forces(E_pred, pos)

            loss, e_loss, f_loss = energy_force_loss(E_pred, batch.energy, F_pred, batch.forces, args.rho)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            running_loss += loss.item()

        val_loss, val_f_mae = evaluate(model, val_loader, device, args.rho)
        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d} | Train Loss: {running_loss/len(train_loader):.4f} "
                f"| Val Loss: {val_loss:.4f} | Val F-MAE: {val_f_mae:.4f} kcal/mol/Å"
            )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "checkpoints/schnet_best.pt")

    print(f"\nBest validation loss: {best_val:.4f}")
    print("Model saved to checkpoints/schnet_best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SchNet on MD17 ethanol")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--rho", type=float, default=0.01, help="Energy weight in joint loss (low = force-dominated)")
    parser.add_argument("--max_samples", type=int, default=5000)
    args = parser.parse_args()
    train(args)
