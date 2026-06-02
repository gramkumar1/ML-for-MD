"""
Velocity Verlet molecular dynamics integrator with velocity-rescaling thermostat.

Physical units:
  Positions:  Angstroms (Å)
  Energies:   kcal/mol
  Forces:     kcal/(mol·Å)
  Masses:     amu
  Time:       femtoseconds (fs)
  Velocities: Å/fs
"""

import os
from typing import Optional

import numpy as np
import torch
from models.schnet import SchNet, compute_forces
from utils import radius_graph_pure

# 1 kcal/(mol·Å) / 1 amu  →  Å/fs²  (verified: 4.184e-4)
FORCE_TO_ACCEL = 4.184e-4

KB = 0.001987  # kcal/(mol·K)

ATOMIC_MASS = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998, 16: 32.06}
ELEMENT_SYM  = {1: "H",  6: "C",    7: "N",    8: "O",    9: "F",    16: "S"}


def get_masses(z: torch.Tensor) -> torch.Tensor:
    return torch.tensor([ATOMIC_MASS[zi.item()] for zi in z], dtype=torch.float32)


def maxwell_boltzmann_velocities(z: torch.Tensor, temperature: float, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    masses = get_masses(z)
    sigma = torch.sqrt(KB * temperature / masses)
    vel = sigma.unsqueeze(-1) * torch.randn(z.size(0), 3)
    vel -= vel.mean(dim=0, keepdim=True)   # remove COM drift
    return vel


def _instantaneous_temperature(vel: torch.Tensor, masses: torch.Tensor) -> float:
    """T = 2 KE / (3 N kB), in Kelvin."""
    KE = 0.5 * (masses.unsqueeze(-1) * vel ** 2).sum().item() / FORCE_TO_ACCEL
    n_dof = 3 * vel.size(0) - 3   # subtract 3 for removed COM
    return 2.0 * KE / (n_dof * KB)


def _rescale_velocities(vel: torch.Tensor, masses: torch.Tensor, target_T: float) -> torch.Tensor:
    """Instantaneously rescale velocities to match target temperature."""
    current_T = _instantaneous_temperature(vel, masses)
    if current_T < 1e-6:
        return vel
    return vel * (target_T / current_T) ** 0.5


def _forward(model, z, pos, cutoff, device):
    pos = pos.to(device).detach().requires_grad_(True)
    edge_index = radius_graph_pure(pos, r=cutoff, loop=False).to(device)
    energy = model(z.to(device), pos, edge_index)
    forces = compute_forces(energy, pos, create_graph=False)
    return energy.detach().cpu(), forces.detach().cpu()


def _kinetic_energy(vel: torch.Tensor, masses: torch.Tensor) -> float:
    return 0.5 * (masses.unsqueeze(-1) * vel ** 2).sum().item() / FORCE_TO_ACCEL


def run_md(
    model: SchNet,
    z: torch.Tensor,
    pos0: torch.Tensor,
    n_steps: int = 1000,
    dt: float = 0.5,
    temperature: float = 300.0,
    thermostat_interval: int = 25,
    cutoff: float = 5.0,
    output_xyz: str = "trajectory.xyz",
    device: Optional[torch.device] = None,
    progress_callback=None,
) -> dict:
    """
    Velocity Verlet MD with a velocity-rescaling thermostat.

    thermostat_interval: rescale velocities every N steps (0 = NVE, no thermostat).
                         With an imperfect model, keep this at 10-50 to prevent explosion.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device).eval()
    masses = get_masses(z)
    inv_m  = (1.0 / masses).unsqueeze(-1)

    pos = pos0.clone().float()
    vel = maxwell_boltzmann_velocities(z, temperature)

    E_pot_t, F = _forward(model, z, pos, cutoff, device)

    history = {"pos": [], "E_pot": [], "E_kin": [], "E_tot": []}

    os.makedirs(os.path.dirname(os.path.abspath(output_xyz)), exist_ok=True)
    xyz_file = open(output_xyz, "w")

    def _write_frame(pos_np, step):
        n = pos_np.shape[0]
        xyz_file.write(f"{n}\nStep {step}\n")
        for zi, (x, y, zz) in zip(z.tolist(), pos_np.tolist()):
            xyz_file.write(f"{ELEMENT_SYM.get(zi, 'X')} {x:.6f} {y:.6f} {zz:.6f}\n")

    for step in range(n_steps):
        accel = F * FORCE_TO_ACCEL * inv_m

        # Verlet position update
        pos = pos + vel * dt + 0.5 * accel * dt ** 2

        E_pot_t, F_new = _forward(model, z, pos, cutoff, device)
        accel_new = F_new * FORCE_TO_ACCEL * inv_m

        # Verlet velocity update
        vel = vel + 0.5 * (accel + accel_new) * dt
        F = F_new

        # Velocity-rescaling thermostat — corrects drift from imperfect forces
        if thermostat_interval > 0 and (step + 1) % thermostat_interval == 0:
            vel = _rescale_velocities(vel, masses, temperature)

        E_kin = _kinetic_energy(vel, masses)
        E_pot = E_pot_t.item()
        E_tot = E_pot + E_kin

        history["pos"].append(pos.numpy().copy())
        history["E_pot"].append(E_pot)
        history["E_kin"].append(E_kin)
        history["E_tot"].append(E_tot)

        _write_frame(pos.numpy(), step)

        if progress_callback is not None:
            progress_callback(step, pos.numpy(), E_pot, E_kin, E_tot)

    xyz_file.close()
    history["z"] = z.numpy()
    return history
