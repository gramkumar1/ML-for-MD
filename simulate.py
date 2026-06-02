"""
Run a trained SchNet MD simulation.

Outputs:
  outputs/trajectory.xyz  — XYZ trajectory (open in OVITO, VMD, or ASE)
  outputs/simulation.gif  — animated 3D preview

Usage:
  python simulate.py
  python simulate.py --steps 1000 --dt 0.5 --temp 300 --thermostat 25
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import torch

from data.dataset import get_dataset
from models.schnet import SchNet
from simulation.integrator import run_md

ELEMENT = {
    1: {"color": "#CCCCCC", "size": 80},
    6: {"color": "#404040", "size": 160},
    7: {"color": "#3050F8", "size": 140},
    8: {"color": "#EE1111", "size": 140},
}
BOND_CUTOFF = 1.8  # Å


def _bond_pairs(pos: np.ndarray) -> list[tuple[int, int]]:
    n = len(pos)
    return [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if np.linalg.norm(pos[i] - pos[j]) < BOND_CUTOFF
    ]


def make_gif(z_arr: np.ndarray, positions: list[np.ndarray], path: str, fps: int = 15, frame_skip: int = 1):
    fig = plt.figure(figsize=(6, 6), facecolor="#111111")
    ax = fig.add_subplot(111, projection="3d")

    all_pos = np.concatenate(positions, axis=0)
    pad = 0.8
    lims = [(all_pos[:, k].min() - pad, all_pos[:, k].max() + pad) for k in range(3)]

    colors = [ELEMENT.get(int(z), {"color": "#FFFFFF"})["color"] for z in z_arr]
    sizes  = [ELEMENT.get(int(z), {"size": 100})["size"]          for z in z_arr]
    bonds  = _bond_pairs(positions[0])

    def _draw(fi):
        ax.cla()
        pos = positions[fi]
        for i, j in bonds:
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], [pos[i, 2], pos[j, 2]],
                    color="#888888", lw=1.2)
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                   c=colors, s=sizes, depthshade=True, edgecolors="none")
        ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
        ax.set_facecolor("#111111")
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#333333")
        ax.tick_params(colors="#555555", labelsize=6)
        ax.set_title(f"Step {fi * frame_skip}", color="#AAAAAA", fontsize=9)
        ax.view_init(elev=20, azim=fi * 0.5)

    ani = animation.FuncAnimation(fig, _draw, frames=len(positions), interval=1000 // fps)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ani.save(path, writer=animation.PillowWriter(fps=fps), dpi=120)
    plt.close(fig)
    print(f"GIF saved → {path}  ({len(positions)} frames @ {fps} fps)")


def main():
    parser = argparse.ArgumentParser(description="Run NeuralMD-SchNet simulation")
    parser.add_argument("--steps",      type=int,   default=500)
    parser.add_argument("--dt",         type=float, default=0.5,  help="Time step in fs")
    parser.add_argument("--temp",       type=float, default=300,  help="Temperature in K")
    parser.add_argument("--thermostat", type=int,   default=25,   help="Rescale velocities every N steps (0=NVE)")
    parser.add_argument("--frame_skip", type=int,   default=5,    help="GIF records every N-th frame")
    parser.add_argument("--checkpoint", type=str,   default="checkpoints/schnet_best.pt")
    parser.add_argument("--hidden_dim", type=int,   default=128)
    parser.add_argument("--num_layers", type=int,   default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SchNet(hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"WARNING: {args.checkpoint} not found — using random weights")
    model.eval()

    frame = get_dataset(max_samples=10)[0]
    z, pos0 = frame.z, frame.pos
    print(f"Molecule: {len(z)} atoms | {args.steps} steps | dt={args.dt} fs | T={args.temp} K")

    os.makedirs("outputs", exist_ok=True)
    history = run_md(
        model=model, z=z, pos0=pos0,
        n_steps=args.steps, dt=args.dt, temperature=args.temp,
        thermostat_interval=args.thermostat,
        output_xyz="outputs/trajectory.xyz",
        device=device,
    )
    print("XYZ trajectory saved → outputs/trajectory.xyz")

    E_tot = np.array(history["E_tot"])
    print(f"\nEnergy conservation — std: {E_tot.std():.4f}  max drift: {np.abs(E_tot - E_tot[0]).max():.4f} kcal/mol")

    frames = history["pos"][::args.frame_skip]
    print(f"Rendering GIF ({len(frames)} frames)...")
    make_gif(history["z"], frames, "outputs/simulation.gif", fps=15, frame_skip=args.frame_skip)

    print("\nOutputs:")
    print("  outputs/trajectory.xyz  — open in OVITO / VMD / ASE")
    print("  outputs/simulation.gif  — animated preview")


if __name__ == "__main__":
    main()
