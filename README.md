# AI for MD

A machine learning molecular dynamics pipeline that replaces classical force fields with a graph neural network. The model (SchNet) learns to predict potential energy from atomic positions, then derives forces analytically. Those forces drive a Velocity Verlet integrator to simulate atomic motion over time.

SchNet maps a molecular graph (atom types + 3D positions) to a scalar potential energy. It operates only on pairwise distances, making it invariant to rotations and translations by construction.
Forces are computed as `F = -dE/dr` via PyTorch autograd. 

Simulation is velocity Verlet integration with a velocity-rescaling thermostat to prevent drift with imperfect forces.

## Usage

**Train**
```bash
python train.py --epochs 100 --batch_size 32 --rho 0.01
```

Flags:
- `--epochs` — number of training epochs (default: 100)
- `--max_samples` — frames to use from MD17 (default: 5000)
- `--hidden_dim` / `--num_layers` — model size (default: 128, 3)

**Simulate**
```bash
python simulate.py --steps 500 --temp 300
```
Loads the trained checkpoint and runs MD. Writes two files to `outputs/`:
- `trajectory.xyz` — full trajectory, readable by OVITO, VMD, or ASE
- `simulation.gif` — animated 3D preview

Flags:
- `--steps` — number of MD steps at 0.5 fs each (default: 500)
- `--temp` — temperature in Kelvin for initial velocity sampling (default: 300)
- `--thermostat` — rescale velocities every N steps (default: 25; set to 0 for NVE)
- `--frame_skip` — record every N-th frame in the GIF (default: 5)

