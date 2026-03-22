# Physics-Informed Graph Neural Network (PIGNN) for BlueROV2 Dynamics

This project implements a **heterogeneous Physics-Informed Graph Neural Network** to model the 4-DOF dynamics of a BlueROV2 Heavy underwater vehicle.  It extends the [PINC](https://github.com/eivacom/pinc-xyz-yaw) codebase by replacing the feedforward DNN with a GNN whose graph topology encodes the physical subsystem structure of the vehicle.

## Architecture

The heterogeneous graph has **four node types** representing distinct physical subsystems:

| Node Type       | Count | Feature Dim | Description                              |
|-----------------|-------|-------------|------------------------------------------|
| Hull            | 1     | 9           | 4-DOF rigid-body state                   |
| Thruster        | 8     | 8           | One per T200 thruster (Heavy config)     |
| Hydrodynamic    | 1     | 4           | Drag and added-mass effects              |
| Buoyancy        | 1     | 6           | Restoring forces (gravity / buoyancy)    |

**Three directed edge types** encode force interactions:

| Edge Type              | Count | Feature Dim | Physical Meaning                        |
|------------------------|-------|-------------|-----------------------------------------|
| Thruster → Hull        | 8     | 7           | Force/torque from each thruster         |
| Hydrodynamic → Hull    | 1     | 8           | Drag + added-mass coupling              |
| Buoyancy → Hull        | 1     | 4           | Restoring force (primarily heave)       |

### Physics-Informed Loss

The model uses the **Fossen equation** (4-DOF):

**M·ν̇ + C(ν)·ν + D(ν)·ν + g(η) = τ**

as a soft constraint.  The total loss combines:

- **Data loss** — 1-step-ahead MSE
- **Physics-residual loss** — penalises deviations from the Fossen ODE
- **Initial-condition loss** — self-consistency at t=0
- **Rollout loss** — multi-step prediction error

Hard constraints are encoded via the graph topology itself (only thrusters introduce external forces; all edges target the hull).

## Project Structure

```
pignn_project/
├── data/
│   ├── create_data.py        # Trajectory generation via simulator
│   └── data_utility.py       # Dataset class & input generators
├── models/
│   ├── graph_builder.py      # Heterogeneous graph construction
│   ├── pignn.py              # PIGNN model (core architecture)
│   └── model_utility.py      # Loss functions, training loop
├── training/
│   └── train_pignn.py        # Main training script
├── scripts/
│   └── evaluate_model.py     # Rollout evaluation & plotting
├── src/
│   ├── parameters.py         # BlueROV2 physical parameters
│   ├── bluerov.py            # NumPy simulator (Numba-accelerated)
│   └── bluerov_torch.py      # PyTorch differentiable simulator
├── tests/
│   └── test_pignn.py         # Comprehensive test suite
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone and enter the project
cd pignn_project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** PyTorch Geometric requires separate installation.  See https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

## Usage

### 1. Generate Data

```bash
python data/create_data.py
```

This creates `training_set/`, `dev_set/`, `test_set_interp/`, and `test_set_extrap/` directories containing `.pt` files.

### 2. Train the Model

```bash
python training/train_pignn.py
```

Monitor with TensorBoard:

```bash
tensorboard --logdir runs
```

### 3. Evaluate

```bash
python scripts/evaluate_model.py \
    --model_path models_saved/pignn_bluerov2_direct_best_dev_epoch_100 \
    --dataset dev_set \
    --n_trajs 5
```

### 4. Run Tests

```bash
pytest tests/ -v
```

## Governing Equations

The 4-DOF Fossen model (surge, sway, heave, yaw):

- **State:** η = [x, y, z, ψ], ν = [u, v, w, r]
- **Representation:** ψ → (cos ψ, sin ψ) for continuity → 9-dim state
- Roll and pitch are excluded (BlueROV2 is passively stable in these DOFs)

## References

- Fossen, T.I. (2011). *Handbook of Marine Craft Hydrodynamics and Motion Control*
- PINC repository: https://github.com/eivacom/pinc-xyz-yaw
- ConFIG gradient method: https://github.com/tum-pbs/ConFIG

## License

GPL-3.0 (following the upstream PINC repository)
