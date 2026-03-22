"""
Vectorized heterogeneous graph construction for the BlueROV2 PIGNN.

Graph topology:
    Node types:  hull (1), thruster (8), hydrodynamic (1), buoyancy (1)
    Edge types:  thruster→hull, hydrodynamic→hull, buoyancy→hull

This version pre-computes all static tensors once and builds batch graphs
entirely with tensor operations — no Python loops over thrusters.
"""

import torch
import numpy as np
from torch_geometric.data import HeteroData, Batch

from src.parameters import (
    m, X_ud, Y_vd, Z_wd, I_zz, N_rd,
    X_u, X_uc, Y_v, Y_vc, Z_w, Z_wc,
    N_r, N_rc, g, F_bouy, THRUSTER_CONFIG,
)

NUM_THRUSTERS = 8


# ===================================================================
# Pre-computed static tensors (created once at import time)
# ===================================================================

def _build_allocation_matrix():
    B = np.zeros((4, NUM_THRUSTERS), dtype=np.float32)
    for i in range(NUM_THRUSTERS):
        pos  = np.array(THRUSTER_CONFIG[i]["position"],   dtype=np.float32)
        odir = np.array(THRUSTER_CONFIG[i]["orientation"], dtype=np.float32)
        B[0, i] = odir[0]
        B[1, i] = odir[1]
        B[2, i] = odir[2]
        B[3, i] = pos[0] * odir[1] - pos[1] * odir[0]
    return B

_B_alloc = _build_allocation_matrix()
_B_pinv  = np.linalg.pinv(_B_alloc).astype(np.float32)

# Thruster positions (8, 3) and orientations (8, 3) — static geometry
_THRUSTER_POS = torch.tensor(
    [THRUSTER_CONFIG[i]["position"]    for i in range(NUM_THRUSTERS)],
    dtype=torch.float32,
)
_THRUSTER_DIR = torch.tensor(
    [THRUSTER_CONFIG[i]["orientation"] for i in range(NUM_THRUSTERS)],
    dtype=torch.float32,
)

# Static node features
_HYDRO_NODE = torch.tensor(
    [[X_u + X_uc, Y_v + Y_vc, Z_w + Z_wc, N_r + N_rc]],
    dtype=torch.float32,
)
_BUOY_NODE = torch.tensor(
    [[F_bouy, 0.0, 0.0, 0.0, 0.0, -0.1]],
    dtype=torch.float32,
)

# Static edge features
_HYDRO_EDGE = torch.tensor(
    [[X_u, Y_v, Z_w, N_r, X_ud, Y_vd, Z_wd, N_rd]],
    dtype=torch.float32,
)
_BUOY_EDGE = torch.tensor(
    [[F_bouy, 0.0, 0.0, -0.1]],
    dtype=torch.float32,
)

# Single-graph edge indices (never change)
_THRUSTER_EDGE_INDEX = torch.stack([
    torch.arange(NUM_THRUSTERS, dtype=torch.long),
    torch.zeros(NUM_THRUSTERS, dtype=torch.long),
])
_SINGLE_EDGE_INDEX = torch.tensor([[0], [0]], dtype=torch.long)


# ===================================================================
# Thruster allocation (vectorized)
# ===================================================================

def allocate_thrusts(tau: torch.Tensor) -> torch.Tensor:
    """Map generalised forces τ (B, 4) → individual thruster forces f (B, 8)."""
    B_pinv = torch.tensor(_B_pinv, dtype=tau.dtype, device=tau.device)
    return tau @ B_pinv.T


# ===================================================================
# Per-device static tensor cache
# ===================================================================

_device_cache: dict = {}

def _get_static(device: torch.device):
    """Return all static tensors on `device`, cached after first call."""
    key = str(device)
    if key not in _device_cache:
        _device_cache[key] = {
            "pos":         _THRUSTER_POS.to(device),       # (8, 3)
            "dir":         _THRUSTER_DIR.to(device),       # (8, 3)
            "hydro_node":  _HYDRO_NODE.to(device),         # (1, 4)
            "buoy_node":   _BUOY_NODE.to(device),          # (1, 6)
            "hydro_edge":  _HYDRO_EDGE.to(device),         # (1, 8)
            "buoy_edge":   _BUOY_EDGE.to(device),          # (1, 4)
            "thruster_ei": _THRUSTER_EDGE_INDEX.to(device), # (2, 8)
            "single_ei":   _SINGLE_EDGE_INDEX.to(device),   # (2, 1)
        }
    return _device_cache[key]


# ===================================================================
# Single-graph builder (vectorized — no thruster loops)
# ===================================================================

def build_graph(
    hull_state: torch.Tensor,
    tau: torch.Tensor,
    device: torch.device = None,
) -> HeteroData:
    """
    Build one HeteroData graph.

    Parameters
    ----------
    hull_state : (9,)
    tau        : (4,)
    device     : target device (defaults to hull_state.device)
    """
    if device is None:
        device = hull_state.device
    S = _get_static(device)

    data = HeteroData()

    # ---- Node features ----
    data["hull"].x = hull_state.unsqueeze(0)                               # (1, 9)

    f_ind = allocate_thrusts(tau.unsqueeze(0)).squeeze(0)                   # (8,)
    f_col = f_ind.unsqueeze(1)                                             # (8, 1)
    data["thruster"].x = torch.cat([f_col, f_col, S["pos"], S["dir"]], 1)  # (8, 8)

    data["hydrodynamic"].x = S["hydro_node"]                               # (1, 4)
    data["buoyancy"].x     = S["buoy_node"]                                # (1, 6)

    # ---- Edge indices & features ----
    data["thruster", "forces", "hull"].edge_index = S["thruster_ei"]
    thrust_vecs = f_ind.unsqueeze(1) * S["dir"]                            # (8, 3)
    eff = torch.ones(NUM_THRUSTERS, 1, device=device)
    data["thruster", "forces", "hull"].edge_attr = torch.cat(
        [thrust_vecs, S["pos"], eff], dim=1,
    )                                                                      # (8, 7)

    data["hydrodynamic", "drag", "hull"].edge_index = S["single_ei"]
    data["hydrodynamic", "drag", "hull"].edge_attr  = S["hydro_edge"]

    data["buoyancy", "restoring", "hull"].edge_index = S["single_ei"]
    data["buoyancy", "restoring", "hull"].edge_attr  = S["buoy_edge"]

    return data


# ===================================================================
# Batch builder (thruster allocation vectorized across batch)
# ===================================================================

def build_graph_batch(
    hull_states: torch.Tensor,
    taus: torch.Tensor,
) -> "list[HeteroData]":
    """
    Build a list of HeteroData graphs for a batch.

    Thruster allocation is done once for the full batch.

    Parameters
    ----------
    hull_states : (B, 9)
    taus        : (B, 4)
    """
    device = hull_states.device
    S = _get_static(device)
    B_sz = hull_states.shape[0]

    f_all = allocate_thrusts(taus)                                         # (B, 8)

    graphs = []
    for b in range(B_sz):
        data = HeteroData()

        data["hull"].x = hull_states[b].unsqueeze(0)

        f_col = f_all[b].unsqueeze(1)                                      # (8, 1)
        data["thruster"].x = torch.cat([f_col, f_col, S["pos"], S["dir"]], 1)
        data["hydrodynamic"].x = S["hydro_node"]
        data["buoyancy"].x    = S["buoy_node"]

        data["thruster", "forces", "hull"].edge_index = S["thruster_ei"]
        thrust_vecs = f_all[b].unsqueeze(1) * S["dir"]
        eff = torch.ones(NUM_THRUSTERS, 1, device=device)
        data["thruster", "forces", "hull"].edge_attr = torch.cat(
            [thrust_vecs, S["pos"], eff], dim=1,
        )

        data["hydrodynamic", "drag", "hull"].edge_index = S["single_ei"]
        data["hydrodynamic", "drag", "hull"].edge_attr  = S["hydro_edge"]
        data["buoyancy", "restoring", "hull"].edge_index = S["single_ei"]
        data["buoyancy", "restoring", "hull"].edge_attr  = S["buoy_edge"]

        graphs.append(data)

    return graphs
