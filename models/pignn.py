"""
Physics-Informed Heterogeneous Graph Neural Network (PIGNN) for BlueROV2.

Optimised version:
    - Uses vectorized batch graph construction (no per-thruster Python loops).
    - Batch thruster allocation computed once per forward pass.
    - Static tensors cached per device.

Architecture unchanged:
    1. Build heterogeneous graph from state & control.
    2. Encode hull node with full input vector (state + control + time).
    3. N rounds of message passing (thruster→hull, hydro→hull, buoy→hull).
    4. Readout hull embedding → state increment Δx̂.
    5. Residual + body→world rotation for (x, y).
"""

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData, Batch

from models.graph_builder import (
    build_graph, build_graph_batch, allocate_thrusts, NUM_THRUSTERS,
)
from src.parameters import THRUSTER_CONFIG


# ---------------------------------------------------------------------------
# Adaptive Softplus
# ---------------------------------------------------------------------------
class AdaptiveSoftplus(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.sp   = nn.Softplus()
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        return torch.reciprocal(self.beta) * self.sp(self.beta * x)


# ---------------------------------------------------------------------------
# MLP helper
# ---------------------------------------------------------------------------
def _mlp(in_dim: int, hidden: int, out_dim: int, n_layers: int = 2):
    layers = []
    layers.append(nn.Linear(in_dim, hidden))
    layers.append(AdaptiveSoftplus())
    layers.append(nn.LayerNorm(hidden))
    for _ in range(n_layers - 2):
        layers.append(nn.Linear(hidden, hidden))
        layers.append(AdaptiveSoftplus())
        layers.append(nn.LayerNorm(hidden))
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Per-edge-type message functions
# ---------------------------------------------------------------------------
class ThrusterToHullConv(nn.Module):
    def __init__(self, thruster_dim, edge_dim, hull_dim, out_channels, hidden=32):
        super().__init__()
        self.mlp = _mlp(thruster_dim + edge_dim + hull_dim, hidden, out_channels)

    def forward(self, x_src, x_dst, edge_attr, edge_index):
        src, dst = edge_index
        msg_in = torch.cat([x_src[src], edge_attr, x_dst[dst]], dim=-1)
        return self.mlp(msg_in)


class HydroToHullConv(nn.Module):
    def __init__(self, hydro_dim, edge_dim, hull_dim, out_channels, hidden=32):
        super().__init__()
        self.mlp = _mlp(hydro_dim + edge_dim + hull_dim, hidden, out_channels)

    def forward(self, x_src, x_dst, edge_attr, edge_index):
        src, dst = edge_index
        msg_in = torch.cat([x_src[src], edge_attr, x_dst[dst]], dim=-1)
        return self.mlp(msg_in)


class BuoyToHullConv(nn.Module):
    def __init__(self, buoy_dim, edge_dim, hull_dim, out_channels, hidden=32):
        super().__init__()
        self.mlp = _mlp(buoy_dim + edge_dim + hull_dim, hidden, out_channels)

    def forward(self, x_src, x_dst, edge_attr, edge_index):
        src, dst = edge_index
        msg_in = torch.cat([x_src[src], edge_attr, x_dst[dst]], dim=-1)
        return self.mlp(msg_in)


# ---------------------------------------------------------------------------
# Heterogeneous message-passing layer
# ---------------------------------------------------------------------------
class PIGNNLayer(nn.Module):
    def __init__(self, node_dims, edge_dims, hidden=32, msg_dim=32):
        super().__init__()
        self.thruster_conv = ThrusterToHullConv(
            node_dims["thruster"], edge_dims[("thruster", "forces", "hull")],
            node_dims["hull"], msg_dim, hidden,
        )
        self.hydro_conv = HydroToHullConv(
            node_dims["hydrodynamic"], edge_dims[("hydrodynamic", "drag", "hull")],
            node_dims["hull"], msg_dim, hidden,
        )
        self.buoy_conv = BuoyToHullConv(
            node_dims["buoyancy"], edge_dims[("buoyancy", "restoring", "hull")],
            node_dims["hull"], msg_dim, hidden,
        )
        self.hull_update = _mlp(
            node_dims["hull"] + 3 * msg_dim, hidden, node_dims["hull"], n_layers=2,
        )

    def forward(self, data: HeteroData) -> HeteroData:
        # Thruster messages — sum-aggregated (force superposition)
        ei_t = data["thruster", "forces", "hull"].edge_index
        ea_t = data["thruster", "forces", "hull"].edge_attr
        msg_t = self.thruster_conv(data["thruster"].x, data["hull"].x, ea_t, ei_t)
        num_hull = data["hull"].x.size(0)
        agg_t = torch.zeros(num_hull, msg_t.size(-1),
                            device=msg_t.device, dtype=msg_t.dtype)
        agg_t.scatter_add_(0, ei_t[1].unsqueeze(-1).expand_as(msg_t), msg_t)

        # Hydrodynamic message
        ei_h = data["hydrodynamic", "drag", "hull"].edge_index
        ea_h = data["hydrodynamic", "drag", "hull"].edge_attr
        agg_h = self.hydro_conv(data["hydrodynamic"].x, data["hull"].x, ea_h, ei_h)

        # Buoyancy message
        ei_b = data["buoyancy", "restoring", "hull"].edge_index
        ea_b = data["buoyancy", "restoring", "hull"].edge_attr
        agg_b = self.buoy_conv(data["buoyancy"].x, data["hull"].x, ea_b, ei_b)

        # Update hull
        hull_in = torch.cat([data["hull"].x, agg_t, agg_h, agg_b], dim=-1)
        data["hull"].x = self.hull_update(hull_in)
        return data


# ---------------------------------------------------------------------------
# Full PIGNN model
# ---------------------------------------------------------------------------
class PIGNN(nn.Module):
    """
    Physics-Informed Graph Neural Network for BlueROV2 dynamics.

    Interface matches the original DNN for drop-in compatibility:
        Input:  Z = [state(9) | control(4) | time(1)]  shape (B, 14)
        Output: x̂(t+dt)                                 shape (B, 9)
    """

    def __init__(
        self,
        N_in:  int = 14,
        N_out: int = 9,
        hidden:     int = 32,
        msg_dim:    int = 32,
        n_mp_layers: int = 2,
    ):
        super().__init__()
        self.N_in  = N_in
        self.N_out = N_out

        hull_enc_dim = hidden
        self.node_dims = {
            "hull": hull_enc_dim, "thruster": 8,
            "hydrodynamic": 4,    "buoyancy": 6,
        }
        self.edge_dims = {
            ("thruster",     "forces",    "hull"): 7,
            ("hydrodynamic", "drag",      "hull"): 8,
            ("buoyancy",     "restoring", "hull"): 4,
        }

        self.hull_encoder = _mlp(N_in, hidden, hull_enc_dim)

        self.mp_layers = nn.ModuleList([
            PIGNNLayer(self.node_dims, self.edge_dims, hidden, msg_dim)
            for _ in range(n_mp_layers)
        ])

        self.readout = _mlp(hull_enc_dim, hidden, N_out, n_layers=3)
        self._init_weights()

    def _init_weights(self):
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        if Z.dim() == 1:
            Z = Z.unsqueeze(0)

        B = Z.shape[0]
        device = Z.device

        # --- Build batch of graphs (vectorized) ---
        states = Z[:, :9].detach()
        taus   = Z[:, 9:13].detach()
        graphs = build_graph_batch(states, taus)

        # Encode hull nodes (differentiable — uses full Z including time)
        hull_enc = self.hull_encoder(Z)                        # (B, hull_enc_dim)
        for b in range(B):
            graphs[b]["hull"].x = hull_enc[b].unsqueeze(0)

        # Batch into single HeteroData for efficient message passing
        batch = Batch.from_data_list(graphs)

        # Message passing
        for mp_layer in self.mp_layers:
            batch = mp_layer(batch)

        # Readout
        hull_emb = batch["hull"].x                             # (B, hull_enc_dim)
        delta    = self.readout(hull_emb)                      # (B, N_out)

        # Residual + body→world rotation
        state_in = Z[:, :self.N_out]

        cos_psi_hat = delta[:, 3] + state_in[:, 3]
        sin_psi_hat = delta[:, 4] + state_in[:, 4]

        x_world = cos_psi_hat * delta[:, 0] - sin_psi_hat * delta[:, 1] + state_in[:, 0]
        y_world = sin_psi_hat * delta[:, 0] + cos_psi_hat * delta[:, 1] + state_in[:, 1]

        X_hat = delta + state_in
        X_hat = X_hat.clone()
        X_hat[:, 0] = x_world
        X_hat[:, 1] = y_world

        return X_hat
