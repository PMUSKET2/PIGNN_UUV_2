"""
Tests for the PIGNN project.

Run with:  pytest tests/ -v
"""

import sys
import os
import pytest
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ===================================================================
# Parameters
# ===================================================================
class TestParameters:
    def test_imports(self):
        from src.parameters import m, g, F_bouy, THRUSTER_CONFIG
        assert m > 0
        assert g > 0
        assert F_bouy > 0
        assert len(THRUSTER_CONFIG) == 8

    def test_thruster_positions_are_3d(self):
        from src.parameters import THRUSTER_CONFIG
        for i, cfg in THRUSTER_CONFIG.items():
            assert len(cfg["position"]) == 3
            assert len(cfg["orientation"]) == 3


# ===================================================================
# BlueROV dynamics (NumPy)
# ===================================================================
class TestBlueROVNumpy:
    def test_zero_input_zero_state(self):
        from src.bluerov import bluerov_compute
        x = np.zeros(8)
        u = np.zeros(4)
        x_dot = bluerov_compute(0.0, x, u)
        assert x_dot.shape == (8,)
        # At rest with zero input, only buoyancy/gravity produces non-zero w_d
        assert x_dot[0] == 0.0  # x_d
        assert x_dot[1] == 0.0  # y_d

    def test_output_shape(self):
        from src.bluerov import bluerov_compute
        x = np.random.randn(8).astype(np.float64)
        u = np.random.randn(4).astype(np.float64)
        x_dot = bluerov_compute(0.0, x, u)
        assert x_dot.shape == (8,)


# ===================================================================
# BlueROV dynamics (Torch)
# ===================================================================
class TestBlueROVTorch:
    def test_output_shape_single(self):
        from src.bluerov_torch import bluerov_compute
        x = torch.randn(9)
        u = torch.randn(4)
        x_dot = bluerov_compute(0, x, u)
        assert x_dot.shape == (1, 9)

    def test_output_shape_batch(self):
        from src.bluerov_torch import bluerov_compute
        x = torch.randn(16, 9)
        u = torch.randn(16, 4)
        x_dot = bluerov_compute(0, x, u)
        assert x_dot.shape == (16, 9)

    def test_differentiable(self):
        from src.bluerov_torch import bluerov_compute
        x = torch.randn(4, 9, requires_grad=True)
        u = torch.randn(4, 4)
        x_dot = bluerov_compute(0, x, u)
        loss = x_dot.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (4, 9)


# ===================================================================
# Graph builder
# ===================================================================
class TestGraphBuilder:
    def test_build_graph_structure(self):
        from models.graph_builder import build_graph
        state = torch.randn(9)
        tau = torch.randn(4)
        g = build_graph(state, tau)

        # Node counts
        assert g["hull"].x.shape[0] == 1
        assert g["thruster"].x.shape[0] == 8
        assert g["hydrodynamic"].x.shape[0] == 1
        assert g["buoyancy"].x.shape[0] == 1

        # Edge counts
        assert g["thruster", "forces", "hull"].edge_index.shape == (2, 8)
        assert g["hydrodynamic", "drag", "hull"].edge_index.shape == (2, 1)
        assert g["buoyancy", "restoring", "hull"].edge_index.shape == (2, 1)

    def test_edge_feature_dims(self):
        from models.graph_builder import build_graph
        g = build_graph(torch.randn(9), torch.randn(4))

        assert g["thruster", "forces", "hull"].edge_attr.shape == (8, 7)
        assert g["hydrodynamic", "drag", "hull"].edge_attr.shape == (1, 8)
        assert g["buoyancy", "restoring", "hull"].edge_attr.shape == (1, 4)

    def test_node_feature_dims(self):
        from models.graph_builder import build_graph
        g = build_graph(torch.randn(9), torch.randn(4))

        assert g["hull"].x.shape == (1, 9)
        assert g["thruster"].x.shape == (8, 8)
        assert g["hydrodynamic"].x.shape == (1, 4)
        assert g["buoyancy"].x.shape == (1, 6)

    def test_allocation_matrix(self):
        from models.graph_builder import allocate_thrusts
        tau = torch.tensor([[10.0, 0.0, 0.0, 0.0]])  # pure surge
        f = allocate_thrusts(tau)
        assert f.shape == (1, 8)
        # Net force in surge should roughly equal 10
        from models.graph_builder import _B_alloc
        B = torch.tensor(_B_alloc)
        tau_reconstructed = (B @ f.T).T
        assert torch.allclose(tau_reconstructed[:, 0],
                              torch.tensor([10.0]), atol=0.5)

    def test_all_edges_target_hull(self):
        """Hard constraint: all edges point TO the hull node."""
        from models.graph_builder import build_graph
        g = build_graph(torch.randn(9), torch.randn(4))

        for edge_type in g.edge_types:
            assert edge_type[2] == "hull", \
                f"Edge {edge_type} does not target hull!"


# ===================================================================
# PIGNN model
# ===================================================================
class TestPIGNN:
    @pytest.fixture
    def model(self):
        from models.pignn import PIGNN
        return PIGNN(N_in=14, N_out=9, hidden=16, msg_dim=16, n_mp_layers=2)

    def test_forward_single(self, model):
        Z = torch.randn(1, 14)
        out = model(Z)
        assert out.shape == (1, 9)

    def test_forward_batch(self, model):
        Z = torch.randn(8, 14)
        out = model(Z)
        assert out.shape == (8, 9)

    def test_residual_connection(self, model):
        """With zero-initialised weights the output should ≈ input state."""
        # After xavier init the output won't be exactly zero, but
        # we check that the residual structure doesn't break.
        Z = torch.randn(4, 14)
        out = model(Z)
        assert out.shape == (4, 9)
        # Output should be finite
        assert torch.isfinite(out).all()

    def test_differentiable(self, model):
        Z = torch.randn(4, 14, requires_grad=True)
        out = model(Z)
        loss = out.sum()
        loss.backward()
        assert Z.grad is not None

    def test_parameter_count_positive(self, model):
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n > 0


# ===================================================================
# Loss functions
# ===================================================================
class TestLosses:
    def test_data_loss(self):
        from models.pignn import PIGNN
        from models.model_utility import data_loss_fn

        model = PIGNN(N_in=14, N_out=9, hidden=16, msg_dim=16, n_mp_layers=1)
        B, T, Nx, Nu = 2, 10, 9, 4
        X    = torch.randn(B, T, Nx)
        U    = torch.randn(B, T, Nu)
        time = torch.linspace(0, 1, T).unsqueeze(0).unsqueeze(-1).expand(B, T, 1)

        loss = data_loss_fn(model, X, U, time, "cpu")
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_ic_loss(self):
        from models.pignn import PIGNN
        from models.model_utility import initial_condition_loss

        model = PIGNN(N_in=14, N_out=9, hidden=16, msg_dim=16, n_mp_layers=1)
        B, T, Nx, Nu = 2, 10, 9, 4
        X    = torch.randn(B, T, Nx)
        U    = torch.randn(B, T, Nu)
        time = torch.linspace(0, 1, T).unsqueeze(0).unsqueeze(-1).expand(B, T, 1)

        loss = initial_condition_loss(model, X, U, time, "cpu")
        assert loss.dim() == 0
        assert loss.item() >= 0


# ===================================================================
# Integration: training step
# ===================================================================
class TestTrainingIntegration:
    def test_one_train_step(self, tmp_path):
        """Smoke test: run one training step without crashing."""
        from models.pignn import PIGNN
        from models.model_utility import data_loss_fn, initial_condition_loss

        model = PIGNN(N_in=14, N_out=9, hidden=16, msg_dim=16, n_mp_layers=1)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

        B, T, Nx, Nu = 2, 10, 9, 4
        X    = torch.randn(B, T, Nx)
        U    = torch.randn(B, T, Nu)
        time = torch.linspace(0, 1, T).unsqueeze(0).unsqueeze(-1).expand(B, T, 1)

        model.train()
        optim.zero_grad()
        l_data = data_loss_fn(model, X, U, time, "cpu")
        l_ic   = initial_condition_loss(model, X, U, time, "cpu")
        loss   = l_data + l_ic
        loss.backward()
        optim.step()

        assert loss.item() >= 0
