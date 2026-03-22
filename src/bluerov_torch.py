"""
BlueROV2 4-DOF dynamics (PyTorch, differentiable).

State representation uses cos/sin parameterisation for yaw:
    x_ = [x, y, z, cos(ψ), sin(ψ), u, v, w, r]   (dim = 9)
Control:
    u_ = [X, Y, Z, M_z]                              (dim = 4)
"""

import torch
from src.parameters import (
    m, X_ud, Y_vd, Z_wd, I_zz, N_rd,
    X_u, X_uc, Y_v, Y_vc, Z_w, Z_wc,
    N_r, N_rc, g, F_bouy
)


def ssa(angle):
    """Smallest signed angle (tensor version)."""
    return angle - 2 * torch.pi * torch.floor_divide(angle + torch.pi, 2 * torch.pi)


def bluerov_compute(t, x_, u_):
    """
    Compute ẋ for the BlueROV2 4-DOF model (batched, differentiable).

    Parameters
    ----------
    t  : unused
    x_ : Tensor (B, 9)  — [x, y, z, cos(ψ), sin(ψ), u, v, w, r]
    u_ : Tensor (B, 4)  — [X, Y, Z, M_z]

    Returns
    -------
    x_dot : Tensor (B, 9)
    """
    x_ = x_.unsqueeze(0) if x_.dim() == 1 else x_
    u_ = u_.unsqueeze(0) if u_.dim() == 1 else u_

    cos_psi = x_[:, 3]
    sin_psi = x_[:, 4]

    u, v, w, r = x_[:, 5], x_[:, 6], x_[:, 7], x_[:, 8]
    X, Y, Z_f, M_z = u_[:, 0], u_[:, 1], u_[:, 2], u_[:, 3]

    # Kinematics
    x_d = cos_psi * u - sin_psi * v
    y_d = sin_psi * u + cos_psi * v
    z_d = w
    cos_psi_d = -sin_psi * r
    sin_psi_d =  cos_psi * r

    eta_dot = torch.stack([x_d, y_d, z_d, cos_psi_d, sin_psi_d], dim=1)

    # Kinetics
    u_d = 1 / (m - X_ud) * (X   + (m - Y_vd) * v * r + (X_u + X_uc * torch.abs(u)) * u)
    v_d = 1 / (m - Y_vd) * (Y   - (m - X_ud) * u * r + (Y_v + Y_vc * torch.abs(v)) * v)
    w_d = 1 / (m - Z_wd) * (Z_f + (Z_w + Z_wc * torch.abs(w)) * w + m * g - F_bouy)
    r_d = 1 / (I_zz - N_rd) * (M_z - (X_ud - Y_vd) * u * v + (N_r + N_rc * torch.abs(r)) * r)

    nu_dot = torch.stack([u_d, v_d, w_d, r_d], dim=1)
    x_dot  = torch.cat([eta_dot, nu_dot], dim=1)
    return x_dot
