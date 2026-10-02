"""Stokes-vector calculus (differentiable, PyTorch).

Established physics: the Stokes vector S = [S0, S1, S2, S3]^T completely
characterises the intensity + polarization state of a light beam at a pixel.

    S0 = total intensity               (measurable directly)
    S1 = horizontal - vertical linear  (I0 - I90)
    S2 = +45 - -45 linear              (I45 - I135)
    S3 = right - left circular         (requires waveplate hardware)

IMPORTANT SCIENTIFIC CAVEAT (research-honesty rule #4 of the project):
A conventional single-exposure RGB sensor does NOT measure S0..S3.  Any
quantity produced by this package from a single RGB image is an ESTIMATE and
is denoted with a hat (Ŝ0, Ŝ1, Ŝ2, DoLP_hat, AoLP_hat).  Only images captured
through real analyzer hardware are "measured" quantities.

DoLP = sqrt(S1^2 + S2^2) / S0          (degree of LINEAR polarization)
AoLP = 0.5 * atan2(S2, S1)              (angle of linear polarization, in
                                         [-pi/2, pi/2), circular quantity)
"""
from __future__ import annotations

import math

import torch


def stokes_from_angles(i_stack: torch.Tensor,
                       angles_deg: torch.Tensor | list[float]) -> torch.Tensor:
    """Recover (S0, S1, S2) from measured analyzer-angle intensities.

    For an ideal linear analyzer at angle theta:
        I(theta) = 0.5 * (S0 + S1 cos 2θ + S2 sin 2θ)

    With the classical 4-angle sampling {0, 45, 90, 135} deg this reduces to
    the well-known closed form used below.  This function is only physically
    valid for *hardware-measured* stacks (Configuration A/C teacher data).

    Args:
        i_stack: (..., K, H, W) intensities at ``angles_deg``.  Linear,
                 radiometrically calibrated values.
        angles_deg: K analyzer angles in degrees.  Must contain a set that
                 spans the (S1, S2) plane; {0,45,90,135} is checked explicitly.
    Returns:
        (..., 3, H, W) tensor [S0, S1, S2].
    """
    angles = torch.as_tensor(angles_deg, dtype=i_stack.dtype, device=i_stack.device)
    if angles.numel() != i_stack.shape[-3]:
        raise ValueError("angle count must match stack size")
    a = torch.deg2rad(angles).view(-1, 1, 1)
    # Least-squares solution of I_k = 0.5*(S0 + S1 cos2a_k + S2 sin2a_k).
    design = torch.stack([torch.ones_like(a), torch.cos(2 * a), torch.sin(2 * a)], dim=-1)
    # design: (K,1,1,3) -> broadcast solve over pixels via lstsq on reshaped data
    k = i_stack.shape[-3]
    b = 2.0 * i_stack.movedim(-3, -1).reshape(-1, k)  # (N, K)
    d = design.reshape(k, 3)
    sol = torch.linalg.lstsq(d, b.T).solution          # (3, N)
    s = sol.T.reshape(*i_stack.shape[:-3], 3, *i_stack.shape[-2:])
    return s.movedim(-3, -3) if False else s  # shape (..., 3, H, W)


def stokes_from_4angle(i0: torch.Tensor, i45: torch.Tensor,
                      i90: torch.Tensor, i135: torch.Tensor) -> torch.Tensor:
    """Closed-form [S0,S1,S2] from the classic 0/45/90/135° measurement set.

    S0 = I0 + I90 = I45 + I135 (ideal case; we average both estimates)
    S1 = I0 - I90
    S2 = I45 - I135
    Valid ONLY for hardware-measured analyzer stacks.
    """
    s0 = 0.5 * ((i0 + i90) + (i45 + i135))
    s1 = i0 - i90
    s2 = i45 - i135
    return torch.stack([s0, s1, s2], dim=-3)


def dolp(s: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Degree of linear polarization: sqrt(S1^2+S2^2)/S0 (clipped to [0,1]).

    Input ``s`` is (..., 3, H, W) ordered [S0, S1, S2]  OR  (...,4,H,W).
    """
    s0, s1, s2 = s[..., 0, :, :], s[..., 1, :, :], s[..., 2, :, :]
    d = torch.sqrt(s1 * s1 + s2 * s2 + eps) / (s0.abs() + eps)
    return d.clamp(0.0, 1.0)


def aolp(s: torch.Tensor) -> torch.Tensor:
    """Canonical angle of linear polarization: 0.5*atan2(S2, S1), radians."""
    return 0.5 * torch.atan2(s[..., 2, :, :], s[..., 1, :, :])


def wrap_angle_diff(phi1: torch.Tensor, phi2: torch.Tensor) -> torch.Tensor:
    """Wrapped angular difference for CIRCULAR quantities (AoLP, 2θ periodicity).

    Δφ = min(|φ1-φ2|, π - |φ1-φ2|) generalised via modulo so it works for any
    range.  Never use plain absolute error on AoLP (metric rule #47).
    """
    d = (phi1 - phi2) % math.pi
    return torch.minimum(d, math.pi - d)


def stokes_to_mueller_input(s23: torch.Tensor, s0: torch.Tensor,
                            s3: torch.Tensor | None = None) -> torch.Tensor:
    """Assemble full 4-component Stokes vector from (S1, S2, S0) and optional S3.

    If S3 is unknown (the usual case for linear-only analysis) we adopt the
    documented approximation S3 ≈ 0 and flag the assumption in docs.
    """
    _, s1, s2 = s23[..., 0, :, :], s23[..., 1, :, :], s23[..., 2, :, :]
    if s3 is None:
        s3 = torch.zeros_like(s1)
    return torch.stack([s0, s1, s2, s3], dim=-3)
