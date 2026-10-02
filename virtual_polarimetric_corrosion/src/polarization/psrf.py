"""Polarimetric Surface Response Field — PSRF (core novel representation, rule #17).

For every pixel, the PSRF is the analyzer-dependent response
    R_p(θ) ≈ a0 + a1 cos 2θ + b1 sin 2θ + a2 cos 4θ + b2 sin 4θ
i.e. a truncated circular-harmonic expansion of Î(x,y;θ) over θ∈[0,2π).

* The first harmonic (a1,b1) corresponds naturally to linear polarization:
      a1 = Ŝ1/2, b1 = Ŝ2/2, a0 = Ŝ0/2   (for a Malus-consistent field).
* Higher harmonics capture non-sinusoidal angular structure (e.g. if a learned
  per-angle correction or microfacet lobe sharpening is introduced).  They are
  retained ONLY if experiments justify them (ablation A-PSRF-harmonics).

Differentiable: coefficients obtained by a fixed projection matrix (linear in
the stack → gradients flow to the latent optical estimator).
"""
from __future__ import annotations

import math

import numpy as np
import torch


def harmonic_basis(n_harm: int = 2, n_angles: int = 64,
                   dtype=torch.float32) -> torch.Tensor:
    """Design matrix B (K x (1+2*n_harm)) sampling θk uniformly over [0,180°).

    Columns: [1, cos2θ, sin2θ, cos4θ, sin4θ, ...].  Because R_p(θ) is
    π-periodic for linear optics, uniform [0,180°) sampling is sufficient and
    well-conditioned.
    """
    theta = np.linspace(0.0, math.pi, n_angles, endpoint=False)
    cols = [np.ones_like(theta)]
    for m in range(1, n_harm + 1):
        cols += [np.cos(2 * m * theta), np.sin(2 * m * theta)]
    return torch.tensor(np.stack(cols, axis=1), dtype=dtype)


def psrf_from_stack(stack: torch.Tensor, angles_deg, n_harm: int = 2) -> torch.Tensor:
    """Fit circular harmonics to a sampled virtual stack.

    Args:
        stack: (K,H,W) virtual intensities at ``angles_deg`` (need not be uniform).
        angles_deg: K angles (degrees).
    Returns:
        (C,H,W) coefficients [a0, a1, b1, (a2, b2)] via least squares.
    """
    a = torch.deg2rad(torch.as_tensor(angles_deg, dtype=stack.dtype, device=stack.device))
    cols = [torch.ones_like(a)]
    for m in range(1, n_harm + 1):
        cols += [torch.cos(2 * m * a), torch.sin(2 * m * a)]
    b = torch.stack(cols, dim=-1)                      # (K, C)
    k, c = b.shape
    y = stack.reshape(k, -1)                           # (K, N)
    coef = torch.linalg.lstsq(b, y).solution           # (C, N)
    return coef.reshape(c, *stack.shape[-2:])          # (C,H,W)


def psrf_eval(coeffs: torch.Tensor, theta_rad) -> torch.Tensor:
    """Evaluate R_p(θ) from coefficients — differentiable in coeffs."""
    t = torch.as_tensor(theta_rad, dtype=coeffs.dtype, device=coeffs.device)
    out = coeffs[..., 0, :, :].clone()
    m = 1
    while 1 + 2 * m <= coeffs.shape[-3]:
        out = out + coeffs[..., 2 * m - 1, :, :] * torch.cos(2 * m * t) \
                  + coeffs[..., 2 * m, :, :] * torch.sin(2 * m * t)
        m += 1
    return out


def psrf_energy_metrics(coeffs: torch.Tensor) -> dict:
    """Per-pixel summary features used by the corrosion feature vector (rule #18).

    dc          : a0            (mean response level)
    ac_lin      : sqrt(a1²+b1²) (first-harmonic amplitude ≈ DoLP·S0/2)
    ac_quad     : sqrt(a2²+b2²) if present (higher-order justification test)
    harmonic_ratio : ac_lin / (|a0|+eps) — dimensionless "polarization contrast"
    phase       : 0.5·atan2(b1,a1) (AoLP_hat recovered from harmonics)
    """
    a0 = coeffs[..., 0, :, :]
    a1 = coeffs[..., 1, :, :]
    b1 = coeffs[..., 2, :, :]
    eps = 1e-8
    ac_lin = torch.sqrt(a1 * a1 + b1 * b1 + eps)
    out = {"dc": a0, "ac_lin": ac_lin,
           "harmonic_ratio": ac_lin / (a0.abs() + eps),
           "phase": 0.5 * torch.atan2(b1, a1)}
    if coeffs.shape[-3] >= 5:
        a2, b2 = coeffs[..., 3, :, :], coeffs[..., 4, :, :]
        out["ac_quad"] = torch.sqrt(a2 * a2 + b2 * b2 + eps)
        out["quad_ratio"] = out["ac_quad"] / (ac_lin + eps)
    return out
