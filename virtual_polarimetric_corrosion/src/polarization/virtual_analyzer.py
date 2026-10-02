"""Virtual polarizer / analyzer + counterfactual polarization stack (Modules D/E).

Differentiable software emulation of a rotating linear analyzer acting on an
ESTIMATED Stokes field.  Malus-compatible general form (rule #10):

    Î(θ) = ½ [ Ŝ0 + Ŝ1 cos 2θ + Ŝ2 sin 2θ ]

Deployed as either:
  * explicit Mueller calculus (optics.mueller.apply_mueller), or
  * the closed-form first row above (mathematically identical, faster).

The Virtual Polarimetric Stack (rule #11) is the set {Î(θ_k)} for θ over a
configurable sweep — "plausible analyzer-dependent observations predicted
from an ordinary RGB image", NOT physical measurements.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..optics.mueller import mueller_linear_polarizer, apply_mueller


class VirtualAnalyzer(nn.Module):
    """Differentiable virtual linear analyzer, θ ∈ [0°, 180°).

    forward(stokes3, theta_deg) -> intensity image (H,W) or (B,H,W).
    stokes3: (...,3,H,W) ESTIMATED [Ŝ0, Ŝ1, Ŝ2] (hatted quantities!).
    """

    def __init__(self, use_mueller: bool = False):
        super().__init__()
        self.use_mueller = use_mueller

    def forward(self, stokes3: torch.Tensor, theta_deg) -> torch.Tensor:
        t = torch.deg2rad(torch.as_tensor(float(theta_deg) if not torch.is_tensor(theta_deg)
                                          else theta_deg,
                                          dtype=stokes3.dtype, device=stokes3.device))
        if self.use_mueller:
            s4 = torch.nn.functional.pad(stokes3, (0, 0, 0, 1))  # S3≈0 assumption
            m = mueller_linear_polarizer(t, dtype=stokes3.dtype, device=stokes3.device)
            out = apply_mueller(s4, m)
            return out[..., 0, :, :]
        c2, s2 = torch.cos(2 * t), torch.sin(2 * t)
        return 0.5 * (stokes3[..., 0, :, :] + c2 * stokes3[..., 1, :, :]
                      + s2 * stokes3[..., 2, :, :])


DEFAULT_STACK_ANGLES = [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5]


def virtual_polarimetric_stack(stokes3: torch.Tensor,
                               angles=DEFAULT_STACK_ANGLES,
                               analyzer: VirtualAnalyzer | None = None
                               ) -> torch.Tensor:
    """Counterfactual stack Î_θk, returned (K,...,H,W) (rule #11)."""
    analyzer = analyzer or VirtualAnalyzer()
    outs = [analyzer(stokes3, a) for a in angles]
    return torch.stack(outs, dim=-4 if stokes3.dim() == 3 else -3)


def cross_polarized_proxy(stokes3: torch.Tensor) -> torch.Tensor:
    """Virtual 'crossed polarizers' view: Î(90°)−Î(0°)-style orthogonal pair.

    Emulates the glare-killing look of a hardware cross-polarizer without ever
    claiming equivalence to one (see LIMITATIONS.md).
    """
    va = VirtualAnalyzer()
    i0 = va(stokes3, 0.0)
    i90 = va(stokes3, 90.0)
    denom = (i0 + i90).clamp(min=1e-6)
    return (i0 - i90) / denom  # normalized orthogonal-analyzer contrast map


def periodicity_residual(stokes3: torch.Tensor, angles=None,
                         analyzer: VirtualAnalyzer | None = None) -> float:
    """Numerical test of I(θ) ≈ I(θ+180°) (rule #22).

    The Malus form is exactly π-periodic by construction; this function exists
    so tests can assert the residual numerically and catch future regressions
    (e.g. if a learned per-angle bias is added).
    """
    angles = angles or [0.0, 37.0, 81.0, 123.0, 169.0]
    analyzer = analyzer or VirtualAnalyzer()
    worst = 0.0
    for a in angles:
        i_a = analyzer(stokes3, a)
        i_b = analyzer(stokes3, a + 180.0)
        worst = max(worst, float((i_a - i_b).abs().max()))
    return worst
