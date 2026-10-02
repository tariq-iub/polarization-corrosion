"""Mueller calculus for the virtual optical system (differentiable).

Established physics: any linear optical element acting on a Stokes vector is
described by a 4x4 Mueller matrix M with S_out = M S_in.  This module makes
the software emulate the *effect of rotating a physical analyzer*, rather than
applying an ad-hoc brightness filter (project rule #9).

Contents
--------
* mueller_linear_polarizer(theta) : ideal linear polarizer/analyzer M_P(θ)
* mueller_retarder(delta, theta)  : ideal linear retarder (waveplate), used by
  the synthetic renderer for circular-polarization channels
* mueller_depolarizer(p)          : partial-depolarisation surrogate for rough
  corrosion layers (documented approximation, not exact physics)
* apply_mueller(S, M)             : batched application over images
"""
from __future__ import annotations

import torch


def mueller_linear_polarizer(theta_rad: float | torch.Tensor,
                             dtype=torch.float32, device=None) -> torch.Tensor:
    """Mueller matrix of an ideal linear polarizer at transmission axis θ.

        M_P(θ) = 1/2 * [[1,      cos2θ,     sin2θ, 0],
                        [cos2θ,  cos^2 2θ,  sin2θ cos2θ, 0],
                        [sin2θ,  sin2θ cos2θ, sin^2 2θ, 0],
                        [0, 0, 0, 0]]

    The first row gives the transmitted intensity:
        I(θ) = 0.5 (S0 + S1 cos2θ + S2 sin2θ)   (Malus-compatible).
    """
    t = torch.as_tensor(theta_rad, dtype=dtype, device=device)
    c, s = torch.cos(2 * t), torch.sin(2 * t)
    z = torch.zeros_like(c)
    o = torch.ones_like(c)
    m = 0.5 * torch.stack([
        torch.stack([o, c, s, z], dim=-1),
        torch.stack([c, c * c, c * s, z], dim=-1),
        torch.stack([s, c * s, s * s, z], dim=-1),
        torch.stack([z, z, z, z], dim=-1)], dim=-2)
    return m


def mueller_retarder(delta_rad: float | torch.Tensor,
                     theta_rad: float | torch.Tensor,
                     dtype=torch.float32, device=None) -> torch.Tensor:
    """Ideal linear retarder with phase retardance δ and fast axis θ.

    Used only in the SYNTHETIC renderer to generate physically consistent
    circular-polarization test data; real RGB deployment never measures δ.
    """
    d = torch.as_tensor(delta_rad, dtype=dtype, device=device)
    t = torch.as_tensor(theta_rad, dtype=dtype, device=device)
    c2, s2 = torch.cos(2 * t), torch.sin(2 * t)
    cd, sd = torch.cos(d), torch.sin(d)
    o = torch.ones_like(d)
    z = torch.zeros_like(d)
    # Canonical textbook Mueller matrix of a linear retarder (e.g. Collett,
    # "Field Guide to Polarization", Eq. 4-series):
    m = torch.stack([
        torch.stack([o, z, z, z], -1),
        torch.stack([z, c2 * c2 + s2 * s2 * cd, 2 * c2 * s2 * (1 - cd), 2 * s2 * sd], -1),
        torch.stack([z, 2 * c2 * s2 * (1 - cd), c2 * c2 * (1 - cd) + s2 * s2, -2 * c2 * sd], -1),
        torch.stack([z, -2 * s2 * sd, 2 * c2 * sd, cd], -1)], dim=-2)
    return m


def mueller_depolarizer(p: float | torch.Tensor,
                        dtype=torch.float32, device=None) -> torch.Tensor:
    """Partial depolarizer diag(1, p, p, p).  DOCUMENTED APPROXIMATION:

    Rough corrosion deposits scatter light multiple times and reduce net
    polarization.  A diagonal depolarizer is the simplest Mueller-Jones
    consistent model; real corroded surfaces may be diattenuating AND
    depolarizing — see LIMITATIONS.md.
    """
    pp = torch.as_tensor(p, dtype=dtype, device=device)
    o = torch.ones_like(pp)
    z = torch.zeros_like(pp)
    return torch.diag_embed(torch.stack([o, pp, pp, pp], dim=-1))


def apply_mueller(s_in: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Apply Mueller matrix m (...,4,4) to Stokes field s_in (...,4,H,W)."""
    return torch.einsum("...ij,...jhw->...ihw", m, s_in)


def transmitted_intensity(s_in4: torch.Tensor, theta_rad) -> torch.Tensor:
    """I(θ) = [M_P(θ) S]_0 = 0.5 (S0 + S1 cos2θ + S2 sin2θ).

    s_in4: (...,4,H,W) full Stokes (S3 ignored — ideal linear polarizer row 0
    has zero coupling to S3, which we exploit for efficiency).
    """
    c, s = torch.cos(2 * torch.as_tensor(theta_rad, device=s_in4.device)), \
           torch.sin(2 * torch.as_tensor(theta_rad, device=s_in4.device))
    return 0.5 * (s_in4[..., 0, :, :] + c * s_in4[..., 1, :, :] + s * s_in4[..., 2, :, :])
