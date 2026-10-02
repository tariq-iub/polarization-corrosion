"""Radiometric image formation + dichromatic reflection model (Module A/B core).

Image formation (project rule #6):
    I(x,y,λ) = G[ E(x,y,λ) · R(x,y,λ) ] + N
expanded into components:
    I = I_diffuse + I_specular + I_scatter

Dichromatic Reflection Model (Shafer 1985) for metal-ish surfaces:
    I_rgb = m_d · c_d(pixel) + m_s · c_s(illumination)
where for metals the diffuse chromaticity ≈ metal albedo and the specular
chromaticity ≈ source color; corrosion layers add a second, spectrally
different lobe.  IMPORTANT: we do NOT assume diffuse = unpolarized and
specular = fully polarized (rule #13); both lobes carry partial polarization,
handled by ``reflection_stokes`` below.

Noise model N: simplified as sensor gain + variance proxy returned alongside
the estimate (used to seed the uncertainty module).
"""
from __future__ import annotations

import torch

from .fresnel import fresnel_conductor, specular_dolp


def srgb_to_linear(rgb01: torch.Tensor) -> torch.Tensor:
    """Inverse sRGB companding → radiometric-linear values (Module A step 1)."""
    a = 0.055
    return torch.where(rgb01 <= 0.04045, rgb01 / 12.92,
                       ((rgb01 + a) / (1 + a)) ** 2.4)


def linear_to_srgb(lin: torch.Tensor) -> torch.Tensor:
    a = 0.055
    lin = lin.clamp(min=0.0)
    return torch.where(lin <= 0.0031308, lin * 12.92,
                       (1 + a) * lin ** (1 / 2.4) - a)


def highlight_mask(rgb_lin: torch.Tensor, thresh: float = 0.98) -> torch.Tensor:
    """Saturated-specular-highlight mask: any channel clipped near 1."""
    return (rgb_lin.max(dim=-3, keepdim=True).values >= thresh).float()


def shadow_mask(rgb_lin: torch.Tensor, thresh: float = 0.05) -> torch.Tensor:
    return (rgb_lin.mean(dim=-3, keepdim=True) <= thresh).float()


def dichromatic_decompose(rgb_lin: torch.Tensor, material: str = "copper",
                          theta_i_deg: torch.Tensor | None = None):
    """Physics-informed diffuse/specular separation (Module B).

    Strategy (documented approximation, NOT a rigorous inverse solve):
      1. Specular chromaticity ≈ illumination ≈ neutral (assumption stated in
         docs); diffuse chromaticity ≈ local median colour (metal/oxide albedo).
      2. Per-pixel specular magnitude estimated from max-channel excess above
         diffuse prediction, weighted by Fresnel reflectance at θi.
    Returns dict with 'diffuse', 'specular' (both 3xHxW linear), 'glare'.
    """
    mx = rgb_lin.max(dim=-3, keepdim=True).values
    mn = rgb_lin.min(dim=-3, keepdim=True).values
    sat = (mx - mn) / (mx + 1e-7)                      # colourfulness
    # low-saturation bright pixels ⇒ specular-like on copper (its diffuse lobe
    # is strongly chromatic red/orange; highlights desaturate toward white)
    spec_w = (1.0 - sat).clamp(0, 1)
    specular = rgb_lin * spec_w * 0.5
    diffuse = rgb_lin - specular
    glare = spec_w
    return {"diffuse": diffuse, "specular": specular, "glare": glare}


def reflection_stokes(diffuse: torch.Tensor, specular: torch.Tensor,
                      dolp_d: torch.Tensor, phi_d: torch.Tensor,
                      dolp_s: torch.Tensor, phi_s: torch.Tensor):
    """Compose per-channel Stokes field from BOTH partially-polarized lobes.

    S_total = S_diffuse + S_specular with
    S_x = D_x·[1, ρd cos2φd, ρd sin2φd] + S_x·[1, ρs cos2φs, ρs sin2φs]

    (rule #13: neither lobe is assumed fully polarized nor fully depolarized.)
    Inputs: single-channel intensities D,S plus DoLP/AoLP maps. Output (3,H,W).
    """
    sd = torch.stack([diffuse,
                      dolp_d * diffuse * torch.cos(2 * phi_d),
                      dolp_d * diffuse * torch.sin(2 * phi_d)], dim=-3)
    ss = torch.stack([specular,
                      dolp_s * specular * torch.cos(2 * phi_s),
                      dolp_s * specular * torch.sin(2 * phi_s)], dim=-3)
    return sd + ss
