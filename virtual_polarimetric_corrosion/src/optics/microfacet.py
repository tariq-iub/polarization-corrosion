"""Microfacet BRDF: Cook–Torrance with GGX/Trowbridge–Reitz NDF (differentiable).

Established physics (Cook & Torrance 1982; Walter et al. 2007 for GGX):

    f(ωi,ωo) = D(h) F(ωi·h) G(ωi,ωo,h) / (4 (n·ωi)(n·ωo))

* D  : GGX normal distribution   α² / (π ((n·h)²(α⁻²−1)+1)²)
* G  : Smith joint masking-shadowing with GGX visibility
* F  : Fresnel term — Schlick approximation for dielectrics, and the more
       accurate conductor form using complex IOR:
       F(θ) ≈ R_s(θ)+R_p(θ) over 2 evaluated per-channel (we expose both an
       exact-per-angle path via fresnel.py and a fast Schlick-style path).

Polarization extension used by this project: the specular lobe carries a DoLP
determined by the Fresnel asymmetry (see optics.fresnel.specular_dolp), and
roughness α *reduces* net polarization through multiple-scatter depolarisation
(modelled as diag(1, p(α), p(α), ·) with p(α)=exp(−c·α) — DOCUMENTED
APPROXIMATION, cf. Müller et al., "Polarizing multi-layer BRDFs", CGF 2019).

Validity notes (project rule #6): microfacet theory assumes facets larger than
a wavelength and single scattering per facet; nano-scale corrosion crystals,
poridal patina and sub-surface salt scattering violate these assumptions →
treat outputs as smooth proxies, not quantitative BRDF fits.
"""
from __future__ import annotations

import math

import torch

EPS = 1e-7


def ggx_D(n_dot_h: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    a2 = alpha ** 2 + EPS
    d = (n_dot_h ** 2) * (a2.reciprocal() - 1.0) + 1.0
    return a2 / (math.pi * d * d + EPS)


def _smith_ggx_single(n_dot_v: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    a2 = alpha ** 2
    return 2.0 * n_dot_v / (n_dot_v + torch.sqrt(a2 + (1.0 - a2) * n_dot_v ** 2 + EPS))


def smith_ggx_G(n_dot_i, n_dot_o, n_dot_v_light, alpha):  # joint
    """Smith joint masking-shadowing (height-correlated separable form)."""
    return _smith_ggx_single(n_dot_i, alpha) * _smith_ggx_single(n_dot_o, alpha)


def schlick_fresnel(cos_theta: torch.Tensor, f0) -> torch.Tensor:
    """Schlick approximation. For metals use per-channel F0 from complex IOR:
        F0(λ) = ((n−1)² + κ²) / ((n+1)² + κ²)
    """
    f0 = torch.as_tensor(f0, dtype=cos_theta.dtype, device=cos_theta.device)
    return f0 + (1.0 - f0) * (1.0 - cos_theta.clamp(0, 1)) ** 5


def f0_metal(n: float, k: float):
    return ((n - 1) ** 2 + k ** 2) / ((n + 1) ** 2 + k ** 2)


def cook_torrance_specular(n_dot_l, n_dot_v, n_dot_h, v_dot_h, alpha, f0):
    """Scalar (unpolarized) CT specular reflectance factor."""
    d = ggx_D(n_dot_h.clamp(min=EPS), alpha)
    g = smith_ggx_G(n_dot_l.clamp(min=EPS), n_dot_v.clamp(min=EPS), None, alpha)
    f = schlick_fresnel(v_dot_h.clamp(min=EPS), f0)
    denom = 4.0 * n_dot_l.clamp(min=EPS) * n_dot_v.clamp(min=EPS) + EPS
    return (d * g * f) / denom


def specular_polarization(alpha: torch.Tensor, theta_eff: torch.Tensor,
                          r_s: torch.Tensor, r_p: torch.Tensor,
                          c_depolar: float = 2.0):
    """Return (intensity_factor, dolp_factor) of the specular lobe.

    dolp = |Rp−Rs|/(Rp+Rs) × exp(−c_depolar·α): rougher surfaces depolarize.
    DOCUMENTED APPROXIMATION (single-scatter Fresnel × empirical roughness
    attenuation); see docs/MATHEMATICAL_FORMULATION.md §Proposed approximations.
    """
    base = ((r_p - r_s).abs()) / (r_p + r_s + EPS)
    atten = torch.exp(-c_depolar * alpha)
    return 0.5 * (r_p + r_s), base * atten
