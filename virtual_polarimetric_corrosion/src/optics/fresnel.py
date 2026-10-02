"""Fresnel reflection for DIELECTRICS and CONDUCTORS (differentiable).

Established physics.  Amplitude reflection coefficients at a planar interface
for s- (TE) and p- (TM) polarization:

Dielectric (real indices n1, n2, Snell: n1 sinθi = n2 sinθt):

    r_s = (n1 cosθi − n2 cosθt) / (n1 cosθi + n2 cosθt)
    r_p = (n2 cosθi − n1 cosθt) / (n2 cosθi + n1 cosθt)
    R_s = |r_s|²,  R_p = |r_p|²

Conductor / metal — complex refractive index N = n − iκ (engineering e^{+iωt}
convention; equivalently N = n + iκ with the optics e^{-iωt} convention — we
use n+iκ below and state the convention explicitly):

    cosθt = sqrt( (n1/N2)² − (n1/N2)² sin²θi )      (complex square root)

    r_s = (n1 cosθi − n2 cosθt)/(n1 cosθi + n2 cosθt)
    r_p = (n2 cosθi − n1 cosθt)/(n2 cosθi + n1 cosθt)     (same algebra, now
                                                           complex-valued)

Derived polarimetric quantities:
    ρ = rp/rs   (complex ratio) → phase shift Δφ = arg(ρ)
    DoLP of specular reflection from unpolarized illumination:
        DoLP_spec = (R_p − R_s)/(R_p + R_s)   (negative by our sign convention
        meaning the specular lobe is preferentially s-polarized; magnitude
        reported as |DoLP|)
    AoLP_spec ≈ transmission-axis direction perpendicular to the plane of
        incidence for grazing angles (s-dominated).

Copper/bronze note: optical constants of Cu vary strongly across the visible
(n≈0.44, κ≈1.34 @550nm … n≈1.9, κ≈3.2 @700nm), which is why we evaluate
channel-dependent Fresnel coefficients per RGB representative wavelength.
Values are literature approximations (see docs/POLARIZATION_PHYSICS.md); they
are NOT measured for the specific specimens in this study.
"""
from __future__ import annotations

import math

import torch

# Representative wavelengths for R,G,B channels (nm) used in the RGB spectral
# approximation (project rule #41).  Stated approximation: camera spectral
# response unknown → delta sampling at channel centroids.
RGB_WAVELENGTHS_NM = {"R": 630.0, "G": 532.0, "B": 465.0}

# Bulk copper (annealed) approximate complex IOR N = n + iκ at these wavelengths
# (Palik-style tabulated values, rounded; see docs for provenance & caveat).
CU_IOR = {"R": (2.00, 3.20), "G": (1.00, 2.00), "B": (0.46, 1.60)}
# Tin bronze (~88%Cu/12%Sn) — interpolated/rounded literature proxy.
BRONZE_IOR = {"R": (2.20, 3.30), "G": (1.15, 2.15), "B": (0.55, 1.75)}
# Corrosion-product PROXIES (optical behaviour proxies, NOT verified chemistry):
# cuprite Cu2O-like red-brown layer: semiconducting, higher n, moderate κ
CU2O_LIKE_IOR = {"R": (1.90, 0.90), "G": (2.20, 1.10), "B": (2.40, 1.30)}
# tenorite CuO-like black layer: lossy, lower contrast polarization
CUO_LIKE_IOR = {"R": (1.70, 1.90), "G": (1.90, 2.10), "B": (2.10, 2.30)}
# chloride/patina proxy (atacamite-like, greenish, partially transparent salt):
CHLORIDE_LIKE_IOR = {"R": (1.35, 0.35), "G": (1.45, 0.30), "B": (1.55, 0.28)}

MATERIAL_IOR = {
    "copper": CU_IOR,
    "bronze": BRONZE_IOR,
    "cu2o_like": CU2O_LIKE_IOR,
    "cuo_like": CUO_LIKE_IOR,
    "chloride_like": CHLORIDE_LIKE_IOR,
    "dielectric": None,  # uses scalar n passed by caller
}


def fresnel_dielectric(theta_i_rad: torch.Tensor, n1: float, n2: float):
    """Return (R_s, R_p) real reflectances for lossless dielectric interface."""
    ti = theta_i_rad
    sin_t = (n1 / n2) * torch.sin(ti)
    cos_t = torch.sqrt(torch.clamp(1.0 - sin_t ** 2, min=0.0))
    cos_i = torch.cos(ti)
    rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t + 1e-12)
    rp = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t + 1e-12)
    return rs ** 2, rp ** 2


def fresnel_conductor(theta_i_rad: torch.Tensor, n1: float,
                     n: torch.Tensor | float, k: torch.Tensor | float):
    """Complex-index Fresnel reflectances R_s, R_p and amplitude ratio rho.

    N2 = n + i k  (absorbing convention; time factor e^{-iωt}).
    All operations use torch complex tensors → differentiable w.r.t. n, k, θ.
    Returns dict with R_s, R_p (real), dphi (=arg(rp/rs)), and per-pol phase.
    """
    ti = torch.as_tensor(theta_i_rad)
    nn = torch.as_tensor(n, dtype=ti.dtype, device=ti.device)
    kk = torch.as_tensor(k, dtype=ti.dtype, device=ti.device)
    n2 = torch.complex(nn, kk)
    sin_i = torch.sin(ti)
    cos_i = torch.cos(ti)
    # complex Snell
    sin_t2 = (n1 / n2) ** 2 * sin_i ** 2
    cos_t = torch.sqrt(1.0 - sin_t2 + 0j)
    if torch.is_complex(cos_t):
        # choose branch with non-negative imaginary part (decaying into metal)
        cos_t = torch.where(cos_t.imag < 0, -cos_t, cos_t)
    c1 = torch.complex(torch.tensor(n1, dtype=ti.dtype, device=ti.device), torch.zeros_like(nn))
    rs = (c1 * cos_i - n2 * cos_t) / (c1 * cos_i + n2 * cos_t)
    rp = (n2 * cos_i - c1 * cos_t) / (n2 * cos_i + c1 * cos_t)
    r_s = (rs.abs() ** 2).real
    r_p = (rp.abs() ** 2).real
    rho = rp / (rs + 1e-12j)
    return {"R_s": r_s, "R_p": r_p, "rho": rho, "dphi": torch.angle(rho),
            "phi_s": torch.angle(rs), "phi_p": torch.angle(rp)}


def specular_dolp(r_s: torch.Tensor, r_p: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """|DoLP| of specularly reflected UNPOLARIZED light: |Rp−Rs|/(Rp+Rs).

    For metals this rises steeply near grazing incidence; roughness/multiple
    scattering reduce it (modelled separately via depolarizer in mueller.py).
    """
    return ((r_p - r_s).abs()) / (r_p + r_s + eps)


def dolp_for_material(material: str, theta_i_deg: torch.Tensor,
                      rgb: bool = False):
    """Convenience: conductor DoLP at given incidence angles.

    rgb=True returns per-channel dict (spectral dispersion of copper).
    """
    ti = torch.deg2rad(torch.as_tensor(theta_i_deg, dtype=torch.float32))
    if material not in MATERIAL_IOR or MATERIAL_IOR[material] is None:
        raise ValueError(f"unknown material {material}; use fresnel_dielectric")
    table = MATERIAL_IOR[material]
    if not rgb:
        n, k = table["G"]  # green channel as photopic proxy
        out = fresnel_conductor(ti, 1.0, n, k)
        return specular_dolp(out["R_s"], out["R_p"])
    res = {}
    for ch in ("R", "G", "B"):
        n, k = table[ch]
        out = fresnel_conductor(ti, 1.0, n, k)
        res[ch] = specular_dolp(out["R_s"], out["R_p"])
    return res
