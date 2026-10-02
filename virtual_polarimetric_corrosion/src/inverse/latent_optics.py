"""Latent optical-state estimation (Module C, rules #12/#13).

Per-pixel latent vector:
    Z = [D, S, n_x, n_y, n_z, r(roughness), η(ior proxy), k(extinction proxy),
         ρ(polarization magnitude), φ(polarization orientation), g(glare), u(uncert)]

Physics-informed baseline estimator (no learning): derives Z from RGB +
cylinder prior + Fresnel tables.  The learned estimator (models/latent_head.py)
outputs the same schema with bounded activations; both feed Module D/E.

Explicit honesty statement: several entries (η, k, φ absolute frame) are NOT
identifiable from a single RGB image — they are regularised proxies drawn
toward physically plausible ranges via priors, and their posterior spread is
reported by the uncertainty module / counterfactual ensemble (rule #25).
"""
from __future__ import annotations

import math

import torch

from ..optics.fresnel import MATERIAL_IOR, fresnel_conductor, specular_dolp
from ..optics.microfacet import f0_metal
from ..optics.reflection import dichromatic_decompose, srgb_to_linear

LATENT_SCHEMA = ["D", "S", "nx", "ny", "nz", "roughness", "eta", "kappa",
                 "rho", "phi", "glare", "u"]


def physics_baseline_latents(rgb01: torch.Tensor, normals: torch.Tensor,
                             theta_i_deg: torch.Tensor, material="copper"):
    """Deterministic non-learned Z estimate for ablation & warm-start.

    Args:
        rgb01: (3,H,W) sRGB-encoded float in [0,1]
        normals: (3,H,W) unit normals (cylinder prior or SfS proxy)
        theta_i_deg: (H,W) incidence angle map
    Returns dict of (H,W) tensors keyed by LATENT_SCHEMA.
    """
    lin = srgb_to_linear(rgb01)
    dec = dichromatic_decompose(lin)
    d_int = dec["diffuse"].mean(0)
    s_int = dec["specular"].mean(0)
    table = MATERIAL_IOR[material]
    ng, kg = table["G"]
    fr = fresnel_conductor(torch.deg2rad(theta_i_deg.clamp(0, 89.9)), 1.0, ng, kg)
    rho_spec = specular_dolp(fr["R_s"], fr["R_p"])
    # roughness proxy: local intensity variance of diffuse lobe (documented proxy)
    dx = torch.zeros_like(d_int)
    dy = torch.zeros_like(d_int)
    dy[:, :-1] = d_int[1:, :] - d_int[:-1, :]
    dx[:-1, :] = d_int[:, 1:] - d_int[:, :-1]
    grad_mag = torch.sqrt(dx * dx + dy * dy)
    blur = torch.nn.functional.avg_pool2d(grad_mag[None, None], 7, 1, 3)[0, 0]
    rough = (blur / (blur.max() + 1e-6)).clamp(0.02, 0.95)
    # depolarisation by roughness (see microfacet.specular_polarization approx.)
    rho = rho_spec * torch.exp(-2.0 * rough)
    # AoLP proxy: perpendicular to plane-of-incidence projection → use normal
    phi = torch.atan2(normals[1], normals[0])          # documented proxy
    glare = dec["glare"][0]
    eta = torch.full_like(d_int, ng)
    kap = torch.full_like(d_int, kg)
    u = (glare * 0.5 + (1.0 - rho) * 0.25 + rough * 0.25).clamp(0, 1)
    return {"D": d_int, "S": s_int,
            "nx": normals[0], "ny": normals[1], "nz": normals[2],
            "roughness": rough, "eta": eta, "kappa": kap,
            "rho": rho, "phi": phi, "glare": glare, "u": u}


def latents_to_stokes(z: dict) -> torch.Tensor:
    """Convert latent Z → ESTIMATED Stokes field [Ŝ0, Ŝ1, Ŝ2] (3,H,W).

    Ŝ0 = D + S ; Ŝ1 = ρ·(D cos2φ_d + S cos2φ) ... simplified consistent form:
    we treat the *net* polarization magnitude ρ acting along orientation φ:
        Ŝ1 = Ŝ0 · ρ · cos 2φ ;  Ŝ2 = Ŝ0 · ρ · sin 2φ
    This keeps |Ŝ| ≤ S0 automatically when ρ≤1 (physical admissibility, rule #20).
    """
    s0 = (z["D"] + z["S"]).clamp(min=0)
    rho = z["rho"].clamp(0, 1)
    s1 = s0 * rho * torch.cos(2 * z["phi"])
    s2 = s0 * rho * torch.sin(2 * z["phi"])
    return torch.stack([s0, s1, s2], dim=0)
