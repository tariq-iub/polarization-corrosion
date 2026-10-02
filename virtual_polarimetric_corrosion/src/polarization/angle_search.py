"""Automatic virtual analyzer-angle optimization (rule #26).

Objective:
    J(θ) = α·T(θ) + β·C(θ) + γ·E(θ) − δ·G(θ) − λ·U(θ)
with, evaluated on the VIRTUAL stack Î(θ):
  T : texture preservation   — local variance of Î(θ) (Laplacian energy proxy)
  C : corrosion/background contrast — between mean response inside a corrosion
      mask vs outside (uses provided mask or Otsu-split of Ŝ0)
  E : edge visibility        — mean |∇Î(θ)| along object boundary
  G : glare energy           — fraction of saturated pixels in Î(θ)
  U : uncertainty            — mean uncertainty over high-|Δ| regions
θ* = argmax_{θ∈[0,180°)} J(θ), computed by dense sweep (cheap, differentiable
interpolation available via PSRF coefficients).
"""
from __future__ import annotations

import torch

from .virtual_analyzer import VirtualAnalyzer


def _local_variance(img: torch.Tensor) -> torch.Tensor:
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
    lap = torch.nn.functional.conv2d(img[None, None], k[None, None], padding=1)[0, 0]
    return lap.pow(2)


def _gradient_magnitude(img: torch.Tensor) -> torch.Tensor:
    dx = torch.zeros_like(img); dy = torch.zeros_like(img)
    dx[:, :-1] = img[:, 1:] - img[:, :-1]
    dy[:-1, :] = img[1:, :] - img[:-1, :]
    return torch.sqrt(dx * dx + dy * dy)


def angle_objective(stokes3: torch.Tensor, angles_deg, weights=(1.0, 1.0, 1.0, 1.0, 0.5),
                    corrosion_mask: torch.Tensor | None = None,
                    uncertainty_map: torch.Tensor | None = None,
                    sat_thresh: float = 0.98):
    """Return (angles, J values, components dict). Dense sweep over θ."""
    a, b, g, d, lam = weights
    va = VirtualAnalyzer()
    s0 = stokes3[..., 0, :, :]
    if corrosion_mask is None:
        thr = s0.mean() + s0.std()
        corrosion_mask = (s0 > thr).float()          # documented fallback split
    obj_in = corrosion_mask.sum() + 1e-6
    obj_out = (1 - corrosion_mask).sum() + 1e-6
    js, comps = [], {"T": [], "C": [], "E": [], "G": [], "U": []}
    for th in angles_deg:
        i = va(stokes3, th)
        t = _local_variance(i).mean()
        c = (i * corrosion_mask).sum() / obj_in - (i * (1 - corrosion_mask)).sum() / obj_out
        e = _gradient_magnitude(i).mean()
        gsat = (i >= sat_thresh * i.max().clamp(min=1e-6)).float().mean()
        u = 0.0 if uncertainty_map is None else (uncertainty_map * _gradient_magnitude(i)).mean()
        jv = a * t / (t.abs().mean() + 1e-9) + b * c / (s0.mean() + 1e-9) \
             + g * e / (e.mean() + 1e-9) - d * gsat - lam * u
        js.append(float(jv))
        for key, val in zip("TCEGU", [t, c, e, gsat, u]):
            comps[key].append(float(val) if torch.is_tensor(val) else val)
    return list(angles_deg), js, comps


def optimal_angle(angles_deg, j_values):
    best = max(range(len(j_values)), key=lambda i: j_values[i])
    return float(angles_deg[best]), j_values[best]
