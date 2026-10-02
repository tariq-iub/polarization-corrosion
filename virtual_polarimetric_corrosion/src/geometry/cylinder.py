"""Cylindrical cartridge geometry prior (rules #15/#16).

A cartridge is approximately a cylinder; planar assumptions are inadequate.
Given a binary silhouette mask and an estimated axis direction we derive, for
each silhouette row/column, the local surface normal under the generalized
cylinder assumption (axis along ŷ, radius R fitted from silhouette width):

    x(u) = R sin u,  n = (sin u, 0, cos u),  u ∈ (−π/2, π/2) on visible face

so that incidence/viewing angles follow from n and known camera/light rays.
This constrains θi, expected highlight location (specular condition n·h≈1)
and Fresnel coefficients → improves virtual polarimetric estimation (RQ7).

All quantities are *priors/proxies*; occlusion, rim lighting and headspace
curvature violate the ideal-cylinder model (see LIMITATIONS.md).
"""
from __future__ import annotations

import math

import torch


def estimate_axis_and_radius(mask: torch.Tensor):
    """PCA of silhouette coordinates → axis angle + half-width radius proxy.

    Args: mask (H,W) float in {0,1}. Returns (axis_angle_rad, radius_px, center).
    """
    ys, xs = torch.nonzero(mask > 0.5, as_tuple=True)
    if xs.numel() < 16:
        raise ValueError("silhouette too small")
    pts = torch.stack([xs.float(), ys.float()], dim=-1)
    mean = pts.mean(0)
    cov = (pts - mean).T @ (pts - mean) / pts.shape[0]
    evals, evecs = torch.linalg.eigh(cov.double())
    axis = evecs[:, -1].float()                       # principal direction
    ang = torch.atan2(axis[1], axis[0])
    # project onto minor axis → half-extent ≈ radius proxy
    minor = evecs[:, 0].float()
    proj = (pts - mean) @ minor
    radius = proj.abs().max() * 0.5
    return float(ang), float(radius), mean


def cylinder_normals(mask: torch.Tensor, axis_angle_rad: float, radius_px: float,
                     eps: float = 1e-6) -> torch.Tensor:
    """Per-pixel (3,H,W) unit normals under the visible-cylinder assumption.

    Coordinate frame: u = signed distance along image minor axis (perpendicular
    to cartridge axis, inside silhouette); sinθ_geo = u/R; n_z = cos(asin(u/R)).
    Pixels outside mask get n=(0,0,1) (fronto-parallel fallback) — documented.
    """
    h, w = mask.shape
    dev = mask.device
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32, device=dev),
                            torch.arange(w, dtype=torch.float32, device=dev),
                            indexing="ij")
    pts = torch.stack([xx, yy], dim=-1)
    ca, sa = math.cos(axis_angle_rad), math.sin(axis_angle_rad)
    minor = torch.tensor([-sa, ca], dtype=torch.float32, device=dev)
    # center from mask centroid
    m = mask.clamp(min=0)
    total = m.sum() + eps
    cx = (m * xx).sum() / total
    cy = (m * yy).sum() / total
    u = (pts @ minor)[..., 0] - (torch.tensor([cx, cy], device=dev) @ minor)
    s = (u / max(radius_px, eps)).clamp(-1 + eps, 1 - eps)   # sin of wrap angle
    c = torch.sqrt(1 - s * s)
    nx = s * (-sa)          # normal's x component along image minor axis dir
    ny = s * (ca)
    n = torch.stack([nx, ny, c], dim=-1)
    inside = (mask > 0.5)
    fallback = torch.zeros_like(n)
    fallback[..., 2] = 1.0
    n = torch.where(inside[..., None], n, fallback)
    return n.permute(2, 0, 1)                          # (3,H,W)


def incidence_angles(normals: torch.Tensor, view_dir=(0.0, 0.0, 1.0),
                     light_dir=(0.577, 0.0, 0.577)):
    """Cosine terms for θi (light) and θv (view) given (3,H,W) normals.

    Default light_dir ~ 35° off-axis (typical ring/dome illuminator proxy).
    Returns dict(theta_i_deg, theta_v_deg, half_vec).
    """
    v = torch.tensor(view_dir, dtype=normals.dtype, device=normals.device)
    l = torch.tensor(light_dir, dtype=normals.dtype, device=normals.device)
    l = l / l.norm()
    v = v / v.norm()
    cos_i = (normals * l[:, None, None]).sum(0).clamp(-1, 1)
    cos_v = (normals * v[:, None, None]).sum(0).clamp(-1, 1)
    h = l + v
    h = h / h.norm()
    cos_h = (normals * h[:, None, None]).sum(0).clamp(-1, 1)
    return {
        "theta_i_deg": torch.rad2deg(torch.acos(cos_i)),
        "theta_v_deg": torch.rad2deg(torch.acos(cos_v)),
        "n_dot_h": cos_h,
    }


def specular_band_prediction(normals, light_dir=(0.577, 0.0, 0.577), tol_deg=12.0):
    """Where would a mirror highlight sit?  Condition: n ≈ half-vector(l,v).

    Used by robustness/uncertainty modules: pixels near the predicted band but
    NOT bright ⇒ geometry mismatch; bright pixels far from band ⇒ diffuse
    glare (dirt) or wrong light estimate → raises uncertainty (rule #24).
    """
    d = incidence_angles(normals, light_dir=light_dir)
    band = (torch.rad2deg(torch.acos(d["n_dot_h"].clamp(-1, 1))) < tol_deg).float()
    return band
