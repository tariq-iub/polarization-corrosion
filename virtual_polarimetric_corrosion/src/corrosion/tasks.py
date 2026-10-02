"""Corrosion analysis tasks (rules #30/#31): severity, pitting, coverage."""
from __future__ import annotations

import torch


def corroded_area_percentage(seg_pred: torch.Tensor, mask_obj: torch.Tensor | None = None,
                             ignore=255) -> float:
    """Task 6: fraction of object pixels labelled any corrosion class (>0)."""
    cor = (seg_pred > 0) & (seg_pred != ignore)
    if mask_obj is not None:
        cor = cor & (mask_obj > 0.5)
        denom = (mask_obj > 0.5).sum()
    else:
        denom = (seg_pred != ignore).sum()
    return float(cor.sum() / (denom + 1e-6) * 100.0)


def pitting_cues(rgb_lin: torch.Tensor, normals: torch.Tensor | None = None,
                 ksize: int = 5) -> dict:
    """Morphological/geometric pitting cues (colour alone fails for pits).

    * multi-scale Laplacian energy (blob detector proxy)
    * shadow/highlight pair response (DoG on luminance along light direction)
    * normal-discontinuity magnitude (if normals available)
    * local roughness (gradient std over window)
    All are *cues*, not decisions — PittingHead consumes them.
    """
    lum = rgb_lin.mean(0, keepdim=True)[None]
    lap1 = torch.nn.functional.conv2d(lum, _laplacian(), padding=1)
    lap2 = torch.nn.functional.conv2d(lum, _laplacian(dilate=2), padding=2)
    blob = (-lap1).clamp(min=0) * 0.6 + (-lap2).clamp(min=0) * 0.4   # dark-centre pits
    blur = torch.nn.functional.avg_pool2d(lum, ksize, 1, ksize // 2)
    dog = lum - blur                                                  # highlight/shadow pair
    out = {"laplacian_energy": (lap1 ** 2 + lap2 ** 2)[0],
           "dog_pair": dog[0],
           "roughness_local": torch.nn.functional.avg_pool2d(
               (lum - blur).abs(), ksize, 1, ksize // 2)[0]}
    if normals is not None:
        n = normals[None]
        dx = torch.zeros_like(n); dy = torch.zeros_like(n)
        dx[..., :, :-1] = n[..., :, 1:] - n[..., :, :-1]
        dy[..., :-1, :] = n[..., 1:, :] - n[..., :-1, :]
        disc = (dx ** 2 + dy ** 2).sum(1, keepdim=True)
        out["normal_discontinuity"] = torch.nn.functional.avg_pool2d(
            disc, ksize, 1, ksize // 2)[0]
    return out


def _laplacian(dilate: int = 1):
    """4-neighbour Laplacian kernel; dilate=2 uses offset taps (5x5 support)."""
    size = 2 * dilate + 1
    w = torch.zeros(1, 1, size, size)
    c = dilate
    w[0, 0, c, c] = -4.0
    w[0, 0, c - dilate, c] = 1.0
    w[0, 0, c + dilate, c] = 1.0
    w[0, 0, c, c - dilate] = 1.0
    w[0, 0, c, c + dilate] = 1.0
    return w


def severity_from_masks(sev_map: torch.Tensor, seg_pred: torch.Tensor,
                        obj_mask: torch.Tensor | None = None) -> dict:
    """Task 5 image-level severity aggregation: mean/max over corroded area."""
    m = (seg_pred > 0)
    if obj_mask is not None:
        m = m & (obj_mask > 0.5)
    if m.sum() == 0:
        return {"mean_severity": 0.0, "max_severity": 0.0}
    v = sev_map[0][m]
    return {"mean_severity": float(v.mean()), "max_severity": float(v.max())}
