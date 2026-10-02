"""Baselines (rule #34) — deliberately simple, honestly labelled.

* RGBOnlyUNet            : same trunk without any physics/PSRF/uncertainty.
* PseudoPolarizerFilter  : classic 'fake polarizer' image filter used ONLY as a
                           straw-man baseline (brightness/contrast trick).
* CLAHE / Gamma / Highlight-suppression preprocessing + RGBOnlyUNet.
* PhysicsOnlyVirtualPolarizer: non-learned latent estimator (inverse.latent_optics)
                           feeding logistic-regression-style per-pixel classifier
                           on the corrosion feature vector F_p (rule #18).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.vp_corrosion_net import DWConv, ResBottleneck


class RGBOnlyUNet(nn.Module):
    def __init__(self, n_classes=4, base=24, color_space="none"):
        super().__init__()
        self.color_space = color_space
        from ..models.vp_corrosion_net import rgb_to_lab, rgb_to_hsv
        self._lab = rgb_to_lab if color_space == "lab" else None
        self._hsv = rgb_to_hsv if color_space == "hsv" else None
        c_in = 3 + (3 if color_space == "lab" else 0) + (4 if color_space == "hsv" else 0)
        self.stem = nn.Sequential(nn.Conv2d(c_in, base, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(base), nn.SiLU(), ResBottleneck(base))
        self.d1 = nn.Sequential(DWConv(base, base * 2, 2), ResBottleneck(base * 2))
        self.d2 = nn.Sequential(DWConv(base * 2, base * 4, 2), ResBottleneck(base * 4))
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.seg = nn.Conv2d(base, n_classes, 1)
        self.sev = nn.Conv2d(base, 1, 1)

    def forward(self, rgb):
        x = rgb
        if self._lab is not None:
            x = torch.cat([x, self._lab(rgb)], 1)
        elif self._hsv is not None:
            x = torch.cat([x, self._hsv(rgb)], 1)
        e0 = self.stem(x); e1 = self.d1(e0); e2 = self.d2(e1)
        d2 = self.u2(e2) + e1
        d1 = self.u1(d2) + e0
        return {"seg": self.seg(d1), "severity": torch.sigmoid(self.sev(d1))}


def clahe_gray(img: torch.Tensor, clip: float = 2.0, grid: int = 8) -> torch.Tensor:
    """Minimal tile-based CLAHE on luminance (documented approximation of the
    OpenCV algorithm; adequate as a preprocessing baseline)."""
    b, c, h, w = img.shape
    lum = img.mean(1, keepdim=True)
    out = []
    th, tw = h // grid, w // grid
    for i in range(grid):
        for j in range(grid):
            tile = lum[:, :, i * th:(i + 1) * th, j * tw:(j + 1) * tw]
            flat = tile.flatten(2)
            srt, _ = flat.sort(dim=-1)
            cdf = torch.arange(1, srt.shape[-1] + 1, device=img.device).float()[None, :]
            lut_idx = ((srt - srt[..., :1]) / (srt[..., -1:] - srt[..., :1] + 1e-6) * 255).long()
            counts = torch.zeros(b, 256, device=img.device)
            counts.scatter_add_(1, lut_idx.clamp(0, 255).squeeze(1),
                                torch.ones_like(flat).squeeze(1))
            counts = counts.clamp(max=clip * flat.shape[-1] / 256)
            cdfs = counts.cumsum(1)
            cdfs = cdfs / (cdfs[:, -1:] + 1e-6)
            mapv = torch.gather(cdfs, 1, lut_idx.squeeze(1).clamp(0, 255)).unsqueeze(-1)
            res = mapv.reshape(tile.shape)
            out.append(res)
    rows = [torch.cat(out[r * grid:(r + 1) * grid], dim=3) for r in range(grid)]
    newlum = torch.cat(rows, dim=2)
    delta = newlum - lum
    return (img + delta).clamp(0, 1)


def gamma_correct(img: torch.Tensor, gamma: float = 1.2) -> torch.Tensor:
    return img.clamp(min=0) ** (1.0 / gamma)


def naive_highlight_suppress(img_lin: torch.Tensor) -> torch.Tensor:
    """Desaturate near-white pixels toward local median colour (baseline trick)."""
    mx = img_lin.max(1, keepdim=True).values
    med = img_lin.median(1, keepdim=True).values
    glare = (mx > 0.9).float()
    return img_lin * (1 - glare) + med * glare


def pseudo_polarizer_filter(img: torch.Tensor, theta_deg: float = 90.0,
                           strength: float = 0.7) -> torch.Tensor:
    """STRAW-MAN baseline: Malus-like brightness modulation WITHOUT Stokes.

    I_out = I·(1−s) + I·(s·cos²(θ−φ_local)), φ_local = dominant-gradient angle.
    Explicitly NOT polarization physics — included to show what our method must
    beat and to demonstrate why gradient heuristics ≠ polarimetry (rule #72).
    """
    lum = img.mean(1, keepdim=True)
    gy = torch.zeros_like(lum); gx = torch.zeros_like(lum)
    gx[..., :-1] = lum[..., 1:] - lum[..., :-1]
    gy[..., :-1, :] = lum[..., 1:, :] - lum[..., :-1, :]
    phi = torch.atan2(gy, gx)
    malus = torch.cos(torch.deg2rad(theta_deg) - phi) ** 2
    return img * (1 - strength + strength * malus)
