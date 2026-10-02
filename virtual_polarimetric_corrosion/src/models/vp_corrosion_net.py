"""VP-CorrosionNet (Modules F–I, rules #19/#20) — lightweight, CPU-friendly.

Dataflow:
    RGB encoder (depthwise-separable U-Net-ish trunk)
      → Optical Latent Head   (predicts bounded Z fields; Module C learned path)
      → differentiable Virtual Polarimetric Layer (VirtualAnalyzer on Ŝ(Z))
      → PSRF Encoder (harmonic coefficients → 5-channel embedding)
      → multimodal fusion (RGB + Lab/HSV + texture + polarization + geometry + U)
      → multi-scale decoder
      → heads: segmentation | severity | uncertainty (+ optional PittingHead)

Color-space options (user requirement): 'lab', 'hsv', 'none' — configurable so
experiments can compare RGB-only vs CIELAB vs HSV augmentation fairly.

Physics layers are NOT frozen black boxes: they participate in training through
L_rec / L_pol / L_phys (see experiments/losses.py).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..polarization.virtual_analyzer import VirtualAnalyzer, DEFAULT_STACK_ANGLES
from ..polarization.psrf import psrf_from_stack


# ---------------------------------------------------------------- utilities --
def rgb_to_lab(rgb01: torch.Tensor) -> torch.Tensor:
    """Differentiable sRGB→CIELAB (D65). Standard matrix pipeline."""
    a = 0.055
    lin = torch.where(rgb01 <= 0.04045, rgb01 / 12.92, ((rgb01 + a) / (1 + a)) ** 2.4)
    m = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]], device=rgb01.device)
    xyz = torch.einsum("ij,...jhw->...ihw", m, lin)
    ref = torch.tensor([0.95047, 1.0, 1.08883], device=rgb01.device).view(3, 1, 1)
    f = xyz / ref
    e = 216 / 24389.0
    k = 24389 / 27.0
    fp = torch.where(f > e, f.clamp(min=e) ** (1 / 3), (k * f + 16) / 116)
    l = 116 * fp[1] - 16
    aa = 500 * (fp[0] - fp[1])
    bb = 200 * (fp[1] - fp[2])
    return torch.stack([l / 100.0, aa / 128.0, bb / 128.0], dim=-3)


def rgb_to_hsv(rgb01: torch.Tensor) -> torch.Tensor:
    """Differentiable RGB→HSV; H encoded via sin/cos to respect circularity."""
    mx = rgb01.max(-3, keepdim=True).values
    mn = rgb01.min(-3, keepdim=True).values
    diff = (mx - mn) + 1e-8
    r, g, b = rgb01[..., 0:1], rgb01[..., 1:2], rgb01[..., 2:3]
    h_r = ((g - b) / diff) % 6
    h_g = ((b - r) / diff) + 2
    h_b = ((r - g) / diff) + 4
    h = torch.where(mx == r, h_r, torch.where(mx == g, h_g, h_b)) / 6.0
    s = diff / (mx + 1e-8)
    v = mx
    return torch.cat([torch.cos(2 * math.pi * h), torch.sin(2 * math.pi * h), s, v], dim=-3)


class DWConv(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, 3, stride, 1, groups=c_in, bias=False)
        self.pw = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class ResBottleneck(nn.Module):
    def __init__(self, c, expand=2):
        super().__init__()
        hid = max(8, int(c * expand))
        self.b = nn.Sequential(nn.Conv2d(c, hid, 1, bias=False), nn.BatchNorm2d(hid), nn.SiLU(),
                               nn.Conv2d(hid, hid, 3, 1, 1, groups=hid, bias=False),
                               nn.BatchNorm2d(hid), nn.SiLU(),
                               nn.Conv2d(hid, c, 1, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        return x + self.b(x)


class SEAttention(nn.Module):
    """Lightweight channel attention (squeeze-excite)."""
    def __init__(self, c, red=4):
        super().__init__()
        self.f = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                               nn.Linear(c, max(4, c // red)), nn.SiLU(),
                               nn.Linear(max(4, c // red), c), nn.Sigmoid())

    def forward(self, x):
        return x * self.f(x)[..., None, None]


# --------------------------------------------------- optical latent head ------
LATENT_CH = 12  # schema of inverse.latent_optics.LATENT_SCHEMA


class OpticalLatentHead(nn.Module):
    """Learned Module C: features → physically bounded latent optical fields Z.

    Bounded activations encode the physics constraints:
      D,S ≥ 0 ; roughness∈[0,1]; ρ∈[0,1] (admissibility |Ŝ|≤Ŝ0); φ free angle;
      η∈[0.2,4]; κ∈[0,6]; glare,u∈[0,1].
    Also emits per-pixel log-variance for the heteroscedastic uncertainty head.
    """

    def __init__(self, c_in: int):
        super().__init__()
        self.proj = nn.Conv2d(c_in, LATENT_CH + 1, 1)  # +1 = log σ field

    def forward(self, feat):
        raw = self.proj(feat)
        z_raw, logvar = raw[:, :LATENT_CH], raw[:, LATENT_CH:]
        d = F.softplus(z_raw[:, 0:1])
        s = F.softplus(z_raw[:, 1:2])
        nx = torch.tanh(z_raw[:, 2:3])
        ny = torch.tanh(z_raw[:, 3:4])
        nz = torch.sqrt((1 - nx ** 2 - ny ** 2).clamp(min=1e-4))
        rough = torch.sigmoid(z_raw[:, 4:5])
        eta = 0.2 + 3.8 * torch.sigmoid(z_raw[:, 5:6])
        kap = 6.0 * torch.sigmoid(z_raw[:, 6:7])
        rho = torch.sigmoid(z_raw[:, 7:8])
        phi = z_raw[:, 8:9]
        glare = torch.sigmoid(z_raw[:, 9:10])
        u = torch.sigmoid(z_raw[:, 10:11])
        _ = z_raw[:, 11:12] * 0.0  # reserved slot kept for schema symmetry
        z = {"D": d, "S": s, "nx": nx, "ny": ny, "nz": nz, "roughness": rough,
             "eta": eta, "kappa": kap, "rho": rho, "phi": phi, "glare": glare,
             "u": u, "logvar": logvar}
        return z


def stokes_from_latent(z):
    s0 = (z["D"] + z["S"]).clamp(min=0)
    rho = z["rho"]
    s1 = s0 * rho * torch.cos(2 * z["phi"])
    s2 = s0 * rho * torch.sin(2 * z["phi"])
    return torch.cat([s0, s1, s2], dim=1)          # (B,3,H,W)


class VirtualPolarimetricLayer(nn.Module):
    """Module D/E inside the network: differentiable stack + PSRF embedding."""

    def __init__(self, angles=DEFAULT_STACK_ANGLES, n_harm: int = 2):
        super().__init__()
        self.angles = list(angles)
        self.n_harm = n_harm
        self.va = VirtualAnalyzer()

    def forward(self, stokes_b3hw):
        b = stokes_b3hw.shape[0]
        stacks, coeffs = [], []
        for i in range(b):
            s3 = stokes_b3hw[i]                     # (3,H,W)
            stack = torch.stack([self.va(s3, a) for a in self.angles], dim=0)
            stacks.append(stack)
            coeffs.append(psrf_from_stack(stack, self.angles, self.n_harm))
        coef = torch.stack(coeffs, dim=0)           # (B,C,H,W)
        return coef, torch.stack(stacks, dim=0)     # (B,K,H,W)


class PSRFEncoder(nn.Module):
    def __init__(self, c_coef: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c_coef, c_out, 1, bias=False),
                                 nn.BatchNorm2d(c_out), nn.SiLU(),
                                 DWConv(c_out, c_out))

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------- full net --------
class VPCorrosionNet(nn.Module):
    """Segmentation + severity + uncertainty + pitting heads with virtual
    polarimetric physics layer.  color_space ∈ {'lab','hsv','none'} adds the
    corresponding auxiliary channels to the input (comparison option)."""

    def __init__(self, n_classes: int = 4, base: int = 24,
                 color_space: str = "lab", use_psrf: bool = True,
                 use_uncertainty_head: bool = True, dropout: float = 0.1):
        super().__init__()
        assert color_space in ("lab", "hsv", "none")
        self.color_space = color_space
        c_in = 3 + {"lab": 3, "hsv": 4, "none": 0}[color_space] + 1  # +1 edge ch
        self.stem = nn.Sequential(nn.Conv2d(c_in, base, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(base), nn.SiLU(),
                                  ResBottleneck(base))
        self.down1 = nn.Sequential(DWConv(base, base * 2, 2), ResBottleneck(base * 2))
        self.down2 = nn.Sequential(DWConv(base * 2, base * 4, 2), ResBottleneck(base * 4),
                                   SEAttention(base * 4))
        self.latent_head = OpticalLatentHead(base * 4)
        self.use_psrf = use_psrf
        if use_psrf:
            self.vpol = VirtualPolarimetricLayer()
            n_h = 1 + 2 * self.vpol.n_harm
            self.psrf_enc = PSRFEncoder(n_h, base * 2)
        fuse_c = base * 4 + (base * 2 if use_psrf else 0) + 5  # +5 pol feats
        self.fuse = nn.Sequential(DWConv(fuse_c, base * 4), ResBottleneck(base * 4))
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = nn.Sequential(ResBottleneck(base * 2), ResBottleneck(base * 2))
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = nn.Sequential(ResBottleneck(base), nn.Dropout2d(dropout))
        self.seg = nn.Conv2d(base, n_classes, 1)
        self.sev = nn.Conv2d(base, 1, 1)                       # sigmoid severity
        self.pit = nn.Conv2d(base, 1, 1)                       # PittingHead
        self.use_unc = use_uncertainty_head
        if use_uncertainty_head:
            self.unc = nn.Conv2d(base, 1, 1)                   # softplus U map

    def _aux_channels(self, rgb):
        chans = [rgb]
        if self.color_space == "lab":
            chans.append(rgb_to_lab(rgb))
        elif self.color_space == "hsv":
            chans.append(rgb_to_hsv(rgb))
        dx = torch.zeros_like(rgb[:, :1]); dy = torch.zeros_like(rgb[:, :1])
        dx[..., :-1] = rgb[:, :1, :, 1:] - rgb[:, :1, :, :-1]
        dy[..., :-1, :] = rgb[:, :1, 1:, :] - rgb[:, :1, :-1, :]
        chans.append(torch.sqrt(dx * dx + dy * dy + 1e-8))     # edge channel
        return torch.cat(chans, dim=1)

    def forward(self, rgb01):
        x = self._aux_channels(rgb01)
        e0 = self.stem(x)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        z = self.latent_head(e2)
        stokes = stokes_from_latent(z)                          # (B,3,h,w) estimated!
        feats = [e2]
        if self.use_psrf:
            coef, stack = self.vpol(stokes)
            coef_up = F.interpolate(coef, size=e2.shape[-2:], mode="bilinear", align_corners=False)
            feats.append(self.psrf_enc(coef_up))
        pol_feats = torch.cat([stokes,
                               torch.sin(2 * z["phi"]), torch.cos(2 * z["phi"])], dim=1)
        feats.append(pol_feats)
        h = self.fuse(torch.cat(feats, dim=1))
        d2 = self.dec2(self.up2(h) + e1)
        d1 = self.dec1(self.up1(d2) + e0)
        out = {"seg": self.seg(d1), "severity": torch.sigmoid(self.sev(d1)),
               "pitting": torch.sigmoid(self.pit(d1)),
               "latent": {k: F.interpolate(v, size=rgb01.shape[-2:], mode="bilinear",
                                           align_corners=False) for k, v in z.items()},
               "stokes_hat": F.interpolate(stokes, size=rgb01.shape[-2:], mode="bilinear",
                                           align_corners=False)}
        if self.use_unc:
            out["uncertainty"] = F.softplus(self.unc(d1))
        return out
