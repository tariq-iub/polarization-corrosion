"""Synthetic polarimetric renderer (rule #50) + demo dataset generator.

Purpose: validate the physics modules against ground truth we CONTROL —
known normals, roughness, complex IOR, DoLP/AoLP, analyzer angles → known
I0/I45/I90/I135.  Synthetic validation is kept strictly separate from real
corrosion performance claims (docs/EXPERIMENTAL_PROTOCOL.md §S7).

The renderer forward chain per pixel & channel λ∈{R,G,B}:
    unpolarized light E_λ
      → Fresnel conductor reflection at θi(n,k_λ) with roughness depolarization
      → partial diffuse lobe (small residual polarization proxy)
      → Stokes S(λ) = [S0,S1,S2]
      → ideal analyzer at θk  → intensity image
Corrosion classes are simulated as regions with different (n,k,roughness)
PROXIES — labelled "visually annotated corrosion class" analogues, never
claimed as chemistry (rule #14).
"""
from __future__ import annotations

import math

import torch

from ..optics.fresnel import fresnel_conductor, specular_dolp


def render_cartridge(h=96, w=160, seed=0, device="cpu",
                     n_regions=4, circular_polarization=False):
    """Render a synthetic cylindrical cartridge strip with corrosion patches.

    Returns dict:
      rgb01 (3,H,W) sRGB-encoded image (what a normal camera would record),
      seg (H,W) long labels {0 healthy,1 discolor,2 patina,3 pit},
      severity (1,H,W) float in [0,1],
      mask silhouette (H,W),
      normals (3,H,W), theta_i_deg (H,W),
      meas_stokes (3,H,W) TRUE Stokes of reflected light (hardware-equivalent),
      stack (K,H,W) measured intensities at angles [0,45,90,135].
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = torch.device(device)

    def rand(*shape):
        return torch.rand(shape, generator=g).to(dev)

    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32, device=dev),
                            torch.arange(w, dtype=torch.float32, device=dev),
                            indexing="ij")
    # horizontal cylinder centred
    cx = w / 2
    u = ((xx - cx) / (w * 0.42)).clamp(-0.98, 0.98)     # sin wrap angle
    nz = torch.sqrt(1 - u ** 2)
    normals = torch.stack([u, torch.zeros_like(u), nz], dim=0)
    # light 40° off axis in x-z plane
    ldir = torch.tensor([math.sin(math.radians(40)), 0, math.cos(math.radians(40))], device=dev)
    cos_i = (normals * ldir[:, None, None]).sum(0).clamp(0.02, 0.999)
    theta_i = torch.rad2deg(torch.acos(cos_i))

    # material maps by region noise
    seg = torch.zeros(h, w, dtype=torch.long, device=dev)
    for r in range(1, n_regions):
        cy, cxx = int(rand(1) * h), int(rand(1) * w)
        rad = 6 + int(rand(1) * 12)
        m = ((yy - cy) ** 2 + (xx - cxx) ** 2) < rad ** 2
        seg[m] = min(r, 3)
    inside = nz > 0.15
    seg[~inside] = 0

    # optical proxies per class (n,k @green; roughness; albedo tint RGB)
    ior = {"0": (1.00, 2.00), "1": (1.90, 0.90), "2": (1.45, 0.30), "3": (1.70, 1.90)}
    rough_map = {"0": 0.08, "1": 0.30, "2": 0.55, "3": 0.85}
    tint = {"0": (1.0, 0.55, 0.25),   # polished copper
            "1": (0.85, 0.35, 0.18),  # red-brown discoloration
            "2": (0.35, 0.75, 0.55),  # green patina
            "3": (0.25, 0.22, 0.20)}  # dark pitted
    n_g = torch.ones(h, w, device=dev); k_g = torch.ones(h, w, device=dev)
    rough = torch.ones(h, w, device=dev)
    rgb_diff = torch.zeros(3, h, w, device=dev)
    for cls in "0123":
        m = (seg == int(cls)).float()
        nn_, kk_ = ior[cls]
        n_g = n_g * (1 - m) + nn_ * m
        k_g = k_g * (1 - m) + kk_ * m
        rough = rough * (1 - m) + rough_map[cls] * m
        t = tint[cls]
        for ci, tv in enumerate(t):
            rgb_diff[ci] = rgb_diff[ci] * (1 - m) + tv * m

    fr = fresnel_conductor(torch.deg2rad(theta_i.clamp(0, 89.5)), 1.0, n_g, k_g)
    dolp_spec = specular_dolp(fr["R_s"], fr["R_p"])
    rho = dolp_spec * torch.exp(-2.0 * rough)               # approx model
    # AoLP: perpendicular to plane of incidence → here ≈ horizontal (φ≈90°→π/2)
    phi = torch.full_like(theta_i, math.pi / 2)
    # illumination shading (lambert-ish on diffuse + ggx-like lobe on specular)
    diff_shade = cos_i
    vdir = torch.tensor([0.0, 0.0, 1.0], device=dev)
    cos_v = normals[2].clamp(min=0.0)
    spec_shade = torch.exp(-((theta_i - 40.0) ** 2) / (2 * (8 + 60 * rough) ** 2))
    e_int = 0.9
    d_int = e_int * rgb_diff * diff_shade[None] * (1 - spec_shade[None] * 0.3)
    s_int = e_int * 0.8 * spec_shade[None]                  # near-neutral highlight
    s0 = d_int.sum(0) + s_int.sum(0)
    s1 = s0 * rho * torch.cos(2 * phi)
    s2 = s0 * rho * torch.sin(2 * phi)
    meas_stokes = torch.stack([s0, s1, s2], dim=0)
    angles = [0.0, 45.0, 90.0, 135.0]
    stack = torch.stack([0.5 * (s0 + math.cos(2 * math.radians(a)) * s1
                                + math.sin(2 * math.radians(a)) * s2) for a in angles])
    # camera RGB = raw Stokes-independent total intensity (single exposure!)
    rgb_lin = (d_int + s_int).clamp(0, 1.05)
    # clip highlights realistically
    rgb_lin = torch.minimum(rgb_lin, torch.full_like(rgb_lin, 1.0))
    a = 0.055
    rgb01 = torch.where(rgb_lin <= 0.0031308, rgb_lin * 12.92,
                        (1 + a) * rgb_lin ** (1 / 2.4) - a).clamp(0, 1)
    sev = (rho.new_zeros(1, h, w) + (seg > 0).float().mean(0, keepdim=True) * 0.5
           + rough[None] * 0.5).clamp(0, 1)
    return {"rgb01": rgb01, "seg": seg, "severity": sev, "mask": inside.float(),
            "normals": normals, "theta_i_deg": theta_i,
            "meas_stokes": meas_stokes, "stack": stack, "angles": angles,
            "roughness_gt": rough, "dolp_gt": (s1.pow(2) + s2.pow(2)).sqrt() / (s0 + 1e-6)}


def make_demo_dataset(root="data/demo", n_train=24, n_val=6, n_test=6,
                      size=(96, 160), seed=0):
    """Write NPZ demo dataset so pipelines run out-of-the-box (synthetic only!)."""
    import os
    os.makedirs(root, exist_ok=True)
    counts = {"train": n_train, "val": n_val, "test": n_test}
    for split, n in counts.items():
        items = []
        for i in range(n):
            d = render_cartridge(h=size[0], w=size[1], seed=seed + hash(split) % 1000 + i)
            items.append({k: (v.cpu().numpy() if torch.is_tensor(v) else v)
                          for k, v in d.items()})
        keys = items[0].keys()
        arrs = {k: [it[k] for it in items] for k in keys}
        flat = {}
        for k, vs in arrs.items():
            if isinstance(vs[0], list):
                continue
            flat[k] = torch.tensor(vs).numpy() if torch.is_tensor(vs[0]) else \
                __import__("numpy").array(vs)
        __import__("numpy").savez(os.path.join(root, f"{split}.npz"), **flat)
    return root
