"""Unit tests for the physics core.  Run:  python -m pytest tests -q
(or directly: python tests/test_physics.py)"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.optics.mueller import mueller_linear_polarizer, apply_mueller, transmitted_intensity
from src.optics.stokes import stokes_from_4angle, dolp, aolp, wrap_angle_diff
from src.optics.fresnel import fresnel_dielectric, fresnel_conductor, specular_dolp
from src.polarization.virtual_analyzer import VirtualAnalyzer, periodicity_residual
from src.polarization.psrf import psrf_from_stack, psrf_eval
from src.geometry.cylinder import cylinder_normals, estimate_axis_and_radius


def approx(a, b, tol=1e-5):
    assert abs(a - b) < tol, f"{a} != {b}"


def test_malus_law_consistency():
    """Unpolarized light through polarizer → half intensity, no polarization."""
    s = torch.tensor([[[2.0]], [[0.0]], [[0.0]], [[0.0]]])   # S=[2,0,0,0]
    m = mueller_linear_polarizer(torch.tensor(0.37))
    out = apply_mueller(s, m)
    approx(float(out[0, 0, 0]), 1.0)                          # I = S0/2
    approx(float(out[1, 0, 0]), math.cos(0.74))               # S1' = cos2θ
    approx(float(out[2, 0, 0]), math.sin(0.74), 1e-5)


def test_fully_polarized_malus():
    """S=[1,1,0,0] (horizontal). I(θ)=cos²θ."""
    s = torch.tensor([[[1.0]], [[1.0]], [[0.0]], [[0.0]]])
    for deg in (0, 30, 60, 90):
        th = math.radians(deg)
        i = float(transmitted_intensity(s, torch.tensor(th)))
        approx(i, math.cos(th) ** 2, 1e-6)


def test_virtual_analyzer_matches_mueller():
    h, w = 8, 8
    s3 = torch.rand(3, h, w) + 0.1
    va_closed = VirtualAnalyzer(use_mueller=False)
    va_muell = VirtualAnalyzer(use_mueller=True)
    for ang in (0, 22.5, 77.0, 150.0):
        a = va_closed(s3, ang)
        b = va_muell(s3, ang)
        assert torch.allclose(a, b, atol=1e-6)


def test_periodicity_180():
    s3 = torch.rand(3, 16, 16)
    r = periodicity_residual(s3)
    assert r < 1e-6, "I(θ) must equal I(θ+180°)"


def test_stokes_recovery_roundtrip():
    """Render stack from known Stokes via Malus; recover Stokes exactly."""
    h, w = 10, 12
    s0 = torch.rand(h, w) * 2 + 0.5
    rho = torch.rand(h, w) * 0.6
    phi = torch.rand(h, w) * math.pi
    s1 = s0 * rho * torch.cos(2 * phi)
    s2 = s0 * rho * torch.sin(2 * phi)
    angs = [0.0, 45.0, 90.0, 135.0]
    stack = torch.stack([0.5 * (s0 + math.cos(2 * math.radians(a)) * s1
                                + math.sin(2 * math.radians(a)) * s2) for a in angs])
    rec = stokes_from_4angle(*stack)
    assert torch.allclose(rec[0], s0, atol=1e-5)
    assert torch.allclose(rec[1], s1, atol=1e-5)
    assert torch.allclose(rec[2], s2, atol=1e-5)
    d = dolp(rec)
    assert torch.allclose(d, rho.clamp(0, 1), atol=1e-4)


def test_aolp_wrapping():
    a = torch.tensor([[math.radians(179)]])
    b = torch.tensor([[math.radians(-1)]])   # same physical AoLP modulo 180°
    err = wrap_angle_diff(a, b)
    assert float(err) < 1e-5


def test_brewster_dielectric():
    """Brewster angle for n=1.5: R_p→0 at θB=atan(1.5)."""
    th_b = math.atan(1.5)
    rs, rp = fresnel_dielectric(torch.tensor(th_b), 1.0, 1.5)
    assert float(rp) < 1e-6
    assert float(rs) > 0.1


def test_conductor_dolp_monotonic_trend():
    """Copper DoLP should grow toward grazing incidence (established physics)."""
    from src.optics.fresnel import CU_IOR
    n, k = CU_IOR["G"]
    th = torch.deg2rad(torch.linspace(1, 85, 40))
    fr = fresnel_conductor(th, 1.0, n, k)
    dl = specular_dolp(fr["R_s"], fr["R_p"])
    assert dl[-1] > dl[5], "DoLP must rise near grazing angles"
    assert bool((dl >= 0).all() and (dl <= 1).all())


def test_psrf_recovers_malus_field():
    """A pure Malus field has only harmonics 0 and 1; a2,b2 ≈ 0."""
    va = VirtualAnalyzer()
    s0 = torch.full((6, 6), 2.0)
    s1 = torch.full((6, 6), 0.8)
    s2 = torch.full((6, 6), -0.3)
    s3 = torch.stack([s0, s1, s2])
    angs = list(range(0, 180, 10))
    stack = torch.stack([va(s3, a) for a in angs])
    coef = psrf_from_stack(stack, angs, n_harm=2)
    approx(float(coef[0].mean()), 1.0, 1e-4)          # a0 = S0/2
    approx(float(coef[1].mean()), 0.4, 1e-4)          # a1 = S1/2
    approx(float(coef[2].mean()), -0.15, 1e-4)        # b1 = S2/2
    assert float(coef[3].abs().max()) < 1e-4 and float(coef[4].abs().max()) < 1e-4
    # evaluation round-trip
    ev = psrf_eval(coef, torch.deg2rad(torch.tensor(33.0)))
    ref = va(s3, 33.0)
    assert torch.allclose(ev, ref, atol=1e-3)


def test_cylinder_normal_geometry():
    mask = torch.zeros(40, 80)
    mask[:, 10:70] = 1.0
    ang, rad, _ = estimate_axis_and_radius(mask)
    n = cylinder_normals(mask, ang, rad)
    # centre column normal should point at camera (nz≈1); edges tilted
    c = n[:, 20, 40]
    e = n[:, 20, 12]
    approx(float(c[2]), 1.0, 0.05)
    assert float(e[2]) < 0.5


def test_admissibility_rho_bounds():
    """latents_to_stokes must satisfy |Ŝ⊥| ≤ Ŝ0 when ρ≤1."""
    from src.inverse.latent_optics import latents_to_stokes
    z = {"D": torch.rand(4, 4), "S": torch.rand(4, 4),
         "rho": torch.rand(4, 4), "phi": torch.rand(4, 4) * 3}
    s = latents_to_stokes(z)
    perp = torch.sqrt(s[1] ** 2 + s[2] ** 2)
    assert bool((perp <= s[0] + 1e-6).all())


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} physics/unit tests passed.")
