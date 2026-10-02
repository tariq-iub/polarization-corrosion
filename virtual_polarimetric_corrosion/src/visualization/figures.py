"""Publication figure generators (rules #61/#62) — matplotlib, SVG+PDF+PNG≥300dpi.

Scripts here produce the *framework* figures that do not depend on experimental
results (Figs 1–5) and template-driven result figures (Figs 6–15) that read
from results/*.csv so they can be regenerated from real measurements only.
Empty/placeholder data is NEVER fabricated into these figures: if a required
CSV is missing the script prints what to run first and exits without plotting
fake numbers.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

FIGDIR = os.environ.get("VPC_FIGDIR", "figures")


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("svg", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGDIR, f"{name}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure1_architecture():
    """Clean block diagram — VP-CorrosionNet framework (rule #61)."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_axis_off()
    main = [
        "Conventional RGB image",
        "Radiometric linearization (sRGB→linear)",
        "Geometry / cartridge cylinder prior",
        "Diffuse–specular separation (dichromatic)",
        "Latent optical state  Z = [D,S,n,r,η,κ,ρ,φ,g,u]",
        "Physics-constrained virtual polarimetric camera\n(Stokes estimation Ŝ0,Ŝ1,Ŝ2)",
        "Virtual analyzer  Î(θ)=½[Ŝ0+Ŝ1cos2θ+Ŝ2sin2θ]",
        "Virtual polarization stack {Î(θk)}",
        "PSRF — circular harmonics  a0,a1,b1,(a2,b2)",
        "Fusion: RGB + Lab/HSV + texture + pol. + geometry + U",
        "VP-CorrosionNet multi-scale decoder",
    ]
    outs = ["Corrosion mask", "Pitting map", "Severity map", "Uncertainty map"]
    y = 0.97
    boxes = []
    for i, txt in enumerate(main):
        h = 0.055 if "\n" not in txt else 0.075
        b = plt.Rectangle((0.18, y - h), 0.64, h, fc="#f2f2f2", ec="black", lw=1.0)
        ax.add_patch(b)
        ax.text(0.5, y - h / 2, txt, ha="center", va="center", fontsize=8.5)
        boxes.append((y, y - h))
        if i < len(main) - 1:
            ax.annotate("", xy=(0.5, y - h - 0.008), xytext=(0.5, y - h),
                        arrowprops=dict(arrowstyle="-|>", lw=1.0))
        y -= h + 0.018
    # outputs row
    for j, otxt in enumerate(outs):
        x0 = 0.05 + j * 0.24
        b = plt.Rectangle((x0, y - 0.05), 0.2, 0.05, fc="#dce6f1", ec="black", lw=1.0)
        ax.add_patch(b)
        ax.text(x0 + 0.1, y - 0.025, otxt, ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(0.5, y - 0.002), xytext=(0.5, boxes[-1][1]),
                arrowprops=dict(arrowstyle="->", lw=1.0))
    # training-only branch (right side)
    tb = ["Physical polarization camera\n(real hardware, training only)",
          "Measured stack I0/I45/I90/I135",
          "Measured Stokes / DoLP / AoLP",
          "Teacher polarimetric representation",
          "Knowledge distillation  L_distill"]
    ty = 0.97
    tboxes = []
    for txt in tb:
        h = 0.06 if "\n" not in txt else 0.075
        b = plt.Rectangle((0.845, ty - h), 0.15, h, fc="#fdf3e3", ec="black", lw=0.8, ls="--")
        ax.add_patch(b)
        ax.text(0.92, ty - h / 2, txt, ha="center", va="center", fontsize=6.2)
        tboxes.append((ty, ty - h))
        ty -= h + 0.015
    for i in range(len(tboxes) - 1):
        ax.annotate("", xy=(0.92, tboxes[i + 1][0] - 0.001), xytext=(0.92, tboxes[i][1]),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, linestyle="--"))
    ax.annotate("", xy=(0.82, 0.30), xytext=(0.845, tboxes[-1][1] + 0.03),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, linestyle="--"))
    ax.text(0.5, 0.995, "Figure 1 — VP-CorrosionNet: software-defined polarimetric\ncorrosion assessment (solid = deployment path, dashed = training-only supervision)",
            ha="center", fontsize=9, weight="bold")
    _save(fig, "fig01_architecture")


def figure2_physical_vs_virtual(stokes3_demo=None):
    """Side-by-side: physical-capture schematic vs virtual pipeline outputs."""
    from src.polarization.virtual_analyzer import VirtualAnalyzer
    if stokes3_demo is None:
        # deterministic physics demo field (NOT measurement data): synthetic
        # Malus-consistent field used only to illustrate the transform itself.
        h, w = 64, 96
        yy, xx = np.mgrid[0:h, 0:w].astype("float32")
        s0 = (0.5 + 0.3 * np.sin(xx / 9)).astype(np.float32)
        rho = 0.35 + 0.25 * np.cos(yy / 11)
        phi = (xx / w) * np.pi / 2
        s1 = s0 * rho * np.cos(2 * phi); s2 = s0 * rho * np.sin(2 * phi)
        stokes3_demo = torch.tensor(np.stack([s0, s1, s2]))
    va = VirtualAnalyzer()
    angles = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]
    ims = [va(stokes3_demo, a).numpy() for a in angles]
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for ax, im, a in zip(axes.ravel(), ims, angles):
        ax.imshow(im, cmap="gray"); ax.set_title(f"virtual Î({a}°)", fontsize=8)
        ax.axis("off")
    fig.suptitle("Figure 2 — Virtual polarimetric stack generated from one estimated Stokes field\n"
                 "(illustrative synthetic field; NOT measured data)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "fig02_virtual_stack")


def figure3_fresnel_physics():
    """DoLP vs incidence angle for copper channels + corrosion proxies (rule #40)."""
    from src.optics.fresnel import dolp_for_material
    th = np.linspace(0, 89, 180)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    cu = dolp_for_material("copper", th, rgb=True)
    for ch, col in zip(("R", "G", "B"), ("red", "green", "blue")):
        axes[0].plot(th, cu[ch].numpy(), color=col, lw=1.5, label=f"Cu {ch}")
    for mat, style in (("bronze", "--"), ("cu2o_like", ":"), ("chloride_like", "-.")):
        v = dolp_for_material(mat, th)
        axes[1].plot(th, v.numpy(), style, lw=1.5, label=mat)
    v = dolp_for_material("copper", th)
    axes[1].plot(th, v.numpy(), "-", color="k", lw=1.5, label="copper")
    for ax in axes:
        ax.set_xlabel("incidence angle θi (deg)"); ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("|DoLP| of specular reflection")
    axes[0].set_title("(a) spectral dispersion — per-channel Fresnel (Cu)")
    axes[1].set_title("(b) material-proxy comparison")
    fig.suptitle("Figure 3 — Conductor Fresnel polarization response (established physics)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, "fig03_fresnel")


def figure4_analyzer_sweep(stokes3=None):
    """Î(θ) curve + PSRF harmonic decomposition + periodicity check."""
    from src.polarization.virtual_analyzer import VirtualAnalyzer
    from src.polarization.psrf import psrf_from_stack, psrf_eval
    h, w = 48, 48
    if stokes3 is None:
        yy, xx = np.mgrid[0:h, 0:w].astype("float32")
        s0 = np.ones((h, w), np.float32)
        rho = 0.4 + 0.3 * (xx / w)
        phi = (yy / h) * np.pi / 2
        stokes3 = torch.tensor(np.stack([s0,
                                         s0 * rho * np.cos(2 * phi),
                                         s0 * rho * np.sin(2 * phi)]))
    va = VirtualAnalyzer()
    th = np.arange(0, 180, 5)
    pix = (24, 24)
    curve = np.array([float(va(stokes3, t)[pix[0], pix[1]]) for t in th])
    stack = torch.stack([va(stokes3, t) for t in th])
    coef = psrf_from_stack(stack, list(th), n_harm=2)
    recon = psrf_eval(coef, torch.deg2rad(torch.tensor(th))).numpy()[:, pix[0], pix[1]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(th, curve, "o", ms=3, label="sampled points (actual evaluations)")
    axes[0].plot(th, recon, "-", lw=1.2, label="PSRF fit (harmonic model)")
    axes[0].set_xlabel("analyzer angle θ (deg)"); axes[0].set_ylabel("Î(θ)")
    axes[0].set_title("Virtual analyzer sweep at one pixel")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    names = ["a0", "a1", "b1", "a2", "b2"]
    means = coef.mean(dim=(1, 2)).numpy()
    axes[1].bar(names, means, color=["#888", "#2b7bba", "#2b7bba", "#d98843", "#d98843"])
    axes[1].set_ylabel("mean coefficient")
    axes[1].set_title("PSRF circular-harmonic coefficients\n(blue: 1st harmonic ≙ linear pol.; orange: higher-order)")
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("Figure 4 — Virtual analyzer sweep & PSRF (markers=evaluations, line=model)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    _save(fig, "fig04_analyzer_sweep")


def figure5_stokes_maps(rgb01, out_prefix="fig05"):
    """Ŝ0 / DoLP_hat / AoLP_hat maps from the physics baseline estimator."""
    from src.inverse.latent_optics import physics_baseline_latents, latents_to_stokes
    from src.geometry.cylinder import estimate_axis_and_radius, cylinder_normals, incidence_angles
    from src.optics.stokes import dolp, aolp_safe
    h, w = rgb01.shape[-2:]
    mask = (rgb01.mean(0) > 0.15).float()
    ang, rad, _ = estimate_axis_and_radius(mask)
    normals = cylinder_normals(mask, ang, rad)
    inc = incidence_angles(normals)["theta_i_deg"]
    z = physics_baseline_latents(rgb01, normals, inc)
    s = latents_to_stokes(z)
    d = dolp(s); phi = aolp_safe(s)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].imshow(s[0].numpy(), cmap="gray"); axes[0].set_title("Ŝ0 (total)")
    im1 = axes[1].imshow(d.numpy(), cmap="viridis", vmin=0, vmax=1); axes[1].set_title("DoLP_hat")
    plt.colorbar(im1, ax=axes[1])
    im2 = axes[2].imshow(phi.numpy(), cmap="twilight", vmin=-np.pi / 2, vmax=np.pi / 2)
    axes[2].set_title("AoLP_hat (wrapped)")
    plt.colorbar(im2, ax=axes[2])
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Figure 5 — Estimated (hat-notation) polarimetric maps — NOT measured", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    _save(fig, out_prefix)


def figure9_angle_optimization(results_csv=None):
    """Analyzer-angle objective curve from actual computed J(θ) CSV."""
    if results_csv is None or not os.path.exists(str(results_csv)):
        print("figure9: run scripts/run_angle_search.py first (no fabricated curves).")
        return
    dat = np.genfromtxt(results_csv, delimiter=",", names=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(dat["theta_deg"], dat["J"], "o-", ms=3)
    best = dat["theta_deg"][np.nanargmax(dat["J"])]
    ax.axvline(best, ls="--", c="r", label=f"θ* = {best:.1f}°")
    ax.set_xlabel("virtual analyzer angle θ (deg)"); ax.set_ylabel("objective J(θ)")
    ax.set_title("Figure 9 — Automatic virtual analyzer optimization (measured J values)")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, "fig09_angle_opt")


if __name__ == "__main__":
    figure1_architecture()
    figure2_physical_vs_virtual()
    figure3_fresnel_physics()
    figure4_analyzer_sweep()
    print("figures written to", FIGDIR)
