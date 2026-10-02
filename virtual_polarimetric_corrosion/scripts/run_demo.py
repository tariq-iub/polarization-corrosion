"""End-to-end demo pipeline: synthetic data → physics validation → training →
virtual-vs-measured comparison → CSV result tables (no fabricated numbers —
all values are computed live from the synthetic ground truth).

Run from repo root:  python scripts/run_demo.py
Outputs: results/*.csv, figures/*.svg|pdf|png
"""
from __future__ import annotations

import csv
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F

from src.synthetic.renderer import render_cartridge, make_demo_dataset
from src.optics.stokes import stokes_from_4angle, dolp, aolp, wrap_angle_diff
from src.polarization.virtual_analyzer import VirtualAnalyzer, periodicity_residual
from src.polarization.psrf import psrf_from_stack, psrf_eval
from src.polarization.angle_search import angle_objective, optimal_angle
from src.inverse.latent_optics import physics_baseline_latents, latents_to_stokes
from src.geometry.cylinder import cylinder_normals, incidence_angles
from src.models.vp_corrosion_net import VPCorrosionNet
from src.models.distillation import PolarizationTeacher, distill_alignment
from src.experiments.metrics import segmentation_metrics, severity_metrics, wrapped_angular_error_deg
from src.experiments.dataset import CartridgeCorrosionDataset, collate
from src.visualization import figures


def ensure_dirs():
    for d in ("results", "figures", "data/demo"):
        os.makedirs(d, exist_ok=True)


def validate_physics_modules():
    """Rule #50: verify virtual physics against controlled synthetic truth."""
    rows = []
    torch.manual_seed(0)
    for seed in range(6):
        d = render_cartridge(seed=seed)
        s_meas = stokes_from_4angle(d["stack"][0], d["stack"][1], d["stack"][2], d["stack"][3])
        # measured-stokes recovery error vs renderer truth
        err_s0 = float((s_meas[0] - d["meas_stokes"][0]).abs().mean() / d["meas_stokes"][0].mean())
        dolp_m = dolp(s_meas)
        err_dolp = float((dolp_m - d["dolp_gt"]).abs().mean())
        phi_m = aolp(s_meas)
        err_phi = float(wrap_angle_diff(phi_m, torch.full_like(phi_m, math.pi / 2)).mean())
        # periodicity of the virtual analyzer on the TRUE field
        per = periodicity_residual(d["meas_stokes"])
        rows.append(dict(seed=seed, S0_rel_err=err_s0, DoLP_abs_err=err_dolp,
                         AoLP_wrapped_err_rad=err_phi, periodicity_max_resid=per))
    with open("results/physics_validation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print("[physics-validation] wrote results/physics_validation.csv")
    return rows


def run_angle_search_example():
    d = render_cartridge(seed=1)
    angles = list(np.arange(0, 180, 5.0))
    th, js, comps = angle_objective(d["meas_stokes"], angles)
    best, jbest = optimal_angle(th, js)
    with open("results/angle_search_J.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["theta_deg", "J"] + list(comps.keys()))
        for i, t in enumerate(th):
            w.writerow([t, js[i]] + [comps[k][i] for k in comps])
    figures.figure9_angle_optimization("results/angle_search_J.csv")
    print(f"[angle-search] theta*={best}° J={jbest:.4f}")


def train_and_evaluate(epochs=8, lr=3e-4, base=24, color_space="lab"):
    make_demo_dataset("data/demo", n_train=18, n_val=4, n_test=6, seed=42)
    ds_tr = CartridgeCorrosionDataset("data/demo/train.npz")
    ds_te = CartridgeCorrosionDataset("data/demo/test.npz")
    tr_loader = torch.utils.data.DataLoader(ds_tr, batch_size=6, shuffle=True,
                                            collate_fn=collate)
    model = VPCorrosionNet(n_classes=4, base=base, color_space=color_space)
    teacher = PolarizationTeacher(n_classes=4, base=base)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[train] VP-CorrosionNet params={n_par:,} color_space={color_space}")
    step = 0
    for ep in range(epochs):
        model.train()
        tot_l = 0.0
        for b in tr_loader:
            rgb, tgt, stack = b["rgb01"], b["seg"], b["stack"]
            out = model(rgb)
            l_seg = F.cross_entropy(out["seg"], tgt.long(), ignore_index=255)
            l_sev = F.smooth_l1_loss(out["severity"], b["severity"])
            l_pol = distill_alignment(out["stokes_hat"],
                                      stokes_from_4angle(stack[:, 0], stack[:, 1],
                                                         stack[:, 2], stack[:, 3]))
            loss = l_seg + 0.5 * l_sev + 0.3 * l_pol
            opt.zero_grad(); loss.backward(); opt.step()
            tot_l += float(loss); step += 1
        print(f"  epoch {ep}: loss={tot_l/len(tr_loader):.4f}")
    # ---------------- evaluation: Config B (RGB→virtual) vs Config A (real) ---
    model.eval(); teacher.eval()
    rows = []
    with torch.no_grad():
        for i in range(len(ds_te)):
            smp = ds_te[i]
            rgb = smp["rgb01"][None]
            out = model(rgb)
            pred_b = out["seg"].argmax(1)[0]
            mB = segmentation_metrics(pred_b, smp["seg"], 4)
            stk = smp["stack"][None]
            ot = teacher(stk)
            pred_a = ot["seg"].argmax(1)[0]
            mA = segmentation_metrics(pred_a, smp["seg"], 4)
            # polarization reconstruction quality (virtual vs measured truth)
            sh = out["stokes_hat"][0]
            mt = smp["meas_stokes"]
            scale = mt[0].mean().clamp(min=1e-6)
            dolp_hat = (sh[1] ** 2 + sh[2] ** 2).sqrt() / (sh[0] + 1e-6)
            err_dolp = float((dolp_hat - smp["dolp_gt"]).abs().mean())
            err_s0 = float(((sh[0] - mt[0]) / scale).abs().mean())
            sev = severity_metrics(out["severity"][0], smp["severity"])
            rows.append(dict(idx=i, mIoU_ConfigB_virtual=float(mB["mIoU"]),
                             mIoU_ConfigA_physical=float(mA["mIoU"]),
                             acc_B=float(mB["accuracy"]), acc_A=float(mA["accuracy"]),
                             dolp_hat_abs_err=err_dolp, s0_rel_err=err_s0,
                             severity_MAE=sev["MAE"]))
    with open("results/config_B_vs_A.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    mb = float(np.mean([r["mIoU_ConfigB_virtual"] for r in rows]))
    ma = float(np.mean([r["mIoU_ConfigA_physical"] for r in rows]))
    print(f"[eval] mean mIoU — Config B (virtual, RGB-only): {mb:.3f} | "
          f"Config A (physical teacher): {ma:.3f}")
    # runtime benchmark (rule #44)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(10):
            model(torch.rand(1, 3, 96, 160))
    dt = (time.perf_counter() - t0) / 10
    with open("results/runtime.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "device", "input", "params", "latency_ms_mean", "fps_est"])
        w.writerow(["VP-CorrosionNet", "CPU", "96x160", n_par, round(dt * 1000, 2),
                    round(1 / dt, 1)])
    return model, rows


def uncertainty_demo(model):
    """CPE + heuristic U on one synthetic frame; write table + figure inputs."""
    from src.polarization.uncertainty import counterfactual_latents, cpe_stack_summary
    d = render_cartridge(seed=7)
    mask = d["mask"]
    ang, rad = 0.0, float(mask.sum() / mask.shape[0]) / 2.0
    normals = d["normals"]
    z = physics_baseline_latents(d["rgb01"], normals, d["theta_i_deg"])
    va = VirtualAnalyzer()
    states = counterfactual_latents(z, k_states=16)
    stacks = torch.stack([va(latents_to_stokes(zz), 45.0) for zz in states])
    summ = cpe_stack_summary(stacks)
    row = dict(k_states=16, ci_width_mean=float((summ["ci95_hi"] - summ["ci95_lo"]).mean()),
               std_mean=float(summ["std"].mean()),
               u_mean_heuristic=float(z["u"].mean()),
               u_in_glare=float(z["u"][z["glare"] > 0.5].mean())
                              if (z["glare"] > 0.5).any() else float("nan"))
    with open("results/cpe_uncertainty.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys()); w.writeheader(); w.writerow([row])
    print("[CPE] credible-interval width reflects inverse ambiguity:", row)


def main():
    ensure_dirs()
    validate_physics_modules()
    run_angle_search_example()
    model, rows = train_and_evaluate(epochs=int(os.environ.get("VPC_EPOCHS", 8)))
    uncertainty_demo(model)
    # framework figures
    figures.figure1_architecture()
    figures.figure2_physical_vs_virtual()
    figures.figure3_fresnel_physics()
    figures.figure4_analyzer_sweep()
    d = render_cartridge(seed=3)
    figures.figure5_stokes_maps(d["rgb01"])
    # fig 10 physical-vs-virtual scatter from ACTUAL per-sample rows above
    import matplotlib.pyplot as plt
    xs = [r["mIoU_ConfigA_physical"] for r in rows]
    ys = [r["mIoU_ConfigB_virtual"] for r in rows]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(xs, ys, s=28)
    lim = [min(xs + ys) - .05, max(xs + ys) + .05]
    ax.plot(lim, lim, "k--", lw=0.8, label="y=x")
    ax.set_xlabel("Config A — physical polarizer (teacher)")
    ax.set_ylabel("Config B — virtual polarizer (RGB-only)")
    ax.set_title("Figure 10 — Physical vs virtual corrosion mIoU\n(each point = one held-out synthetic test image)")
    ax.legend(); ax.grid(alpha=0.3)
    figures._save(fig, "fig10_physical_vs_virtual")
    print("\nDemo complete. All numbers in results/*.csv were COMPUTED at run time.")
    print("NOTE: synthetic data only — real-cartridge claims require the hardware protocol.")


if __name__ == "__main__":
    main()
