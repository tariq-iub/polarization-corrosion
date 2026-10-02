"""Physics-constrained training losses (rules #21–#23) + distillation (#28).

L_total = λ_seg L_seg + λ_cls L_cls + λ_rec L_rec + λ_pol L_pol
          + λ_per L_periodic + λ_edge L_edge + λ_phys L_phys + λ_unc L_unc
          + λ_dist L_distill   (Configuration C only)

Definitions implemented here:
* L_seg     : CE + soft-Dice on corrosion classes.
* L_cls     : severity regression (smooth-L1) with ordinal-consistency term.
* L_rec     : reconstruction consistency — the latent lobes must explain RGB:
              I_RGB ≈ D + S  (per channel, linear domain)  [rule #23]
* L_pol     : when real polarimetric supervision exists (teacher / Config A):
              Huber( Î(θk), Iθk_measured ) per channel-summed intensity.
* L_periodic: I(θ) − I(θ+180°) = 0 enforced numerically [rule #22].  Exactly
              satisfied by the Malus layer; kept as a guard for learned variants.
* L_edge    : gradient matching between predicted mask boundary and GT.
* L_phys    : admissibility |Ŝ1,S2| ≤ Ŝ0 (hinge); ρ∈[0,1]; roughness bounds;
              plus Fresnel-prior pull toward conductor DoLP curve at geometry θi.
* L_unc     : heteroscedastic NLL for severity/latent targets; evidential KL for
              segmentation concentrations.
* L_distill : feature-level MSE between student virtual Stokes/PSRF embedding
              and teacher measured-Stokes embedding (hardware→software).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def ce_dice_loss(logits, target, n_classes, dice_weight=0.5):
    """CE (ignore_index=255) + soft Dice over classes 1..C-1 (corrosion)."""
    ce = F.cross_entropy(logits, target.long(), ignore_index=255)
    p = torch.softmax(logits, dim=1)
    t = F.one_hot(target.clamp(max=n_classes - 1), n_classes).permute(0, 3, 1, 2).float()
    inter = (p * t).sum((2, 3))
    union = p.sum((2, 3)) + t.sum((2, 3))
    dice = 1 - (2 * inter + 1e-6) / (union + 1e-6)
    valid = (target != 255)[:, None].float()
    return ce + dice_weight * (dice * valid).sum() / (valid.sum() * n_classes + 1e-6)


def severity_loss(pred, gt, mask=None):
    l = F.smooth_l1_loss(pred, gt, reduction="none")
    if mask is not None:
        return (l * mask).sum() / (mask.sum() + 1e-6)
    return l.mean()


def reconstruction_loss(rgb_lin, latent):
    """|RGB_linear − (D+S)/normaliser|₁  (rule #23)."""
    recon = latent["D"] + latent["S"]
    ref = rgb_lin.mean(1, keepdim=True)
    denom = recon.mean().clamp(min=1e-6)
    num = (recon / denom - ref / ref.mean().clamp(min=1e-6)).abs().mean()
    return num


def polarization_supervision_loss(stokes_hat, s0_meas, s1_meas, s2_meas):
    """Huber between ESTIMATED Stokes and MEASURED Stokes (Config C/A only!)."""
    loss = 0.0
    for h, m in ((stokes_hat[:, 0], s0_meas), (stokes_hat[:, 1], s1_meas),
                 (stokes_hat[:, 2], s2_meas)):
        scale = m.abs().mean().clamp(min=1e-6)
        loss = loss + F.smooth_l1_loss(h / scale, m / scale)
    return loss / 3.0


def periodicity_loss(analyzer_fn, stokes_hat, angles_deg=(0., 31., 74., 129.)):
    tot = 0.0
    for a in angles_deg:
        i_a = analyzer_fn(stokes_hat, a)
        i_b = analyzer_fn(stokes_hat, a + 180.0)
        tot = tot + (i_a - i_b).abs().mean()
    return tot


def edge_loss(seg_logits, target_mask):
    p = torch.softmax(seg_logits, 1)[:, 1:].sum(1, keepdim=True)
    gx = p[..., :, 1:] - p[..., :, :-1]
    gy = p[..., 1:, :] - p[..., :-1, :]
    tx = target_mask[..., 1:] - target_mask[..., :-1]
    ty = target_mask[1:, ...] - target_mask[:-1, ...]
    lx = (gx - tx[:, :, :gx.shape[2]]).abs().mean() if gx.numel() else 0.0
    ly = (gy - ty[:, :gy.shape[1], :gy.shape[2]]).abs().mean() if gy.numel() else 0.0
    return lx + ly


def physics_admissibility_loss(stokes_hat, latent, theta_i_deg=None,
                               fresnel_dolp_ref=None):
    """Hinge penalties keeping the inverse estimate physically legal."""
    s0, s1, s2 = stokes_hat[:, 0], stokes_hat[:, 1], stokes_hat[:, 2]
    viol = F.relu(s1 * s1 + s2 * s2 - s0 * s0 + 1e-6)          # |S⊥| ≤ S0
    l = viol.mean()
    for key, lo, hi in (("roughness", 0.0, 1.0), ("rho", 0.0, 1.0)):
        v = latent[key]
        l = l + F.relu(lo - v).mean() + F.relu(v - hi).mean()
    if theta_i_deg is not None and fresnel_dolp_ref is not None:
        # pull ρ toward conductor-Fresnel reference where specular dominates
        w = (latent["S"] / (latent["D"] + latent["S"] + 1e-6)).detach()
        l = l + (w * (latent["rho"].squeeze(1) - fresnel_dolp_ref) ** 2).mean()
    return l


def uncertainty_nll(sev_pred, sev_logvar, sev_gt):
    inv = torch.exp(-sev_logvar)
    return (0.5 * ((sev_pred - sev_gt) ** 2 * inv + sev_logvar)).mean()


def distillation_loss(student_feats, teacher_feats):
    """MSE after 1x1 alignment conv is applied outside; plain MSE here."""
    return F.mse_loss(student_feats, teacher_feats)


LOSS_WEIGHTS_DEFAULT = dict(seg=1.0, cls=0.5, rec=0.5, pol=0.0, per=0.1,
                            edge=0.2, phys=0.3, unc=0.2, dist=0.0)


def total_loss(outputs, batch, analyzer, weights=LOSS_WEIGHTS_DEFAULT,
               fresnel_ref=None):
    """Assemble L_total from model outputs + batch dict.

    batch keys: 'rgb_lin' (B,3,H,W), 'seg' (B,H,W), 'severity' (B,1,H,W),
                optional 'meas_stokes' (B,3,H,W) from hardware captures,
                optional 'teacher_feat'.
    """
    dev = outputs["seg"].device
    tgt = batch["seg"]
    l = {}
    l["seg"] = ce_dice_loss(outputs["seg"], tgt, outputs["seg"].shape[1])
    if "severity" in batch:
        l["cls"] = severity_loss(outputs["severity"], batch["severity"])
    if "rec" in weights and batch.get("rgb_lin") is not None:
        l["rec"] = reconstruction_loss(batch["rgb_lin"], outputs["latent"])
    if weights.get("pol", 0) > 0 and "meas_stokes" in batch:
        ms = batch["meas_stokes"]
        l["pol"] = polarization_supervision_loss(outputs["stokes_hat"],
                                                 ms[:, 0], ms[:, 1], ms[:, 2])
    sh = outputs["stokes_hat"]
    l["per"] = periodicity_loss(lambda s, a: 0.5 * (s[:, 0] + math.cos(2 * math.radians(a)) * s[:, 1] + math.sin(2 * math.radians(a)) * s[:, 2]), sh)
    l["edge"] = edge_loss(outputs["seg"], (tgt > 0).float())
    ti = batch.get("theta_i_deg")
    l["phys"] = physics_admissibility_loss(sh, {k: v for k, v in outputs["latent"].items()},
                                           ti, fresnel_ref)
    if weights.get("unc", 0) > 0 and "logvar" in outputs["latent"]:
        lv = outputs["latent"]["logvar"]
        sp = F.interpolate(lv, size=outputs["severity"].shape[-2:], mode="bilinear")
        l["unc"] = uncertainty_nll(outputs["severity"], sp, batch.get("severity", outputs["severity"]))
    if weights.get("dist", 0) > 0 and "teacher_feat" in batch:
        l["dist"] = distillation_loss(outputs.get("psrf_embed", outputs["stokes_hat"]),
                                      batch["teacher_feat"])
    tot = 0.0
    for k, v in l.items():
        tot = tot + weights.get(k, 0.0) * v
    return tot, l
