"""Teacher / Student polarization distillation (rules #27/#28/#29, Config C).

Teacher input : real hardware polarimetric stack I0,I45,I90,I135 (+RGB) — the
                measurement-derived Stokes [S0,S1,S2] are computed with
                optics.stokes.stokes_from_4angle (MEASURED quantities).
Student input : single conventional RGB image → virtual Stokes (ESTIMATED).
Distillation  : align student Ŝ-field and PSRF coefficients to teacher's
                measured-Stokes-derived features (L_distill), in addition to
                standard task losses.  Deployment = student only (RGB-only).

Honest framing: distillation transfers *task-relevant polarization information*
that is learnable from RGB; it does NOT make the student a polarimeter.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.vp_corrosion_net import DWConv, ResBottleneck, SEAttention
from ..optics.stokes import stokes_from_4angle, dolp, aolp


class PolarizationTeacher(nn.Module):
    """Encoder over the MEASURED stack + derived DoLP/AoLP maps."""

    def __init__(self, n_classes=4, base=24):
        super().__init__()
        # input channels: I0,I45,I90,I135, DoLP, cos2φ, sin2φ, S0norm = 8
        self.stem = nn.Sequential(nn.Conv2d(8, base, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(base), nn.SiLU(), ResBottleneck(base))
        self.d1 = nn.Sequential(DWConv(base, base * 2, 2), ResBottleneck(base * 2))
        self.d2 = nn.Sequential(DWConv(base * 2, base * 4, 2), ResBottleneck(base * 4),
                                 SEAttention(base * 4))
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.seg = nn.Conv2d(base, n_classes, 1)
        self.feat_proj = nn.Conv2d(base * 4, 3, 1)   # auxiliary Stokes decoder

    @staticmethod
    def stack_features(i_stack: torch.Tensor):
        """i_stack (B,4,H,W) at angles [0,45,90,135] → 8-channel tensor."""
        s = stokes_from_4angle(i_stack[:, 0], i_stack[:, 1], i_stack[:, 2], i_stack[:, 3])
        d = dolp(s)
        phi = aolp(s)
        s0n = s[:, 0] / (s[:, 0].amax((1, 2), keepdim=True) + 1e-6)
        return torch.cat([i_stack, d[:, None], torch.cos(2 * phi)[:, None],
                          torch.sin(2 * phi)[:, None], s0n[:, None]], dim=1), s

    def forward(self, i_stack):
        x, s_meas = self.stack_features(i_stack)
        e0 = self.stem(x); e1 = self.d1(e0); e2 = self.d2(e1)
        d2 = self.u2(e2) + e1
        d1 = self.u1(d2) + e0
        return {"seg": self.seg(d1), "stokes_true": s_meas, "feat": e2}


def distill_alignment(student_stokes_hat: torch.Tensor,
                      teacher_stokes_true: torch.Tensor):
    """Scale-invariant L1 alignment of Ŝ→S fields (per-image normalisation).

    Normalising per image prevents the trivial 'predict zero' solution while
    keeping the loss insensitive to absolute radiometry (exposure unknown).
    """
    def norm(t):
        return t / (t.abs().amax(dim=(1, 2, 3), keepdim=True) + 1e-6)
    return F.smooth_l1_loss(norm(student_stokes_hat), norm(teacher_stokes_true))


def train_student_with_distillation(student, teacher, loader, opt=None,
                                    epochs=5, lam_task=1.0, lam_dist=0.5,
                                    device="cpu", log_every=10):
    """Minimal reference training loop for Configuration C.

    Teacher frozen; student trained on task loss (CE on GT seg) + distillation.
    Returns list of loss dicts (usable for Figure/repro logs).
    """
    student.to(device).train()
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    opt = opt or torch.optim.AdamW(student.parameters(), lr=2e-4)
    hist = []
    step = 0
    for ep in range(epochs):
        for batch in loader:
            rgb = batch["rgb01"].to(device)
            tgt = batch["seg"].to(device)
            stack = batch["stack"].to(device)
            out_s = student(rgb)
            with torch.no_grad():
                out_t = teacher(stack)
            l_task = F.cross_entropy(out_s["seg"], tgt.long(), ignore_index=255)
            l_dist = distill_alignment(out_s["stokes_hat"], out_t["stokes_true"])
            loss = lam_task * l_task + lam_dist * l_dist
            opt.zero_grad(); loss.backward(); opt.step()
            hist.append({"epoch": ep, "step": step, "task": float(l_task),
                         "dist": float(l_dist), "total": float(loss)})
            step += 1
            if step % log_every == 0:
                print(f"ep{ep} step{step} task={float(l_task):.4f} dist={float(l_dist):.4f}")
    student.eval()
    return hist
