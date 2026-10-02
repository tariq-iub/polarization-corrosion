"""Evaluation metrics (rules #45–#48).  Pure torch, no heavy deps."""
from __future__ import annotations

import math

import torch


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, n_classes, ignore=255):
    m = target != ignore
    p, t = pred[m].long(), target[m].long()
    cm = torch.zeros(n_classes, n_classes, dtype=torch.float64)
    idx = t * n_classes + p
    cm = torch.bincount(idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes).double()
    return cm


def segmentation_metrics(pred_cls: torch.Tensor, target: torch.Tensor, n_classes):
    """Returns dict with per-class IoU/Dice/P/R/spec + mIoU + balanced acc."""
    cm = confusion_matrix(pred_cls, target, n_classes)
    out = {"per_class": {}}
    ious = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        iou = tp / (tp + fp + fn + 1e-9)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-9)
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        out["per_class"][c] = dict(iou=float(iou), dice=float(dice), precision=float(prec),
                                   recall=float(rec), specificity=float(spec), f1=float(f1))
        ious.append(float(iou))
    present = [i for i, c in zip(ious, range(n_classes)) if (cm[c, :].sum() + cm[:, c].sum()) > 0]
    out["mIoU"] = float(sum(present) / max(len(present), 1))
    diag = cm.diagonal().sum()
    out["accuracy"] = float(diag / (cm.sum() + 1e-9))
    per_class_acc = cm.diagonal() / (cm.sum(1) + 1e-9)
    out["balanced_accuracy"] = float(per_class_acc[cm.sum(1) > 0].mean()) if (cm.sum(1) > 0).any() else 0.0
    return out


def severity_metrics(pred: torch.Tensor, gt: torch.Tensor, mask=None):
    if mask is not None:
        pred, gt = pred[mask], gt[mask]
    err = pred - gt
    mae = float(err.abs().mean())
    rmse = float(err.pow(2).mean().sqrt())
    ss_res = float(err.pow(2).sum())
    ss_tot = float((gt - gt.mean()).pow(2).sum()) + 1e-9
    r2 = 1 - ss_res / ss_tot
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def wrapped_angular_error_deg(phi_pred, phi_true, period_deg=180.0):
    """AoLP error with circular wrapping (rule #47)."""
    d = (phi_pred - phi_true) % period_deg
    return float(torch.minimum(d, period_deg - d).mean())


def glare_suppression_ratio(img_before, img_after, sat_thresh=0.98):
    b = (img_before >= sat_thresh * img_before.max().clamp(min=1e-6)).float().mean()
    a = (img_after >= sat_thresh * img_after.max().clamp(min=1e-6)).float().mean()
    return float((b + 1e-9) / (a + 1e-9)), float(b), float(a)


def image_quality_metrics(ref: torch.Tensor, proc: torch.Tensor):
    """Local contrast (std), gradient energy, entropy proxy of luminance."""
    lr = ref.mean(0) if ref.dim() == 3 else ref
    lp = proc.mean(0) if proc.dim() == 3 else proc
    dx = torch.zeros_like(lp); dy = torch.zeros_like(lp)
    dx[:, :-1] = lp[:, 1:] - lp[:, :-1]
    dy[:-1, :] = lp[1:, :] - lp[:-1, :]
    grad = float(torch.sqrt(dx * dx + dy * dy).mean())
    hist = torch.histc(lp, bins=64, min=0, max=1)
    p = hist / hist.sum()
    ent = float(-(p[p > 0] * p[p > 0].log()).sum())
    return {"contrast_std": float(lp.std()), "gradient_energy": grad,
            "entropy": ent, "local_contrast_ref_std": float(lr.std())}


def bootstrap_ci(values, stat_fn=lambda v: v.mean(), n_boot=1000, alpha=0.05, seed=0):
    """Nonparametric bootstrap CI over an array of per-sample statistics."""
    g = torch.Generator().manual_seed(seed)
    v = torch.as_tensor(values, dtype=torch.float64)
    n = v.numel()
    stats = []
    for _ in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g)
        stats.append(stat_fn(v[idx]))
    s = torch.tensor(stats)
    lo = float(torch.quantile(s, alpha / 2))
    hi = float(torch.quantile(s, 1 - alpha / 2))
    return {"point": float(stat_fn(v)), "ci95_lo": lo, "ci95_hi": hi,
            "std": float(s.std())}
