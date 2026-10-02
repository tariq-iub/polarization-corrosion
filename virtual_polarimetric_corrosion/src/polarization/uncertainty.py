"""Uncertainty estimation (Module I, rule #24) + Counterfactual Polarimetric
Ensemble — CPE (rule #25).

Because RGB-only inverse polarimetry is fundamentally underdetermined, every
virtual-polarimetric quantity ships with an uncertainty estimate.  Implemented
strategies:

1. Deterministic heuristic map U(x,y): glare / clipping / shadow / ambiguity.
2. MC-Dropout over the learned latent head (``mc_dropout_predict``).
3. Deep ensemble spread (mean/std over model copies).
4. Heteroscedastic NLL training objective (``nll_gaussian_loss``).
5. Evidential-style Dirichlet concentration for segmentation confidence
   (``evidential_nll``; simplified, documented).
6. CPE: sample K physically-plausible latent states Z_k (perturbing
   non-identifiable directions), propagate through the virtual analyzer and
   report mean/variance/credible interval of Î(θ) — explicitly acknowledging
   single-image ambiguity.
"""
from __future__ import annotations

import math

import torch


def heuristic_uncertainty(glare: torch.Tensor, rho: torch.Tensor,
                          roughness: torch.Tensor,
                          clipped: torch.Tensor | None = None,
                          shadow: torch.Tensor | None = None) -> torch.Tensor:
    """U = w·glare + w·(1−ρ identifiability) + w·rough + clip + shadow ∈ [0,1]."""
    u = 0.45 * glare + 0.25 * (1.0 - rho.clamp(0, 1)) + 0.30 * roughness
    if clipped is not None:
        u = u + clipped.float()
    if shadow is not None:
        u = u + 0.5 * shadow.float()
    return u.clamp(0.0, 1.0)


@torch.no_grad()
def mc_dropout_predict(model, x, n_samples: int = 20):
    """Monte-Carlo dropout: enable dropout at inference, average outputs.

    model must run in train() mode only to activate dropout but we disable
    other stochastic layers by assumption.  Returns (mean, std) stacked.
    """
    was_training = model.training
    model.train()
    outs = []
    for _ in range(n_samples):
        y = model(x)
        if isinstance(y, dict):
            y = y["latent"]
        outs.append(y)
    if not was_training:
        model.eval()
    st = torch.stack(outs)
    return st.mean(0), st.std(0)


def nll_gaussian_loss(pred: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor,
                      mask: torch.Tensor | None = None):
    """Heteroscedastic regression NLL: ½[ (y−μ)²/σ² + log σ² ]."""
    inv2s2 = torch.exp(-log_var)
    l = 0.5 * ((pred - target) ** 2 * inv2s2 + log_var)
    if mask is not None:
        return (l * mask).sum() / (mask.sum() + 1e-8)
    return l.mean()


def evidential_nll(logits_conc: torch.Tensor, labels: torch.Tensor,
                   anneal: float = 1.0, kl_weight: float = 1.0):
    """Simplified Dirichlet-evidential loss for segmentation confidence.

    logits_conc: (B,C,H,W) ≥0 concentrations α (softplus applied outside).
    Encourages correct-class evidence while penalising total evidence when
    wrong → model can express 'I don't know' (abstention support, rule #48).
    """
    labels_oh = torch.nn.functional.one_hot(labels.long(), logits_conc.shape[1]).permute(0, 3, 1, 2).float()
    alpha = logits_conc + 1.0
    s = alpha.sum(1, keepdim=True)
    err = ((labels_oh - alpha / s) ** 2).sum(1, keepdim=True)
    acc = (labels_oh * alpha / s).sum(1, keepdim=True)
    dice_err = (1.0 - 2.0 * acc) / (s + 1.0)
    res = err + dice_err
    # KL[Dir(α) || Dir(1)] — Amini et al. 2020, eq. (7):
    c = alpha.shape[1]                       # number of classes
    s_alpha = alpha.sum(1, keepdim=True)
    dkl = (torch.lgamma(s_alpha) - torch.lgamma(alpha).sum(1, keepdim=True)
           + ((alpha - 1.0) * torch.digamma(alpha)).sum(1, keepdim=True)
           + (c - 1.0) * torch.lgamma(torch.tensor(float(c), device=alpha.device))
           - (c - 1.0) * torch.lgamma(torch.ones_like(s_alpha)))
    return res.mean() + kl_weight * anneal * dkl.clamp(min=0.0).mean()


def expected_calibration_error(probs: torch.Tensor, targets: torch.Tensor,
                               n_bins: int = 15):
    """ECE for pixel-wise classification confidence (rule #48)."""
    conf, pred = probs.max(dim=-1)
    acc = (pred == targets).float()
    bins = torch.linspace(0, 1, n_bins + 1, device=conf.device)
    ece = 0.0
    n = conf.numel()
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.float().sum() / n) * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


# ---------------------------------------------------------------- CPE --------
def counterfactual_latents(z_base: dict, k_states: int = 8,
                           phi_sigma_deg: float = 22.5,
                           rho_rel_sigma: float = 0.25,
                           eta_rel_sigma: float = 0.15,
                           generator: torch.Generator | None = None):
    """Sample K plausible latent optical states around z_base (rule #25).

    Perturbation directions are chosen along NON-IDENTIFIABLE axes of the
    single-RGB inverse problem: absolute AoLP offset (φ), polarization
    magnitude (ρ) and IOR proxy (η).  D,S,geometry stay fixed so that the
    reconstruction-consistency loss remains anchored.
    Returns list of dicts length k_states.
    """
    dev = z_base["rho"].device
    out = []
    for j in range(k_states):
        z = dict(z_base)
        dphi = torch.deg2rad(torch.randn((), generator=generator, device=dev) * phi_sigma_deg)
        rho_mult = 1.0 + torch.randn((), generator=generator, device=dev) * rho_rel_sigma
        eta_mult = 1.0 + torch.randn((), generator=generator, device=dev) * eta_rel_sigma
        z["phi"] = z_base["phi"] + dphi
        z["rho"] = (z_base["rho"] * rho_mult).clamp(0, 1)
        z["eta"] = (z_base["eta"] * eta_mult).clamp(0.2, 4.0)
        out.append(z)
    return out


def cpe_stack_summary(stacks: torch.Tensor):
    """stacks: (K,...,H,W) virtual Î(θ) per counterfactual state.

    Returns mean, std, and 95% credible interval (empirical quantiles over K).
    """
    mean = stacks.mean(0)
    std = stacks.std(0)
    lo = torch.quantile(stacks, 0.025, dim=0)
    hi = torch.quantile(stacks, 0.975, dim=0)
    return {"mean": mean, "std": std, "ci95_lo": lo, "ci95_hi": hi}
