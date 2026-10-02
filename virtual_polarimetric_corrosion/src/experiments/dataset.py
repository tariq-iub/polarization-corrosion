"""Dataset loading + group-aware splitting (rules #32/#33).

Splitting is ACQUISITION-GROUP level: every sample carries a `group_id`
(one physical cartridge / one capture session).  Samples sharing a group are
kept entirely within a single split — prevents near-duplicate leakage.
The demo synthetic dataset assigns group_id = image index (each render is an
independent 'object'), which trivially satisfies the constraint; real data
MUST set meaningful group ids (see docs/DATASET_PROTOCOL.md).
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


class CartridgeCorrosionDataset(Dataset):
    """NPZ-backed dataset. Required arrays: rgb01 (N,3,H,W), seg (N,H,W).
    Optional: severity, mask, normals, theta_i_deg, meas_stokes, stack,
              roughness_gt, dolp_gt, group_id (N,)."""

    def __init__(self, npz_path):
        z = np.load(npz_path)
        self.data = {k: z[k] for k in z.files}
        n = self.data["rgb01"].shape[0]
        if "group_id" not in self.data:
            self.data["group_id"] = np.arange(n)

    def __len__(self):
        return self.data["rgb01"].shape[0]

    def __getitem__(self, i):
        out = {}
        for k, v in self.data.items():
            arr = v[i]
            t = torch.from_numpy(arr.copy())
            if k == "angles":
                out[k] = arr.tolist()
            elif t.dtype in (torch.float64,):
                out[k] = t.float()
            else:
                out[k] = t
        return out


def collate(batch_list):
    out = {}
    keys = batch_list[0].keys()
    for k in keys:
        if k == "angles":
            out[k] = batch_list[0][k]
        else:
            out[k] = torch.stack([b[k] for b in batch_list])
    return out


def group_stratified_split(npz_in, npz_out_prefix, fractions=(0.7, 0.15, 0.15),
                           seed=0):
    """Split an NPZ by unique group ids into train/val/test files."""
    rng = np.random.default_rng(seed)
    z = np.load(npz_in)
    groups = z.get("group_id", np.arange(z["rgb01"].shape[0]))
    uniq = np.unique(groups)
    order = rng.permutation(uniq)
    n = len(uniq)
    a, b = int(n * fractions[0]), int(n * (fractions[0] + fractions[1]))
    splits = {"train": set(order[:a].tolist()),
              "val": set(order[a:b].tolist()),
              "test": set(order[b:].tolist())}
    written = {}
    for name, gset in splits.items():
        idx = np.where(np.isin(groups, list(gset)))[0]
        if idx.size == 0:
            continue
        payload = {k: z[k][idx] for k in z.files}
        payload["group_id"] = groups[idx]
        path = f"{npz_out_prefix}_{name}.npz"
        np.savez(path, **payload)
        written[name] = path
    return written
