"""A trained MLP word-lock scorer usable as the payload of a context_rules.ContextRule (kind "gbm": anything with predict_proba).

The recipe is Addendum 1 of results/lctx_features2_20261111/PREREGISTERED.md (fit_mlp.py): inputs standardised on the fit rows, NaN -> 0 after standardisation plus one 0/1
missing-indicator column per feature that had any NaN in the fit rows; two hidden layers of 128 ReLU units (dropout only in training); the score is the mean of the sigmoid outputs of
several seeds (an ensemble). Stored as plain numpy arrays and evaluated in numpy float32, so a pickled rule needs neither torch nor the training script to load."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MLPEnsemble:
    mu: np.ndarray
    sd: np.ndarray
    has_nan: np.ndarray
    layers: list = field(default_factory=list)          # per seed: [(W [in, out] float32, b [out] float32), ...] for the three Linear layers

    def prep(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, dtype=np.float32) - self.mu) / self.sd
        miss = np.isnan(Z)
        Z[miss] = 0.0
        cols = [Z] + ([miss[:, self.has_nan].astype(np.float32)] if self.has_nan.any() else [])
        return np.concatenate(cols, 1).astype(np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = self.prep(X)
        ps = []
        for net in self.layers:
            h = Z
            for i, (W, b) in enumerate(net):
                h = h @ W + b
                if i < len(net) - 1:
                    h = np.maximum(h, 0.0)
            ps.append(1.0 / (1.0 + np.exp(-h[:, 0].astype(np.float64))))
        p = np.mean(ps, 0)
        return np.stack([1.0 - p, p], 1)

    @staticmethod
    def from_torch(prep, models) -> "MLPEnsemble":
        """prep: fit_mlp.Prep (mu, sd, has_nan); models: torch Sequential(Linear, ReLU, Dropout, Linear, ReLU, Dropout, Linear) per seed."""
        import torch
        nets = []
        for m in models:
            lin = [x for x in m if isinstance(x, torch.nn.Linear)]
            nets.append([(x.weight.detach().cpu().numpy().T.astype(np.float32).copy(), x.bias.detach().cpu().numpy().astype(np.float32).copy()) for x in lin])
        return MLPEnsemble(np.asarray(prep.mu, np.float32), np.asarray(prep.sd, np.float32), np.asarray(prep.has_nan, bool), nets)
