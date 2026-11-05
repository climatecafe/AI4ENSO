"""Model-agnostic explainable-AI tools for ENSO predictors.

All functions take a prediction function f(X) -> y (e.g. model.predict for a scikit-learn
model, or a wrapper around a PyTorch network), so they work with any ML model.

permutation_importance  Drop in skill when one input feature (or group) is shuffled.
occlusion               Change in prediction when a feature group is set to a baseline
                        (e.g. climatology = 0): a regional "what if this were neutral".
integrated_gradients    Sundararajan et al. (2017) attribution, with gradients from
                        central finite differences so no autodiff is needed.

Attribution methods can disagree in geoscience applications (e.g. Mamalakis et al. 2022),
so compare several and check them against physical expectations.
"""
import numpy as np


def _groups(n_features, groups):
    return {f"x{i}": [i] for i in range(n_features)} if groups is None else groups


def permutation_importance(f, X, y, groups=None, n_repeats=10, seed=0):
    """Mean drop in correlation skill when each feature group is permuted across samples."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    base = np.corrcoef(f(X), y)[0, 1]
    out = {}
    for name, cols in _groups(X.shape[1], groups).items():
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[:, cols] = X[perm][:, cols]
            drops.append(base - np.corrcoef(f(Xp), y)[0, 1])
        out[name] = float(np.mean(drops))
    return out


def occlusion(f, X, groups=None, baseline=0.0):
    """Mean absolute change in prediction when each group is replaced by a baseline."""
    X = np.asarray(X, dtype=float)
    y0 = f(X)
    out = {}
    for name, cols in _groups(X.shape[1], groups).items():
        Xo = X.copy()
        Xo[:, cols] = baseline
        out[name] = float(np.mean(np.abs(y0 - f(Xo))))
    return out


def integrated_gradients(f, X, baseline=None, steps=32, h=1e-3):
    """Integrated gradients for a scalar-output model; returns (n_samples, n_features).

    Attributions satisfy completeness approximately: they sum to f(x) - f(baseline).
    """
    X = np.asarray(X, dtype=float)
    B = np.zeros_like(X) if baseline is None else np.broadcast_to(baseline, X.shape)
    alphas = (np.arange(steps) + 0.5) / steps
    grads = np.zeros_like(X)
    for a in alphas:
        P = B + a * (X - B)
        for j in range(X.shape[1]):
            Pp, Pm = P.copy(), P.copy()
            Pp[:, j] += h
            Pm[:, j] -= h
            grads[:, j] += (f(Pp) - f(Pm)) / (2 * h)
    return (X - B) * grads / steps
