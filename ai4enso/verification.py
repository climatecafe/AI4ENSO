"""Forecast verification: deterministic and probabilistic scores, and lead-time helpers.

Conventions used throughout ai4enso:
    ens : array (n_starts, n_members) of one forecast quantity (e.g. winter-2 Nino-3.4)
    obs : array (n_starts,) of the verifying observations
    fc  : array (n_starts,) of a deterministic forecast (e.g. an ensemble mean)
"""
import numpy as np


def corr(fc, obs):
    """Pearson correlation of two 1-D series."""
    return float(np.corrcoef(fc, obs)[0, 1])


def acc(ens, obs):
    """Anomaly correlation of the ensemble mean with observations."""
    return corr(np.asarray(ens).mean(axis=1), obs)


def rmse(fc, obs):
    return float(np.sqrt(np.mean((np.asarray(fc) - np.asarray(obs)) ** 2)))


def msss(fc, obs, ref=None):
    """Mean-squared skill score against a reference forecast (climatology = 0 by default)."""
    obs = np.asarray(obs)
    ref = np.zeros_like(obs) if ref is None else np.asarray(ref)
    return float(1 - np.mean((fc - obs) ** 2) / np.mean((ref - obs) ** 2))


def spread_skill(ens, obs):
    """Ratio of mean ensemble spread to RMSE of the ensemble mean (1 = reliable)."""
    ens = np.asarray(ens)
    spread = np.sqrt(ens.var(axis=1, ddof=1).mean())
    return float(spread / rmse(ens.mean(axis=1), obs))


def crps(ens, obs):
    """Mean continuous ranked probability score of an ensemble forecast (lower is better).

    CRPS = E|X - y| - 0.5 E|X - X'|, computed from sorted members for efficiency.
    """
    x = np.sort(np.asarray(ens), axis=1)
    n = x.shape[1]
    term1 = np.abs(x - np.asarray(obs)[:, None]).mean(axis=1)
    w = (2 * np.arange(1, n + 1) - n - 1) / n ** 2
    term2 = (x * w).sum(axis=1)  # equals 0.5 E|X - X'|
    return float(np.mean(term1 - term2))


def skill_by_lead(fc, obs):
    """Correlation and RMSE at each lead.

    fc, obs : arrays (n_starts, n_leads). Returns dict of arrays (n_leads,).
    """
    fc, obs = np.asarray(fc), np.asarray(obs)
    return {"corr": np.array([corr(fc[:, k], obs[:, k]) for k in range(fc.shape[1])]),
            "rmse": np.sqrt(np.mean((fc - obs) ** 2, axis=0))}


def winter_mean(x, start_month, winter_index, months=(12, 1, 2)):
    """Mean over the DJF months of a given winter after initialization.

    x : array (..., lead_months), lead 0 = start_month. winter_index 0 = first winter.
    """
    lead = x.shape[-1]
    cal = (start_month - 1 + np.arange(lead)) % 12 + 1
    blocks, current = [], []
    for k, c in enumerate(cal):
        if c in months:
            current.append(k)
            if c == months[-1]:
                blocks.append(current)
                current = []
        elif current:
            current = []
    if winter_index >= len(blocks):
        raise ValueError("lead too short for that winter")
    return x[..., blocks[winter_index]].mean(axis=-1)


def multiyear_elnino_fraction(w1, w2, sd):
    """Fraction of first-winter El Nino events that persist into a second winter.

    Thresholds follow Bai et al. (2026): winter-1 Nino-3.4 > 0.5 sd and winter-2 > 0.4 sd.
    w1, w2 : arrays of first- and second-winter values (any shape, matched).
    """
    first = w1 > 0.5 * sd
    if first.sum() == 0:
        return float("nan")
    return float(np.mean(w2[first] > 0.4 * sd))
