"""Skill and signal-to-noise diagnostics for ensemble hindcasts.

Conventions: ens has shape (n_starts, n_members) for one target quantity (e.g. the
Nino-3.4 winter mean at lead year 2); obs has shape (n_starts,).

RPC (ratio of predictable components; Eade et al. 2014):
    RPC = ACC / sqrt(var(ensemble mean) / mean var(member))
RPC > 1 means the model is better at predicting the real world than at predicting
itself: the signal-to-noise paradox.
"""
import numpy as np


def acc(ens, obs):
    """Anomaly correlation of the ensemble mean with observations."""
    em = ens.mean(axis=1)
    return float(np.corrcoef(em, obs)[0, 1])


def model_snr(ens):
    """Model signal-to-noise ratio: std(ensemble mean) / mean member spread."""
    signal = ens.mean(axis=1).std()
    noise = np.sqrt(ens.var(axis=1, ddof=1).mean())
    return float(signal / noise)


def rpc(ens, obs):
    em = ens.mean(axis=1)
    var_signal = em.var()
    var_total = ens.var()  # variance of individual members across starts and members
    return float(acc(ens, obs) / np.sqrt(var_signal / var_total))


def spread_skill(ens, obs):
    """Ratio of mean ensemble spread to RMSE of the ensemble mean (1 = reliable)."""
    em = ens.mean(axis=1)
    rmse = np.sqrt(np.mean((em - obs) ** 2))
    spread = np.sqrt(ens.var(axis=1, ddof=1).mean())
    return float(spread / rmse)


def convergence(ens, obs, sizes, n_boot=200, seed=0):
    """ACC and RPC as a function of ensemble size, by resampling members.

    Returns dict of arrays (len(sizes),) with mean ACC, RPC and model SNR.
    """
    rng = np.random.default_rng(seed)
    n_members = ens.shape[1]
    out = {"size": np.asarray(sizes), "acc": [], "rpc": [], "snr": []}
    for n in sizes:
        a, r, s = [], [], []
        for _ in range(n_boot):
            idx = rng.choice(n_members, size=n, replace=n > n_members)
            sub = ens[:, idx]
            a.append(acc(sub, obs))
            r.append(rpc(sub, obs))
            s.append(model_snr(sub))
        out["acc"].append(np.mean(a))
        out["rpc"].append(np.mean(r))
        out["snr"].append(np.mean(s))
    for k in ("acc", "rpc", "snr"):
        out[k] = np.asarray(out[k])
    return out


def rpc_adjust(ens, obs_train, ens_train):
    """'Correct for' the SNP by rescaling the ensemble-mean signal (Eade et al. 2014).

    The scaling factor is estimated on training starts and applied to new starts, so the
    calibrated ensemble mean has the variance implied by the observed predictable signal.
    """
    em_train = ens_train.mean(axis=1)
    r = acc(ens_train, obs_train)
    target_std = r * obs_train.std()
    scale = target_std / em_train.std()
    em = ens.mean(axis=1)
    return em * scale, float(scale)


def winter_mean(x, start_month, winter_index, months=(12, 1, 2)):
    """Mean over the DJF months of a given winter after initialization.

    x : array (..., lead_months). winter_index 0 = first winter after the start.
    """
    lead = x.shape[-1]
    cal = (start_month - 1 + np.arange(lead)) % 12 + 1
    # identify the DJF blocks in order
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

    Uses the Bai et al. (2026) thresholds: winter-1 Nino-3.4 > 0.5 sd and winter-2 > 0.4 sd.
    w1, w2 : arrays of first- and second-winter values (any shape, matched).
    """
    first = w1 > 0.5 * sd
    if first.sum() == 0:
        return float("nan")
    return float(np.mean(w2[first] > 0.4 * sd))
