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
    """Naive RPC: the ensemble-mean variance is used as the signal variance.

    With few members the ensemble mean still contains noise (variance noise/N), so the
    signal is overestimated and RPC is biased low. Use rpc_corrected for small ensembles.
    """
    em = ens.mean(axis=1)
    var_signal = em.var()
    var_total = ens.var()  # variance of individual members across starts and members
    return float(acc(ens, obs) / np.sqrt(var_signal / var_total))


def signal_variance(ens):
    """Unbiased estimate of the model's predictable (signal) variance.

    var(ensemble mean) = var(signal) + var(noise)/N, so subtract the noise term.
    """
    n = ens.shape[1]
    em = ens.mean(axis=1)
    var_noise = ens.var(axis=1, ddof=1).mean()
    return float(max(em.var() - var_noise / n, 1e-12))


def acc_infinite(ens, obs):
    """Estimate of the ACC an infinitely large ensemble would reach.

    A finite ensemble mean contains residual noise, which lowers its correlation with
    observations: r_N = r_inf * sqrt(var_signal / var(ensemble mean)). Inverting gives r_inf.
    """
    em_var = ens.mean(axis=1).var()
    return float(acc(ens, obs) * np.sqrt(em_var / signal_variance(ens)))


def rpc_corrected(ens, obs):
    """RPC with the small-ensemble bias removed from both numerator and denominator.

    Uses the infinite-ensemble ACC and the unbiased signal variance, so its expected value
    does not depend on ensemble size: a larger ensemble measures the SNP more precisely
    (less scatter), but does not make it go away.
    """
    return float(acc_infinite(ens, obs) / np.sqrt(signal_variance(ens) / ens.var()))


def perfect_model_acc(ens, max_members=10):
    """Skill of the model at predicting itself.

    Each of up to max_members members in turn is treated as 'truth' and correlated with
    the mean of the remaining members. The SNP shows up as acc(ens, obs) > this value.
    """
    n = ens.shape[1]
    if n < 2:
        return float("nan")
    vals = []
    for j in range(min(n, max_members)):
        rest = np.delete(ens, j, axis=1).mean(axis=1)
        vals.append(np.corrcoef(rest, ens[:, j])[0, 1])
    return float(np.mean(vals))


def spread_skill(ens, obs):
    """Ratio of mean ensemble spread to RMSE of the ensemble mean (1 = reliable)."""
    em = ens.mean(axis=1)
    rmse = np.sqrt(np.mean((em - obs) ** 2))
    spread = np.sqrt(ens.var(axis=1, ddof=1).mean())
    return float(spread / rmse)


def convergence(ens, obs, sizes, n_boot=200, seed=0):
    """ACC and RPC as a function of ensemble size, by resampling members.

    Returns dict of arrays (len(sizes),): ACC against observations, perfect-model ACC
    (the model predicting itself), naive RPC, bias-corrected RPC and model SNR.
    """
    rng = np.random.default_rng(seed)
    n_members = ens.shape[1]
    keys = ("acc", "acc_perfect", "rpc", "rpc_corrected", "snr")
    out = {"size": np.asarray(sizes), **{k: [] for k in keys}}
    for n in sizes:
        vals = {k: [] for k in keys}
        for _ in range(n_boot):
            idx = rng.choice(n_members, size=n, replace=n > n_members)
            sub = ens[:, idx]
            vals["acc"].append(acc(sub, obs))
            vals["acc_perfect"].append(perfect_model_acc(sub))
            vals["rpc"].append(rpc(sub, obs))
            vals["rpc_corrected"].append(rpc_corrected(sub, obs))
            vals["snr"].append(model_snr(sub))
        for k in keys:
            out[k].append(np.nanmean(vals[k]))
    for k in keys:
        out[k] = np.asarray(out[k])
    return out


def calibrate_ensemble(ens, obs_train, ens_train):
    """Correct a whole ensemble for the SNP (after Eade et al. 2014).

    1. Signal: the ensemble mean is rescaled so its variance equals the observed predictable
       variance, r^2 var(obs). This factor, r std(obs) / std(ensemble mean), is identical to
       the regression slope of observations on the ensemble mean (the RPS scaling used by
       Smith et al. 2025), so the two corrections coincide for the ensemble mean.
    2. Noise: member deviations from the ensemble mean are rescaled so their variance equals
       the unpredictable part, (1 - r^2) var(obs). This step makes probabilities reliable.

    Factors are estimated on training starts and applied to new starts.
    Returns the calibrated ensemble (same shape as ens) and the two factors.
    """
    em_tr = ens_train.mean(axis=1)
    r = acc(ens_train, obs_train)
    sig_scale = r * obs_train.std() / em_tr.std()
    noise_var = ens_train.var(axis=1, ddof=1).mean()
    noise_scale = np.sqrt(max(1 - r ** 2, 1e-6) * obs_train.var() / noise_var)
    em = ens.mean(axis=1, keepdims=True)
    out = em * sig_scale + (ens - em) * noise_scale
    return out, float(sig_scale), float(noise_scale)


def crps(ens, obs):
    """Mean continuous ranked probability score of an ensemble forecast (lower is better).

    CRPS = E|X - y| - 0.5 E|X - X'|, computed from sorted members for efficiency.
    """
    x = np.sort(ens, axis=1)
    n = x.shape[1]
    term1 = np.abs(x - obs[:, None]).mean(axis=1)
    w = (2 * np.arange(1, n + 1) - n - 1) / n ** 2
    term2 = (x * w).sum(axis=1)  # equals 0.5 E|X - X'|
    return float(np.mean(term1 - term2))


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
