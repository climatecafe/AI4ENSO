"""Signal-to-noise paradox (SNP) diagnostics for ensemble predictions.

The SNP (Scaife & Smith 2018): a model's ensemble mean correlates better with the real
world than with the model's own members, i.e. the model underestimates the predictability
of the real world. Common measures:

RPC  ratio of predictable components (Eade et al. 2014)
         RPC = r(ensemble mean, obs) / sqrt(var_signal / var_total)
     The denominator is the model's predictable fraction (the correlation an infinite
     ensemble mean would have with one of its own members). RPC > 1 indicates the SNP.

RSC  ratio of squared correlations
         RSC = r^2(ensemble mean, obs) / r^2(ensemble mean, member)
     i.e. the predictable variance fraction seen in observations over that seen in the
     model ("perfect-model" skill). For an infinite ensemble RSC = RPC^2. See
     ai4enso.snp.markov for its form in the Markov (AR1) framework.

Conventions: ens (n_starts, n_members); obs (n_starts,).

Finite ensembles bias these measures: the ensemble mean still contains noise of variance
var_noise / N, which inflates the apparent signal and lowers the correlation with
observations. The *_corrected functions remove that bias, so their expected values do
not depend on ensemble size.
"""
import numpy as np

from ..verification import acc, corr


def noise_variance(ens):
    """Mean within-ensemble variance (the unpredictable part)."""
    return float(np.asarray(ens).var(axis=1, ddof=1).mean())


def signal_variance(ens):
    """Unbiased estimate of the predictable (signal) variance:
    var(ensemble mean) - var_noise / N."""
    ens = np.asarray(ens)
    n = ens.shape[1]
    return float(max(ens.mean(axis=1).var() - noise_variance(ens) / n, 1e-12))


def model_snr(ens):
    """Model signal-to-noise ratio, std(signal) / std(noise), bias-corrected."""
    return float(np.sqrt(signal_variance(ens) / noise_variance(ens)))


def predictable_fraction(ens):
    """Model predictable variance fraction, var_signal / var_total (bias-corrected)."""
    ens = np.asarray(ens)
    return float(signal_variance(ens) / ens.var())


def acc_infinite(ens, obs):
    """ACC an infinitely large ensemble would reach.

    r_N = r_inf * sqrt(var_signal / var(ensemble mean)); inverting gives r_inf.
    """
    ens = np.asarray(ens)
    return float(acc(ens, obs) * np.sqrt(ens.mean(axis=1).var() / signal_variance(ens)))


def perfect_model_acc(ens, max_members=10):
    """Skill of the model at predicting itself, for the ensemble size at hand.

    Each of up to max_members members in turn is the 'truth' and is correlated with the
    mean of the remaining members.
    """
    ens = np.asarray(ens)
    n = ens.shape[1]
    if n < 2:
        return float("nan")
    vals = [corr(np.delete(ens, j, axis=1).mean(axis=1), ens[:, j]) for j in range(min(n, max_members))]
    return float(np.mean(vals))


def perfect_model_acc_infinite(ens):
    """Perfect-model skill of an infinite ensemble: sqrt(predictable fraction)."""
    return float(np.sqrt(predictable_fraction(ens)))


def rpc(ens, obs):
    """Naive RPC (ensemble-mean variance used as the signal variance). Biased low for
    small ensembles; prefer rpc_corrected."""
    ens = np.asarray(ens)
    return float(acc(ens, obs) / np.sqrt(ens.mean(axis=1).var() / ens.var()))


def rpc_corrected(ens, obs):
    """RPC with the finite-ensemble bias removed from numerator and denominator."""
    return float(acc_infinite(ens, obs) / np.sqrt(predictable_fraction(ens)))


def rsc(ens, obs):
    """Ratio of squared correlations for the ensemble at hand:
    r^2(ensemble mean, obs) / r^2(ensemble mean of the others, member)."""
    return float(acc(ens, obs) ** 2 / perfect_model_acc(ens) ** 2)


def rsc_corrected(ens, obs):
    """Ratio of squared correlations for an infinite ensemble (= rpc_corrected^2)."""
    return float(rpc_corrected(ens, obs) ** 2)


def summary(ens, obs):
    """All SNP measures for one ensemble."""
    return {"acc": acc(ens, obs), "acc_infinite": acc_infinite(ens, obs),
            "acc_perfect": perfect_model_acc(ens), "acc_perfect_infinite": perfect_model_acc_infinite(ens),
            "rpc": rpc(ens, obs), "rpc_corrected": rpc_corrected(ens, obs),
            "rsc": rsc(ens, obs), "rsc_corrected": rsc_corrected(ens, obs), "model_snr": model_snr(ens)}


def convergence(ens, obs, sizes, n_boot=200, seed=0):
    """SNP measures as a function of ensemble size, by resampling members.

    Returns dict of arrays (len(sizes),) for acc, acc_perfect, rpc, rpc_corrected, rsc, snr.
    """
    ens = np.asarray(ens)
    rng = np.random.default_rng(seed)
    n_members = ens.shape[1]
    keys = ("acc", "acc_perfect", "rpc", "rpc_corrected", "rsc", "snr")
    out = {"size": np.asarray(sizes), **{k: [] for k in keys}}
    for n in sizes:
        vals = {k: [] for k in keys}
        for _ in range(n_boot):
            sub = ens[:, rng.choice(n_members, size=n, replace=n > n_members)]
            vals["acc"].append(acc(sub, obs))
            vals["acc_perfect"].append(perfect_model_acc(sub))
            vals["rpc"].append(rpc(sub, obs))
            vals["rpc_corrected"].append(rpc_corrected(sub, obs))
            vals["rsc"].append(rsc(sub, obs))
            vals["snr"].append(model_snr(sub))
        for k in keys:
            out[k].append(np.nanmean(vals[k]))
    for k in keys:
        out[k] = np.asarray(out[k])
    return out


def calibrate_ensemble(ens, obs_train, ens_train):
    """Correct a whole ensemble for the SNP (after Eade et al. 2014).

    1. Signal: rescale the ensemble mean so its variance equals the observed predictable
       variance r^2 var(obs). The factor r std(obs) / std(ensemble mean) equals the
       regression slope of obs on the ensemble mean (the RPS scaling of Smith et al. 2025).
    2. Noise: rescale member deviations so their variance equals (1 - r^2) var(obs).

    Factors come from training starts and are applied to new starts. Use training and
    target ensembles of the same size, since the ensemble-mean variance depends on N.
    Returns (calibrated ensemble, signal factor, noise factor).
    """
    ens, ens_train = np.asarray(ens), np.asarray(ens_train)
    r = acc(ens_train, obs_train)
    sig_scale = r * np.std(obs_train) / ens_train.mean(axis=1).std()
    noise_scale = np.sqrt(max(1 - r ** 2, 1e-6) * np.var(obs_train) / noise_variance(ens_train))
    em = ens.mean(axis=1, keepdims=True)
    return em * sig_scale + (ens - em) * noise_scale, float(sig_scale), float(noise_scale)
