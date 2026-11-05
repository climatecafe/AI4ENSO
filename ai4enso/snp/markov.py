"""Markov (AR1) framework for the signal-to-noise paradox.

After Zhang & Kirtman (2019, GRL), who represented ensemble forecasts with a simple Markov
model and argued that the SNP depends mainly on the persistence and noise variance of the
model relative to observations. The closed-form expressions below are derived in this
package for that setting (see docs/snp.md); check them against the paper's notation before
quoting them.

Setting (monthly AR1 processes, all anomalies):
    observations   y(t+1) = a_o y(t) + e_o,   var(e_o) = sig_o^2
    model member   x(t+1) = a_m x(t) + e_m,   var(e_m) = sig_m^2
Forecasts start from the observed state y0 (perfectly known), with stationary variance
s_o^2 = sig_o^2 / (1 - a_o^2). At lead tau:
    obs        y   = a_o^tau y0 + noise,  noise var V_o = s_o^2 (1 - a_o^(2 tau))
    member i   x_i = a_m^tau y0 + noise,  noise var V_m = s_m^2 (1 - a_m^(2 tau))
with s_m^2 = sig_m^2 / (1 - a_m^2). Members have independent noise.

Results (N members):
    ACC_obs(N)   = a_o^tau * S / sqrt(S^2 + V_m / N),   S = a_m^tau s_o
    PM(N)        = S^2 / sqrt((S^2 + V_m/(N-1)) (S^2 + V_m))   (mean of N-1 vs 1 member)
    ACC_obs(inf) = a_o^tau
    PM(inf)      = S / sqrt(S^2 + V_m)       (= sqrt of the model predictable fraction)
    RPC          = ACC_obs(inf) / PM(inf),    RSC = RPC^2
If the model has the observed variance (s_m = s_o), RPC = (a_o / a_m)^tau: the SNP
appears exactly when the model is less persistent than the real world. With equal
persistence, a noisier model (s_m > s_o) also gives RPC > 1.
"""
import numpy as np


def stationary_variance(a, sigma):
    return sigma ** 2 / (1.0 - a ** 2)


def ar1_fit(x):
    """Lag-1 autocorrelation and innovation standard deviation of a series."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    a = np.sum(x[1:] * x[:-1]) / np.sum(x[:-1] ** 2)
    sigma = np.std(x[1:] - a * x[:-1])
    return float(a), float(sigma)


def ar1_skill(a_obs, sigma_obs, a_model, sigma_model, lead, n_members=np.inf):
    """Closed-form skill and SNP measures in the AR1 framework.

    lead may be a scalar or an array (time steps). Returns a dict of arrays.
    """
    lead = np.asarray(lead, dtype=float)
    s2_o = stationary_variance(a_obs, sigma_obs)
    s2_m = stationary_variance(a_model, sigma_model)
    S2 = a_model ** (2 * lead) * s2_o              # model signal variance
    V_m = s2_m * (1 - a_model ** (2 * lead))       # model noise variance
    acc_inf = a_obs ** lead
    pm_inf = np.sqrt(S2 / (S2 + V_m))
    out = {"lead": lead, "acc_inf": acc_inf, "acc_perfect_inf": pm_inf,
           "rpc": acc_inf / pm_inf, "rsc": (acc_inf / pm_inf) ** 2,
           "model_snr": np.sqrt(S2 / V_m)}
    if np.isfinite(n_members):
        n = float(n_members)
        out["acc"] = a_obs ** lead * np.sqrt(S2) / np.sqrt(S2 + V_m / n)
        out["acc_perfect"] = S2 / np.sqrt((S2 + V_m / (n - 1)) * (S2 + V_m)) if n > 1 else np.nan
        out["rsc_finite"] = (out["acc"] / out["acc_perfect"]) ** 2
    return out


def simulate_ar1_hindcasts(a_obs, sigma_obs, a_model, sigma_model, n_starts, n_members,
                           max_lead, seed=0):
    """Monte Carlo hindcasts in the AR1 framework, to check ar1_skill.

    Returns obs (n_starts, max_lead+1) and ens (n_starts, n_members, max_lead+1); lead 0 is
    the initial state.
    """
    rng = np.random.default_rng(seed)
    y0 = rng.standard_normal(n_starts) * np.sqrt(stationary_variance(a_obs, sigma_obs))
    obs = np.empty((n_starts, max_lead + 1))
    ens = np.empty((n_starts, n_members, max_lead + 1))
    obs[:, 0] = y0
    ens[:, :, 0] = y0[:, None]
    for k in range(1, max_lead + 1):
        obs[:, k] = a_obs * obs[:, k - 1] + sigma_obs * rng.standard_normal(n_starts)
        ens[:, :, k] = a_model * ens[:, :, k - 1] + sigma_model * rng.standard_normal((n_starts, n_members))
    return obs, ens


def rpc_map(a_obs, sigma_obs, a_model_values, sigma_model_values, lead):
    """RPC on a grid of model persistence (rows) and model noise (columns)."""
    A, Sg = np.meshgrid(np.asarray(a_model_values), np.asarray(sigma_model_values), indexing="ij")
    return ar1_skill(a_obs, sigma_obs, A, Sg, lead)["rpc"]
