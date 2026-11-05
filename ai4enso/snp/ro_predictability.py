"""Signal-to-noise analysis for a linear stochastic recharge oscillator.

The two-variable linear RO,  d[T, h]/dt = A [T, h] + white noise (covariance Sigma),
has eigenvalues lambda = growth +/- i omega, so its ENSO growth rate and period
(2 pi / omega) follow directly from A. Jin (2022, GRL) showed with a recharge oscillator
that the ENSO SNP can arise from differences in growth rate and period, not only from
differences in persistence. This module computes, exactly, the skill and RPC of a "model"
oscillator forecasting an "observed" oscillator, both linear with CONSTANT coefficients.

Caveat: without a seasonal cycle in the growth rate there is no spring persistence
barrier or phase locking. In this simplified setting a more damped model (lower growth
rate) produces RPC > 1, while a period mismatch mainly adds phase error and lowers RPC.
Use the seasonally varying stochastic RO (ai4enso.dynamics.stochastic_ro) with Monte Carlo
hindcasts to study period effects that act through the seasonal cycle.

Method: the propagator is M(tau) = expm(A tau); the noise covariance accumulated over a
lead tau is obtained with Van Loan's (1978) block-matrix exponential; the stationary
covariance solves A C + C A^T + Sigma = 0. Forecasts start from the observed state.
"""
import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov


def ro_matrix(R, F1, F2, eps=0.0):
    """Linear RO matrix [[R, F1], [-F2, -eps]] for d[T, h]/dt (Burgers et al. 2005 form)."""
    return np.array([[R, F1], [-F2, -eps]], dtype=float)


def growth_and_period(A):
    """Growth rate (1/time) and period (time) of the oscillatory mode of a 2x2 matrix.

    Period is inf if the eigenvalues are real (no oscillation).
    """
    lam = np.linalg.eigvals(np.asarray(A, dtype=float))
    k = np.argmax(np.abs(lam.imag))
    omega = abs(lam[k].imag)
    return float(lam[k].real), float(2 * np.pi / omega) if omega > 0 else float("inf")


def matrix_from_growth_period(growth, period, F1=7.5, eps=0.0):
    """RO matrix with a prescribed growth rate and period, for a chosen F1 and eps.

    From tr(A) = R - eps = 2 growth and det(A) = -R eps + F1 F2 = growth^2 + omega^2.
    """
    omega = 2 * np.pi / period
    R = 2 * growth + eps
    F2 = (growth ** 2 + omega ** 2 + R * eps) / F1
    return ro_matrix(R, F1, F2, eps)


def _noise_cov(A, Sigma, tau):
    """Q(tau) = int_0^tau expm(A s) Sigma expm(A s)^T ds (Van Loan's method)."""
    n = A.shape[0]
    blk = np.zeros((2 * n, 2 * n))
    blk[:n, :n] = -A
    blk[:n, n:] = Sigma
    blk[n:, n:] = A.T
    F = expm(blk * tau)
    return F[n:, n:].T @ F[:n, n:]


def stationary_cov(A, Sigma):
    """Stationary covariance C solving A C + C A^T + Sigma = 0."""
    return solve_continuous_lyapunov(np.asarray(A, float), -np.asarray(Sigma, float))


def ro_skill(A_obs, Sigma_obs, A_model, Sigma_model, leads, var_index=0, match_variance=False):
    """Infinite-ensemble skill and SNP measures for variable var_index (0 = T).

    match_variance : rescale the model noise so the model's stationary variance of the
        chosen variable equals the observed one (as a tuned model would), so that RPC
        reflects dynamics rather than a variance bias.
    Returns dict of arrays over leads: acc_inf (model ensemble mean vs obs), acc_perfect_inf
    (model vs itself), rpc and rsc.
    """
    A_o, A_m = np.asarray(A_obs, float), np.asarray(A_model, float)
    S_o, S_m = np.asarray(Sigma_obs, float), np.asarray(Sigma_model, float)
    C0 = stationary_cov(A_o, S_o)  # stationary covariance of observations
    if match_variance:
        Cm = stationary_cov(A_m, S_m)
        S_m = S_m * C0[var_index, var_index] / Cm[var_index, var_index]
    e = np.zeros(A_o.shape[0])
    e[var_index] = 1.0
    out = {k: [] for k in ("acc_inf", "acc_perfect_inf", "rpc", "rsc")}
    for tau in np.atleast_1d(leads):
        Mo, Mm = expm(A_o * tau), expm(A_m * tau)
        cov = e @ Mm @ C0 @ Mo.T @ e
        var_sig = e @ Mm @ C0 @ Mm.T @ e
        var_noise = e @ _noise_cov(A_m, S_m, tau) @ e
        var_obs = C0[var_index, var_index]
        a = cov / np.sqrt(var_sig * var_obs)
        p = np.sqrt(var_sig / (var_sig + var_noise))
        out["acc_inf"].append(a)
        out["acc_perfect_inf"].append(p)
        out["rpc"].append(a / p)
        out["rsc"].append((a / p) ** 2)
    out = {k: np.asarray(v) for k, v in out.items()}
    out["lead"] = np.atleast_1d(leads)
    return out
