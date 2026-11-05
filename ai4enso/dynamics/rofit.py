"""Recharge-oscillator "fingerprint": fit SRO parameters from monthly T and h.

Following Vijayeta & Dommenget (2018) and Bai et al. (2026): regress monthly tendencies
of T and h on their anomalies. R is fitted per calendar month; F1, F2 are constant; the
residual standard deviations give the noise amplitudes sigma_T and sigma_h.

The same estimator is applied to observations, to physics-model hindcasts and to
emulator output, so the parameters can be compared like for like. A faithful emulator
must reproduce the model's fingerprint, not just its mean and variance.
"""
import numpy as np


def _pairs(T, h, months):
    """Centered tendencies from monthly series.

    T, h: arrays (..., n_months) of contiguous monthly values; months: (n_months,) calendar
    month of each time step. Returns flattened (T, h, dT, dh, month) at interior points.
    """
    T = np.atleast_2d(T)
    h = np.atleast_2d(h)
    T2 = T.reshape(-1, T.shape[-1])
    h2 = h.reshape(-1, h.shape[-1])
    dT = (T2[:, 2:] - T2[:, :-2]) / 2.0
    dh = (h2[:, 2:] - h2[:, :-2]) / 2.0
    Tm = T2[:, 1:-1]
    hm = h2[:, 1:-1]
    mm = np.broadcast_to(np.asarray(months)[1:-1], Tm.shape)
    return Tm.ravel(), hm.ravel(), dT.ravel(), dh.ravel(), mm.ravel()


def fit_ro(T, h, months):
    """Fit R(m) (12 values), F1, F2, sigma_T, sigma_h.

    T equation:  dT = sum_m R_m * T * 1[month=m] + F1 * h
    h equation:  dh = -F2 * T
    """
    Tm, hm, dT, dh, mm = _pairs(T, h, months)
    X = np.zeros((Tm.size, 13))
    for m in range(1, 13):
        X[mm == m, m - 1] = Tm[mm == m]
    X[:, 12] = hm
    coef, *_ = np.linalg.lstsq(X, dT, rcond=None)
    resid_T = dT - X @ coef
    F2 = -np.sum(Tm * dh) / np.sum(Tm * Tm)
    resid_h = dh + F2 * Tm
    return {
        "R": coef[:12],
        "F1": coef[12],
        "F2": F2,
        "sigma_T": resid_T.std(),
        "sigma_h": resid_h.std(),
    }


def summarize(fit):
    """Compact scalar summary used in comparison tables and plots."""
    return {
        "R_mean": float(np.mean(fit["R"])),
        "R_range": float(np.ptp(fit["R"])),
        "F1": float(fit["F1"]),
        "F2": float(fit["F2"]),
        "sigma_T": float(fit["sigma_T"]),
        "sigma_h": float(fit["sigma_h"]),
    }
