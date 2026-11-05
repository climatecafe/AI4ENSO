"""Simple reference forecasts every ENSO prediction should beat."""
import numpy as np


def persistence(x0, n_leads):
    """Anomaly persistence: x(t + tau) = x(t)."""
    x0 = np.asarray(x0, dtype=float)
    return np.repeat(x0[..., None], n_leads, axis=-1)


def damped_persistence(x0, autocorr):
    """x(t + tau) = r(tau) x(t), with r(tau) the lag autocorrelation (array over leads)."""
    return np.asarray(x0, dtype=float)[..., None] * np.asarray(autocorr)[None, :]


def lag_autocorrelation(x, max_lag, months=None, start_month=None):
    """Lag autocorrelation for lags 1..max_lag, optionally only from one calendar month
    (seasonally dependent persistence, e.g. the spring barrier)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    t0 = np.arange(len(x) - max_lag)
    if start_month is not None:
        t0 = t0[np.asarray(months)[t0] == start_month]
    return np.array([np.corrcoef(x[t0], x[t0 + k])[0, 1] for k in range(1, max_lag + 1)])
