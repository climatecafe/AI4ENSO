"""Model-analog forecasting (after Ding et al. 2018; Lenssen et al. 2024).

Given a library of model states and what followed them (e.g. a long control run), find
the K library states closest to the current observed state and use their subsequent
evolution as an ensemble forecast. Cheap, nonlinear, and a strong multi-year ENSO baseline.
"""
import numpy as np


class ModelAnalog:
    def __init__(self, k=20, weights=None):
        self.k = k
        self.weights = weights  # optional per-variable weights for the distance

    def fit(self, states, futures):
        """states : (n_lib, n_vars) library states; futures : (n_lib, n_leads, ...)."""
        self.states_ = np.asarray(states, dtype=float)
        self.futures_ = np.asarray(futures, dtype=float)
        self.scale_ = self.states_.std(axis=0)
        return self

    def _dist(self, x):
        z = (self.states_ - x) / self.scale_
        w = np.ones(z.shape[1]) if self.weights is None else np.asarray(self.weights)
        return np.sqrt(((z ** 2) * w).sum(axis=1))

    def forecast(self, x0):
        """x0 : (n_starts, n_vars). Returns the analog ensemble
        (n_starts, k, n_leads, ...) and the analog indices."""
        x0 = np.atleast_2d(x0)
        idx = np.stack([np.argsort(self._dist(x))[:self.k] for x in x0])
        return self.futures_[idx], idx


def build_library(series, months, lead, start_month=None, n_lag=1):
    """Turn a long run into (states, futures) pairs.

    series : (n_times, n_vars). The state at time t is series[t - n_lag + 1 : t + 1]
    flattened; the future is series[t + 1 : t + 1 + lead, :]. If start_month is given,
    only states at the month before start_month are kept (to match a start date).
    """
    series = np.asarray(series, dtype=float)
    months = np.asarray(months)
    states, futures = [], []
    for t in range(n_lag - 1, len(series) - lead - 1):
        if start_month is not None and months[t + 1] != start_month:
            continue
        states.append(series[t - n_lag + 1:t + 1].ravel())
        futures.append(series[t + 1:t + 1 + lead])
    return np.array(states), np.array(futures)
