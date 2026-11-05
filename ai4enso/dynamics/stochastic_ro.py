"""Stochastic recharge oscillator (SRO) used as a toy "truth" and a toy "forecast model".

Equations follow Bai et al. (2026, GRL), after Vialard et al. (2025):

    dT/dt = R(m) T + F1 h + sigma_T * w_T(t) * [1 + B H(T) T]
    dh/dt = -F2 T + sigma_h * w_h(t)

T: Nino-3.4 SST anomaly (degC); h: equatorial thermocline-depth anomaly (m, SSH proxy).
Time unit: months. R(m) varies with calendar month to give ENSO phase locking.

Running two SRO configurations, one as "observations" and one as a biased "forecast
model", produces hindcast ensembles that show a signal-to-noise paradox (RPC > 1). That
gives a cheap, fully controlled test bed for the emulator and diagnostics before they
are applied to SPEAR or CESM2 output.
"""
from dataclasses import dataclass, replace

import numpy as np

DAYS_PER_MONTH = 30


@dataclass
class SROParams:
    R0: float = -0.03      # annual-mean growth/damping rate (1/month)
    Ra: float = 0.07       # seasonal amplitude of R (1/month)
    R_peak_month: int = 9  # calendar month (1-12) of maximum growth
    F1: float = 7.48       # degC / m / month
    F2: float = 0.0042     # m / degC / month
    sigma_T: float = 0.12  # degC / month (noise amplitude on T)
    sigma_h: float = 0.0055  # m / month (noise amplitude on h)
    B: float = 0.5         # state-dependent (multiplicative) noise for T > 0

    def R(self, month):
        """Growth rate for calendar month(s) 1..12 (array or scalar)."""
        return self.R0 + self.Ra * np.cos(2 * np.pi * (np.asarray(month) - self.R_peak_month) / 12.0)

    def but(self, **kw):
        return replace(self, **kw)


def _step(T, h, month, p, dt, rng):
    """One Euler-Maruyama step for arrays T, h."""
    wT = rng.standard_normal(T.shape) * np.sqrt(dt)
    wh = rng.standard_normal(h.shape) * np.sqrt(dt)
    mult = 1.0 + p.B * np.where(T > 0, T, 0.0)
    T_new = T + (p.R(month) * T + p.F1 * h) * dt + p.sigma_T * mult * wT
    h_new = h + (-p.F2 * T) * dt + p.sigma_h * wh
    return T_new, h_new


def simulate(p, n_months, start_month=1, T0=0.0, h0=0.0, n_runs=1, seed=0, spinup_months=0):
    """Integrate the SRO and return monthly means.

    Returns
    -------
    T, h : arrays of shape (n_runs, n_months)
    months : calendar month (1..12) of each output month, shape (n_months,)
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / DAYS_PER_MONTH
    T = np.full(n_runs, T0, dtype=float)
    h = np.full(n_runs, h0, dtype=float)
    total = spinup_months + n_months
    out_T = np.empty((n_runs, n_months))
    out_h = np.empty((n_runs, n_months))
    months = np.empty(n_months, dtype=int)
    for k in range(total):
        cal = (start_month - 1 + k - spinup_months) % 12 + 1
        acc_T = np.zeros(n_runs)
        acc_h = np.zeros(n_runs)
        for _ in range(DAYS_PER_MONTH):
            T, h = _step(T, h, cal, p, dt, rng)
            acc_T += T
            acc_h += h
        if k >= spinup_months:
            j = k - spinup_months
            out_T[:, j] = acc_T / DAYS_PER_MONTH
            out_h[:, j] = acc_h / DAYS_PER_MONTH
            months[j] = cal
    return out_T, out_h, months


def make_hindcasts(model_p, T_init, h_init, start_month, n_members, lead_months,
                   init_noise=(0.02, 0.0005), seed=1):
    """Initialized ensemble hindcasts from given initial states.

    T_init, h_init : arrays (n_starts,) of initial conditions (taken from "observations").
    Returns T, h of shape (n_starts, n_members, lead_months); lead 0 is the first month
    after initialization.
    """
    rng = np.random.default_rng(seed)
    n_starts = len(T_init)
    T0 = np.repeat(T_init[:, None], n_members, axis=1) + init_noise[0] * rng.standard_normal((n_starts, n_members))
    h0 = np.repeat(h_init[:, None], n_members, axis=1) + init_noise[1] * rng.standard_normal((n_starts, n_members))
    dt = 1.0 / DAYS_PER_MONTH
    T, h = T0.copy(), h0.copy()
    out_T = np.empty((n_starts, n_members, lead_months))
    out_h = np.empty((n_starts, n_members, lead_months))
    for k in range(lead_months):
        cal = (start_month - 1 + k) % 12 + 1
        acc_T = np.zeros_like(T)
        acc_h = np.zeros_like(h)
        for _ in range(DAYS_PER_MONTH):
            T, h = _step(T, h, cal, model_p, dt, rng)
            acc_T += T
            acc_h += h
        out_T[:, :, k] = acc_T / DAYS_PER_MONTH
        out_h[:, :, k] = acc_h / DAYS_PER_MONTH
    return out_T, out_h
