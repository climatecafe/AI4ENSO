"""Conceptual ENSO models beyond the stochastic recharge oscillator.

LinearRO          Two-variable linear recharge oscillator, Burgers et al. (2005) form,
                  dT/dt = R T + F1 h,  dh/dt = -F2 T - eps h  (+ optional noise).
                  Recharge-discharge of equatorial heat content (Jin 1997).
DelayedOscillator Suarez & Schopf (1988) delayed-action oscillator, nondimensional:
                  dT/dt = T - T^3 - alpha T(t - delta)  (+ optional noise).
                  Reflected Rossby/Kelvin waves provide delayed negative feedback.

For the seasonally varying, state-dependent-noise form used in Bai et al. (2026), see
ai4enso.dynamics.stochastic_ro.
"""
from dataclasses import dataclass

import numpy as np

from ..snp.ro_predictability import ro_matrix, growth_and_period


@dataclass
class LinearRO:
    R: float = -0.03        # SST growth/damping rate (1/month)
    F1: float = 7.48        # thermocline feedback (degC / m / month)
    F2: float = 0.0042      # recharge efficiency (m / degC / month)
    eps: float = 0.0        # damping of h (1/month)
    sigma_T: float = 0.18   # noise on T (degC / month^0.5)
    sigma_h: float = 0.007  # noise on h (m / month^0.5)

    @property
    def A(self):
        return ro_matrix(self.R, self.F1, self.F2, self.eps)

    def growth_rate(self):
        """Growth rate of the ENSO mode (1/month); negative means damped."""
        return growth_and_period(self.A)[0]

    def period(self):
        """Period of the ENSO mode (months)."""
        return growth_and_period(self.A)[1]

    def simulate(self, n_months, dt=1 / 30, T0=0.0, h0=0.0, seed=0, noise=True):
        """Euler-Maruyama integration; returns monthly-mean T and h."""
        rng = np.random.default_rng(seed)
        steps = int(round(1 / dt))
        T, h = T0, h0
        out = np.empty((n_months, 2))
        for k in range(n_months):
            acc = np.zeros(2)
            for _ in range(steps):
                dT = (self.R * T + self.F1 * h) * dt
                dh = (-self.F2 * T - self.eps * h) * dt
                if noise:
                    dT += self.sigma_T * np.sqrt(dt) * rng.standard_normal()
                    dh += self.sigma_h * np.sqrt(dt) * rng.standard_normal()
                T, h = T + dT, h + dh
                acc += (T, h)
            out[k] = acc / steps
        return out[:, 0], out[:, 1]


@dataclass
class DelayedOscillator:
    alpha: float = 0.75   # strength of the delayed (wave-reflection) feedback, < 1
    delta: float = 6.0    # delay (nondimensional time)
    sigma: float = 0.0    # optional additive noise

    def simulate(self, n_steps, dt=0.01, T0=0.1, seed=0, record_every=10):
        """Integrate with a constant initial history T(t <= 0) = T0."""
        rng = np.random.default_rng(seed)
        lag = int(round(self.delta / dt))
        n_total = n_steps * record_every
        T = np.empty(n_total + lag + 1)
        T[:lag + 1] = T0
        for k in range(lag, lag + n_total):
            dT = (T[k] - T[k] ** 3 - self.alpha * T[k - lag]) * dt
            if self.sigma:
                dT += self.sigma * np.sqrt(dt) * rng.standard_normal()
            T[k + 1] = T[k] + dT
        t = np.arange(n_steps) * dt * record_every
        return t, T[lag + 1::record_every][:n_steps]


def dominant_period(x, dt=1.0):
    """Period (in units of dt) of the largest spectral peak, excluding the mean."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    f = np.fft.rfftfreq(len(x), dt)
    p = np.abs(np.fft.rfft(x)) ** 2
    k = np.argmax(p[1:]) + 1
    return float(1 / f[k])
