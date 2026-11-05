import numpy as np

from ai4enso.dynamics import LinearRO, DelayedOscillator, dominant_period, fit_ro, SROParams, simulate
from ai4enso.forecast import LIM, ModelAnalog, build_library, lag_autocorrelation
from ai4enso import xai


def test_linear_ro_period_and_fit():
    ro = LinearRO()
    T, h = ro.simulate(12 * 200, seed=1)
    assert abs(dominant_period(T) - ro.period()) / ro.period() < 0.25
    f = fit_ro(T, h, np.arange(len(T)) % 12 + 1)
    assert abs(f["F1"] - ro.F1) / ro.F1 < 0.15
    assert abs(f["F2"] - ro.F2) / ro.F2 < 0.15


def test_delayed_oscillator_self_sustained():
    t, x = DelayedOscillator().simulate(3000)
    assert x[-1000:].std() > 0.5


def test_lim_recovers_linear_dynamics():
    ro = LinearRO()
    T, h = ro.simulate(12 * 300, seed=2)
    X = np.column_stack([T / T.std(), h / h.std()])
    lim = LIM().fit(X, tau0=1)
    lam, _ = lim.eigenmodes()
    period = 2 * np.pi / abs(lam[0].imag)
    assert abs(period - ro.period()) / ro.period() < 0.3
    assert np.all(lim.noise_check() > -1e-6)


def test_model_analog_shapes_and_skill():
    p = SROParams(sigma_T=0.18, sigma_h=0.007)
    T, h, m = simulate(p, 12 * 300, n_runs=1, seed=3, spinup_months=240)
    X = np.column_stack([T[0], h[0]])
    states, fut = build_library(X[:2400], m[:2400], lead=12)
    an = ModelAnalog(k=15).fit(states, fut)
    ens, idx = an.forecast(X[2500:2510])
    assert ens.shape == (10, 15, 12, 2)


def test_lag_autocorrelation_decreases():
    T, _ = LinearRO().simulate(12 * 100, seed=4)
    r = lag_autocorrelation(T, 6)
    assert r[0] > r[2] > r[5]


def test_integrated_gradients_completeness_and_importance():
    w = np.array([2.0, -1.0, 0.0])
    f = lambda X: X @ w
    X = np.random.default_rng(0).standard_normal((50, 3))
    ig = xai.integrated_gradients(f, X)
    assert np.allclose(ig.sum(axis=1), f(X), atol=1e-6)
    imp = xai.permutation_importance(f, X, f(X), n_repeats=5)
    assert imp["x0"] > imp["x1"] > imp["x2"] - 1e-9
