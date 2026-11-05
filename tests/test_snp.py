import numpy as np

from ai4enso.snp import markov as mk, ro_predictability as rp, metrics as sm


def test_markov_closed_form_matches_monte_carlo():
    ao, so, am, smv = 0.9, 0.4, 0.85, 0.5
    obs, ens = mk.simulate_ar1_hindcasts(ao, so, am, smv, 20000, 20, 6, seed=1)
    th = mk.ar1_skill(ao, so, am, smv, np.arange(1, 7), n_members=20)
    for k in (3, 6):
        e, o = ens[:, :, k], obs[:, k]
        assert abs(sm.acc_infinite(e, o) - th["acc_inf"][k - 1]) < 0.02
        assert abs(sm.perfect_model_acc(e) - th["acc_perfect"][k - 1]) < 0.02
        assert abs(sm.rpc_corrected(e, o) - th["rpc"][k - 1]) < 0.05


def test_markov_equal_variance_identity():
    ao, so, am = 0.9, 0.4, 0.8
    sig_m = so * np.sqrt((1 - am ** 2) / (1 - ao ** 2))  # same stationary variance
    out = mk.ar1_skill(ao, so, am, sig_m, 6)
    assert np.isclose(out["rpc"], (ao / am) ** 6)
    assert np.isclose(out["rsc"], out["rpc"] ** 2)


def test_perfect_model_has_no_snp():
    out = mk.ar1_skill(0.9, 0.4, 0.9, 0.4, np.arange(1, 13))
    assert np.allclose(out["rpc"], 1.0)


def test_corrected_rpc_independent_of_ensemble_size():
    obs, ens = mk.simulate_ar1_hindcasts(0.9, 0.4, 0.8, 0.55, 5000, 200, 6, seed=2)
    e, o = ens[:, :, 6], obs[:, 6]
    small = np.mean([sm.rpc_corrected(e[:, i * 5:(i + 1) * 5], o) for i in range(20)])
    large = sm.rpc_corrected(e, o)
    naive_small = np.mean([sm.rpc(e[:, i * 5:(i + 1) * 5], o) for i in range(20)])
    assert abs(small - large) < 0.1 * large
    assert naive_small < small


def test_ro_growth_period_roundtrip():
    A = rp.matrix_from_growth_period(-0.02, 40.0, F1=7.5, eps=0.01)
    g, p = rp.growth_and_period(A)
    assert np.isclose(g, -0.02) and np.isclose(p, 40.0)


def test_ro_perfect_model_rpc_is_one():
    A = rp.ro_matrix(-0.03, 7.48, 0.0042)
    S = np.diag([0.18 ** 2, 0.007 ** 2])
    assert np.allclose(rp.ro_skill(A, S, A, S, [3, 12, 24])["rpc"], 1.0)


def test_calibration_restores_reliability():
    obs, ens = mk.simulate_ar1_hindcasts(0.9, 0.4, 0.75, 0.6, 4000, 50, 6, seed=3)
    e, o = ens[:, :, 6], obs[:, 6]
    cal, s_sig, _ = sm.calibrate_ensemble(e[2000:], o[:2000], e[:2000])
    assert s_sig > 1  # weak model signal is amplified
    from ai4enso.verification import spread_skill
    assert abs(spread_skill(cal, o[2000:]) - 1) < abs(spread_skill(e[2000:], o[2000:]) - 1)
