import numpy as np

from ai4enso import indices as I

LAT = np.arange(-30, 30.5, 1.0)
LON = np.arange(0, 360, 1.0)


def _field(boxes_values, nt=1):
    f = np.zeros((nt, LAT.size, LON.size))
    for box, v in boxes_values:
        f[:, I.box_mask(LAT, LON, box)] = v
    return f


def test_nino34_recovers_box_anomaly():
    f = _field([(I.REGIONS["nino34"], 1.0)])
    assert np.isclose(I.nino_index(f, LAT, LON, "nino34")[0], 1.0)
    assert np.isclose(I.nino_index(f, LAT, LON, "nino12")[0], 0.0)


def test_longitude_conventions_agree():
    f = np.random.default_rng(0).standard_normal((3, LAT.size, LON.size))
    lon180 = np.where(LON > 180, LON - 360, LON)
    for r in ("nino12", "nino3", "nino34", "nino4"):
        assert np.allclose(I.nino_index(f, LAT, LON, r), I.nino_index(f, LAT, lon180, r))


def test_relative_nino34_removes_uniform_warming():
    t = np.arange(240)
    f = np.zeros((t.size, LAT.size, LON.size)) + 0.01 * t[:, None, None]  # uniform trend
    f[:, I.box_mask(LAT, LON, I.REGIONS["nino34"])] += np.sin(2 * np.pi * t / 48)[:, None]
    rel = I.relative_nino34(f, LAT, LON)
    raw = I.nino_index(f, LAT, LON, "nino34")
    assert abs(np.polyfit(t, rel, 1)[0]) < 0.3 * abs(np.polyfit(t, raw, 1)[0])
    assert np.corrcoef(rel, np.sin(2 * np.pi * t / 48))[0, 1] > 0.99


def test_emi_sign():
    f = _field([(I.REGIONS["emi_a"], 1.0)])
    assert I.emi(f, LAT, LON)[0] > 0


def test_e_c_indices_separate_patterns():
    # Correlated eastern and central amplitudes, so EOF1 is basin-wide and EOF2 a dipole,
    # the situation the E/C rotation is designed for.
    rng = np.random.default_rng(1)
    nt = 600
    z = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], nt)
    e_ts, c_ts = z[:, 0], z[:, 1]
    east = _field([((-5, 5, 240, 280), 1.0)])[0]
    cent = _field([((-5, 5, 170, 210), 1.0)])[0]
    f = e_ts[:, None, None] * east + c_ts[:, None, None] * cent + 0.05 * rng.standard_normal((nt, LAT.size, LON.size))
    out = I.e_c_indices(f, LAT, LON)
    rE = [np.corrcoef(out["E"], x)[0, 1] for x in (e_ts, c_ts)]
    rC = [np.corrcoef(out["C"], x)[0, 1] for x in (e_ts, c_ts)]
    assert rE[0] > 0.8 and rE[0] > rE[1]
    assert rC[1] > 0.8 and rC[1] > rC[0]


def test_classify_events_requires_five_seasons():
    x = np.zeros(30)
    x[2:9] = 1.0      # 7 months: event
    x[15:18] = -1.0   # 3 months: too short
    labels, events = I.classify_events(x)
    assert len(events) == 1 and events[0]["sign"] == 1
    assert labels[2:9].tolist() == [1] * 7 and labels[15:18].tolist() == [0] * 3


def test_multiyear_events():
    w = np.array([0.0, 1.0, 0.6, 0.1, -0.8, -0.5, -0.45, 0.0])
    ev = I.multiyear_events(w)
    assert [(e["sign"], e["duration"]) for e in ev] == [(1, 2), (-1, 3)]
    s = I.multiyear_summary(ev)
    assert s["el_nino_multiyear"] == 1 and s["la_nina_multiyear"] == 1


def test_isotherm_depth_linear_profile():
    depth = np.arange(0, 301, 10.0)
    temp = 28 - 0.05 * depth  # crosses 20 C at 160 m
    assert np.isclose(I.isotherm_depth(temp, depth), 160.0)


def test_warm_water_volume_constant_depth():
    lat = np.arange(-10, 10.5, 0.5)
    lon = np.arange(100, 300, 0.5)
    d20 = np.full((lat.size, lon.size), 100.0)
    vol = I.warm_water_volume(d20, lat, lon)
    # analytic area of 5S-5N, 120E-280E
    area = I.EARTH_RADIUS ** 2 * np.deg2rad(160) * 2 * np.sin(np.deg2rad(5))
    assert abs(vol / (100 * area) - 1) < 0.07  # edge cells counted whole


def test_running_mean_and_anomalies():
    months = np.tile(np.arange(1, 13), 10)
    x = np.cos(2 * np.pi * months / 12) + 2.0
    assert np.allclose(I.monthly_anomalies(x, months), 0.0)
    r = I.running_mean(np.arange(10.0), 3)
    assert np.isnan(r[0]) and np.isclose(r[5], 5.0)
