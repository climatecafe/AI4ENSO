"""ENSO indices from gridded fields, plus event and multi-year event detection.

Works on plain numpy arrays so it runs without xarray. Fields have shape (..., lat, lon)
with 1-D lat and lon coordinate arrays; longitudes may be 0..360 or -180..180.
Time series are 1-D monthly arrays with a matching array of calendar months (1-12).

Region boxes (lat_south, lat_north, lon_west, lon_east), longitudes in degrees east:
    Nino 1+2   10S-0,    90W-80W
    Nino 3     5S-5N,   150W-90W
    Nino 3.4   5S-5N,   170W-120W
    Nino 4     5S-5N,   160E-150W
"""
import numpy as np

REGIONS = {
    "nino12": (-10.0, 0.0, 270.0, 280.0),
    "nino3": (-5.0, 5.0, 210.0, 270.0),
    "nino34": (-5.0, 5.0, 190.0, 240.0),
    "nino4": (-5.0, 5.0, 160.0, 210.0),
    "tropics": (-20.0, 20.0, 0.0, 360.0),      # tropical mean, for relative Nino 3.4
    "emi_a": (-10.0, 10.0, 165.0, 220.0),      # El Nino Modoki boxes (Ashok et al. 2007)
    "emi_b": (-15.0, 5.0, 250.0, 290.0),
    "emi_c": (-10.0, 20.0, 125.0, 145.0),
    "wwv": (-5.0, 5.0, 120.0, 280.0),          # warm water volume (Meinen & McPhaden 2000)
    "tropical_pacific": (-10.0, 10.0, 140.0, 280.0),
}

EARTH_RADIUS = 6.371e6  # m


def _lon360(lon):
    return np.mod(np.asarray(lon, dtype=float), 360.0)


def box_mask(lat, lon, box):
    """Boolean (lat, lon) mask for a box; handles boxes that cross the dateline."""
    s, n, w, e = box
    la = np.asarray(lat, dtype=float)[:, None]
    lo = _lon360(lon)[None, :]
    in_lat = (la >= s) & (la <= n)
    w, e = w % 360.0, e % 360.0 if e != 360.0 else 360.0
    in_lon = (lo >= w) & (lo <= e) if w <= e else (lo >= w) | (lo <= e)
    return in_lat & in_lon


def area_mean(field, lat, lon, box):
    """Cos(latitude)-weighted mean over a box, ignoring NaNs (e.g. land).

    field : array (..., lat, lon). Returns array (...).
    """
    field = np.asarray(field, dtype=float)
    mask = box_mask(lat, lon, box)
    w = np.cos(np.deg2rad(np.asarray(lat, dtype=float)))[:, None] * mask
    w = np.broadcast_to(w, field.shape)
    valid = np.isfinite(field) & (w > 0)
    num = np.where(valid, field * w, 0.0).sum(axis=(-2, -1))
    den = np.where(valid, w, 0.0).sum(axis=(-2, -1))
    return num / den


def monthly_anomalies(x, months, base=None):
    """Remove the mean annual cycle from a monthly series (time on the first axis).

    months : calendar month (1-12) of each time step.
    base   : optional boolean array selecting the climatology base period.
    """
    x = np.asarray(x, dtype=float)
    months = np.asarray(months)
    base = np.ones(len(months), bool) if base is None else np.asarray(base)
    out = np.empty_like(x)
    for m in range(1, 13):
        sel = months == m
        out[sel] = x[sel] - np.nanmean(x[sel & base], axis=0)
    return out


def running_mean(x, window=3):
    """Centered running mean along the first axis; edges are NaN."""
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    h = window // 2
    c = np.cumsum(np.insert(x, 0, 0.0, axis=0), axis=0)
    out[h:len(x) - h] = (c[window:] - c[:-window]) / window
    return out


def standardize(x):
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / np.nanstd(x)


# ---- SST indices -------------------------------------------------------------------------

def nino_index(sst_anom, lat, lon, region="nino34"):
    """Area-mean SST anomaly over a Nino region ('nino12', 'nino3', 'nino34', 'nino4')."""
    return area_mean(sst_anom, lat, lon, REGIONS[region])


def oni(nino34_anom):
    """Oceanic Nino Index: 3-month centered running mean of Nino 3.4 anomalies.

    NOAA computes the anomalies against centered 30-year base periods updated every 5 years;
    pass anomalies computed that way for an exact match.
    """
    return running_mean(nino34_anom, 3)


def relative_nino34(sst_anom, lat, lon):
    """Relative Nino 3.4 index (van Oldenborgh et al. 2021).

    Nino 3.4 anomaly minus the tropical-mean (20S-20N) anomaly, rescaled to the standard
    deviation of the original Nino 3.4 index. Removes the uniform tropical warming trend.
    """
    n34 = nino_index(sst_anom, lat, lon, "nino34")
    trop = area_mean(sst_anom, lat, lon, REGIONS["tropics"])
    rel = n34 - trop
    return rel * np.nanstd(n34) / np.nanstd(rel)


def tni(nino12, nino4, smooth=5):
    """Trans-Nino Index (Trenberth & Stepaniak 2001): standardized Nino 1+2 minus
    standardized Nino 4, optionally smoothed with a running mean."""
    x = standardize(nino12) - standardize(nino4)
    return running_mean(x, smooth) if smooth and smooth > 1 else x


def emi(sst_anom, lat, lon):
    """El Nino Modoki Index (Ashok et al. 2007): A - 0.5 B - 0.5 C."""
    a = area_mean(sst_anom, lat, lon, REGIONS["emi_a"])
    b = area_mean(sst_anom, lat, lon, REGIONS["emi_b"])
    c = area_mean(sst_anom, lat, lon, REGIONS["emi_c"])
    return a - 0.5 * b - 0.5 * c


def e_c_indices(sst_anom, lat, lon, box=REGIONS["tropical_pacific"]):
    """E (eastern Pacific) and C (central Pacific) indices after Takahashi et al. (2011).

    EOF analysis of tropical Pacific SST anomalies, then E = (PC1 - PC2) / sqrt(2) and
    C = (PC1 + PC2) / sqrt(2). PCs are normalized to unit variance here, as in common
    implementations; check against Takahashi et al. (2011) if exact amplitudes matter.
    Sign conventions as implemented here: PC1 correlates positively with Nino 3.4, and
    EOF2 is positive over the Nino 4 region, so C peaks for central-Pacific warming.
    Returns dict with E, C, pc1, pc2, eof1, eof2 (maps on the full grid, NaN outside box)
    and the explained-variance fractions.
    """
    x = np.asarray(sst_anom, dtype=float)
    nt = x.shape[0]
    mask = box_mask(lat, lon, box)
    w = np.sqrt(np.cos(np.deg2rad(np.asarray(lat, dtype=float))))[:, None] * np.ones((1, len(lon)))
    flat = (x * w)[:, mask]
    good = np.all(np.isfinite(flat), axis=0)
    X = flat[:, good] - flat[:, good].mean(axis=0)
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    pcs = u[:, :2] * s[:2]
    frac = s[:2] ** 2 / np.sum(s ** 2)
    eofs = []
    for k in range(2):
        m = np.full(mask.shape, np.nan)
        vals = np.full(good.shape, np.nan)
        vals[good] = vt[k]
        m[mask] = vals
        eofs.append(m / w)
    n34 = nino_index(x, lat, lon, "nino34")
    if np.corrcoef(pcs[:, 0], n34)[0, 1] < 0:
        pcs[:, 0] *= -1
        eofs[0] *= -1
    if np.nanmean(np.where(box_mask(lat, lon, REGIONS["nino4"]), eofs[1], np.nan)) < 0:
        pcs[:, 1] *= -1
        eofs[1] *= -1
    p1, p2 = pcs[:, 0] / pcs[:, 0].std(), pcs[:, 1] / pcs[:, 1].std()
    return {"E": (p1 - p2) / np.sqrt(2), "C": (p1 + p2) / np.sqrt(2), "pc1": p1, "pc2": p2,
            "eof1": eofs[0], "eof2": eofs[1], "variance_fraction": frac, "n_times": nt}


# ---- atmospheric and subsurface indices -----------------------------------------------------

def soi(slp_tahiti, slp_darwin, months):
    """Southern Oscillation Index: standardized Tahiti minus standardized Darwin SLP
    anomalies, divided by the standard deviation of the difference."""
    t = monthly_anomalies(slp_tahiti, months)
    d = monthly_anomalies(slp_darwin, months)
    diff = t / np.nanstd(t) - d / np.nanstd(d)
    return diff / np.nanstd(diff)


def isotherm_depth(temp, depth, iso=20.0):
    """Depth of an isotherm (default 20 C, 'D20') by linear interpolation.

    temp  : array (..., depth) with depth as the LAST axis, warm at the surface.
    depth : 1-D array of positive depths (m), increasing.
    Returns array (...) of depths; NaN where the column never crosses the isotherm.
    """
    temp = np.asarray(temp, dtype=float)
    depth = np.asarray(depth, dtype=float)
    above = temp >= iso
    cross = above[..., :-1] & ~above[..., 1:]
    k = np.argmax(cross, axis=-1)
    has = cross.any(axis=-1)
    t0 = np.take_along_axis(temp, k[..., None], -1)[..., 0]
    t1 = np.take_along_axis(temp, (k + 1)[..., None], -1)[..., 0]
    z0, z1 = depth[k], depth[k + 1]
    z = z0 + (iso - t0) * (z1 - z0) / (t1 - t0)
    return np.where(has, z, np.nan)


def warm_water_volume(d20, lat, lon, box=REGIONS["wwv"]):
    """Warm water volume: volume above the 20 C isotherm over 5S-5N, 120E-80W, in m^3
    (Meinen & McPhaden 2000). d20 : array (..., lat, lon) of D20 in metres.

    Assumes a regular grid; cells whose centres fall in the box are counted whole, so the
    result depends slightly on the grid near the box edges.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    dlat = np.deg2rad(np.abs(np.gradient(lat)))
    dlon = np.deg2rad(np.abs(np.gradient(_lon360(lon))))
    area = (EARTH_RADIUS ** 2 * np.cos(np.deg2rad(lat))[:, None] * dlat[:, None] * dlon[None, :])
    mask = box_mask(lat, lon, box)
    d = np.where(np.isfinite(d20), d20, 0.0)
    return (d * area * mask).sum(axis=(-2, -1))


# ---- events -----------------------------------------------------------------------------------

def classify_events(oni_series, threshold=0.5, min_len=5):
    """Label months as El Nino (+1), La Nina (-1) or neutral (0).

    NOAA convention: ONI at or beyond +/-threshold for at least min_len consecutive
    overlapping seasons. Returns (labels, events), events = list of dicts with
    start, end (inclusive indices), sign and peak value.
    """
    x = np.asarray(oni_series, dtype=float)
    labels = np.zeros(len(x), dtype=int)
    events = []
    for sign in (1, -1):
        hit = np.nan_to_num(sign * x, nan=-np.inf) >= threshold
        k = 0
        while k < len(x):
            if hit[k]:
                j = k
                while j + 1 < len(x) and hit[j + 1]:
                    j += 1
                if j - k + 1 >= min_len:
                    labels[k:j + 1] = sign
                    seg = x[k:j + 1]
                    events.append({"start": k, "end": j, "sign": sign,
                                   "peak": float(seg[np.argmax(sign * seg)])})
                k = j + 1
            else:
                k += 1
    events.sort(key=lambda e: e["start"])
    return labels, events


def winter_means(x, months, winter=(12, 1, 2)):
    """Mean over each boreal winter (default DJF). The winter is labelled by the year
    index of its December. Only complete winters are kept.

    x : monthly series; months : calendar months. Returns (values, start indices).
    """
    x = np.asarray(x, dtype=float)
    months = np.asarray(months)
    vals, starts = [], []
    for k in np.where(months == winter[0])[0]:
        idx = k + np.arange(len(winter))
        if idx[-1] < len(x) and np.all(months[idx] == np.array(winter)):
            vals.append(np.mean(x[idx]))
            starts.append(int(k))
    return np.array(vals), np.array(starts)


def multiyear_events(winter_values, first_threshold=0.5, next_threshold=0.4):
    """Find single- and multi-year events in a series of winter-mean index values.

    An event starts in a winter beyond first_threshold (in index units, e.g. 0.5 C or
    0.5 sd) and continues while following winters stay beyond next_threshold with the
    same sign. Defaults follow Bai et al. (2026) when the index is in sd units.
    Returns a list of dicts: first (winter index), duration (winters), sign.
    """
    w = np.asarray(winter_values, dtype=float)
    events, k = [], 0
    while k < len(w):
        sign = 1 if w[k] >= first_threshold else -1 if w[k] <= -first_threshold else 0
        if sign == 0:
            k += 1
            continue
        j = k
        while j + 1 < len(w) and sign * w[j + 1] >= next_threshold:
            j += 1
        events.append({"first": k, "duration": j - k + 1, "sign": sign})
        k = j + 1
    return events


def multiyear_summary(events):
    """Counts of single- and multi-year El Nino and La Nina events."""
    out = {}
    for sign, name in ((1, "el_nino"), (-1, "la_nina")):
        ev = [e for e in events if e["sign"] == sign]
        out[f"{name}_total"] = len(ev)
        out[f"{name}_multiyear"] = sum(e["duration"] >= 2 for e in ev)
    return out
