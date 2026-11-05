"""ENSO indices on a synthetic SST field: Nino regions, ONI, relative Nino 3.4, E/C,
event detection and multi-year events.

The synthetic field combines an eastern-Pacific (E) and a central-Pacific (C) mode driven
by a stochastic recharge oscillator, a uniform warming trend and noise. Replace it with
real SST anomalies (e.g. ERSSTv5) on a (time, lat, lon) grid to use the same calls.

Run: python examples/01_enso_indices.py
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from ai4enso import indices as I
from ai4enso.dynamics import SROParams, simulate


def synthetic_sst(n_years=80, seed=0):
    lat = np.arange(-30, 30.5, 2.0)
    lon = np.arange(100, 300, 2.0)
    T, _, months = simulate(SROParams(sigma_T=0.18, sigma_h=0.007), n_years * 12,
                            n_runs=1, seed=seed, spinup_months=240)
    T = T[0] / T[0].std()
    rng = np.random.default_rng(seed)
    c = 0.6 * T + 0.8 * np.convolve(rng.standard_normal(T.size), np.ones(6) / 6, "same") * 2
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    east = np.exp(-((LO - 250) / 25) ** 2 - (LA / 6) ** 2)
    cent = np.exp(-((LO - 190) / 20) ** 2 - (LA / 6) ** 2)
    t = np.arange(T.size)
    trend = 0.01 / 12 * t  # 1 C per century, uniform
    sst = (T[:, None, None] * east + c[:, None, None] * cent * 0.8
           + trend[:, None, None] + 0.15 * rng.standard_normal((T.size, lat.size, lon.size)))
    return sst, lat, lon, months


def main(outdir=os.path.join(HERE, "figures")):
    os.makedirs(outdir, exist_ok=True)
    sst, lat, lon, months = synthetic_sst()
    years = np.arange(len(months)) / 12

    idx = {r: I.nino_index(sst, lat, lon, r) for r in ("nino12", "nino3", "nino34", "nino4")}
    oni = I.oni(idx["nino34"])
    roni = I.oni(I.relative_nino34(sst, lat, lon))
    # remove the tropical-mean anomaly first so the EOFs describe ENSO, not the warming trend
    trop = I.area_mean(sst, lat, lon, I.REGIONS["tropics"])
    ec = I.e_c_indices(sst - trop[:, None, None], lat, lon)
    tni = I.tni(idx["nino12"], idx["nino4"])
    emi = I.emi(sst, lat, lon)
    labels, events = I.classify_events(roni)
    djf, _ = I.winter_means(roni, months)
    my = I.multiyear_events(djf / np.nanstd(djf))

    print("Correlations with Nino 3.4:")
    for k in ("nino12", "nino3", "nino4"):
        print(f"  {k:<7} {np.corrcoef(idx[k], idx['nino34'])[0, 1]:.2f}")
    print(f"  E       {np.corrcoef(ec['E'], idx['nino34'])[0, 1]:.2f}   C {np.corrcoef(ec['C'], idx['nino34'])[0, 1]:.2f}")
    print(f"  TNI     {np.corrcoef(np.nan_to_num(tni), idx['nino34'])[0, 1]:.2f}   EMI {np.corrcoef(emi, idx['nino34'])[0, 1]:.2f}")
    slope = lambda x: np.polyfit(years[np.isfinite(x)], x[np.isfinite(x)], 1)[0] * 100
    print(f"Trend per century: ONI {slope(oni):.2f} C, relative ONI {slope(roni):.2f} C")
    print(f"Events (relative ONI): {sum(e['sign'] > 0 for e in events)} El Nino, "
          f"{sum(e['sign'] < 0 for e in events)} La Nina")
    print("Multi-year events (DJF, sd units):", I.multiyear_summary(my))

    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(years, oni, color="0.5", lw=1, label="ONI (raw Nino 3.4)")
    ax[0].plot(years, roni, "k", lw=1.2, label="Relative ONI (trend removed)")
    ax[0].fill_between(years, 0, roni, where=labels > 0, color="tab:red", alpha=0.4, label="El Nino")
    ax[0].fill_between(years, 0, roni, where=labels < 0, color="tab:blue", alpha=0.4, label="La Nina")
    for th in (0.5, -0.5):
        ax[0].axhline(th, color="0.6", lw=0.8, ls="--")
    ax[0].set_ylabel("C")
    ax[0].legend(ncol=4, fontsize=8, loc="upper left")
    ax[0].set_title("ONI vs relative ONI, with NOAA-style event detection")
    ax[1].plot(years, ec["E"], "tab:red", lw=1, label="E index (eastern Pacific)")
    ax[1].plot(years, ec["C"], "tab:green", lw=1, label="C index (central Pacific)")
    ax[1].legend(fontsize=8, loc="upper left")
    ax[1].set_ylabel("sd")
    ax[1].set_title("E and C indices (Takahashi et al. 2011), from SST minus the tropical mean")
    ax[2].plot(years, tni, "tab:purple", lw=1, label="TNI")
    ax[2].plot(years, emi / emi.std(), "tab:orange", lw=1, label="EMI (standardized)")
    ax[2].legend(fontsize=8, loc="upper left")
    ax[2].set_ylabel("sd")
    ax[2].set_xlabel("Year (synthetic data)")
    ax[2].set_title("Flavour indices: Trans-Nino Index and El Nino Modoki Index")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "01_enso_indices.png"), dpi=150)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    for a, k, t in zip(ax, ("eof1", "eof2"), ("EOF1", "EOF2")):
        m = a.pcolormesh(lon, lat, ec[k], cmap="RdBu_r", shading="auto",
                         vmin=-np.nanmax(abs(ec[k])), vmax=np.nanmax(abs(ec[k])))
        a.set_title(f"{t} ({ec['variance_fraction'][int(k[-1]) - 1] * 100:.0f}% of variance)")
        a.set_xlabel("Longitude (E)")
        fig.colorbar(m, ax=a)
    ax[0].set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "01_eofs.png"), dpi=150)
    print(f"Figures written to {outdir}/")


if __name__ == "__main__":
    main()
