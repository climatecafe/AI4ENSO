"""Template: turn SPEAR or CESM2 hindcast files into emulator inputs.

UNTESTED TEMPLATE. It needs xarray and real files; adapt variable names and paths.

Target arrays expected by the emulator and diagnostics:
    T[start, member, lead]  Nino-3.4 SST anomaly (degC)
    h[start, member, lead]  equatorial-mean (120E-80W, 5S-5N) SSH or thermocline-depth anomaly
    T_init[start], h_init[start]  observed or reanalysis state just before initialization

Typical sources
    SPEAR decadal hindcasts (via GFDL): January starts, 10 members, 120 months
    CESM2 SMYLE (NCAR GDEX d651065): Feb/May/Aug/Nov starts, 20 members, 24 months
    CESM2-DP (NCAR GDEX d651082): November starts 1958-2023, 20 members, 122 months
Anomalies must be computed relative to a lead-dependent climatology (drift correction).
"""
import numpy as np


def box_mean(da, lat=(-5, 5), lon=(190, 240), lat_name="lat", lon_name="lon"):
    """Area-weighted mean over a lat/lon box (lon in 0-360). Nino-3.4 is 190-240E."""
    sub = da.sel({lat_name: slice(*lat), lon_name: slice(*lon)})
    w = np.cos(np.deg2rad(sub[lat_name]))
    return sub.weighted(w).mean((lat_name, lon_name))


def drift_corrected_anomaly(x):
    """x: array (start, member, lead). Remove the mean over starts and members at each lead,
    the standard drift correction for initialized hindcasts."""
    return x - x.mean(axis=(0, 1), keepdims=True)


def load_hindcast_index(paths, var, box, lead_dim="lead", member_dim="member", start_dim="start"):
    """Example loader. paths: list of NetCDF files covering all starts and members.

    Returns a numpy array (start, member, lead) of the box-mean index.
    """
    import xarray as xr  # imported here so the rest of the package runs without xarray

    ds = xr.open_mfdataset(paths, combine="by_coords")
    idx = box_mean(ds[var], **box)
    idx = idx.transpose(start_dim, member_dim, lead_dim).values
    return drift_corrected_anomaly(idx)


# Example (adapt to the actual SPEAR or CESM2 file layout):
# T = load_hindcast_index(sst_files, "tos", dict(lat=(-5, 5), lon=(190, 240)))
# h = load_hindcast_index(ssh_files, "zos", dict(lat=(-5, 5), lon=(120, 280)))
