"""The signal-to-noise paradox in two analytic frameworks.

(a) Markov (AR1) framework, after Zhang & Kirtman (2019): RPC depends on the persistence
    and noise variance of the model relative to observations. Closed-form curves are
    checked against Monte Carlo hindcasts.
(b) Linear recharge oscillator with constant coefficients: RPC for a model whose ENSO
    growth rate and period differ from the observed oscillator (variance matched). Jin
    (2022) highlighted growth rate and period as SNP sources; this simplified version has
    no seasonal cycle, so only the growth-rate (damping) effect appears as RPC > 1.

Run: python examples/02_snp_markov.py
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from ai4enso.snp import markov as mk, ro_predictability as rp, metrics as sm


def main(outdir=os.path.join(HERE, "figures")):
    os.makedirs(outdir, exist_ok=True)
    a_o, s_o = 0.92, 0.4          # "observed" monthly persistence and noise
    leads = np.arange(1, 25)

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))

    # (a1) RPC vs lead for models that are less persistent or noisier
    cases = [("Perfect model", a_o, s_o, "k"),
             ("Less persistent, same variance", 0.88, s_o * np.sqrt((1 - 0.88 ** 2) / (1 - a_o ** 2)), "tab:red"),
             ("Same persistence, 30% more noise", a_o, 1.3 * s_o, "tab:blue")]
    obs_mc = {}
    for name, am, sg, col in cases:
        th = mk.ar1_skill(a_o, s_o, am, sg, leads)
        ax[0, 0].plot(leads, th["rpc"], color=col, label=name)
        obs, ens = mk.simulate_ar1_hindcasts(a_o, s_o, am, sg, 20000, 30, leads[-1], seed=1)
        mc = [sm.rpc_corrected(ens[:, :, k], obs[:, k]) for k in leads[::4]]
        ax[0, 0].plot(leads[::4], mc, "o", color=col, mfc="none")
        obs_mc[name] = (th, mc)
    ax[0, 0].axhline(1, color="0.6", lw=1)
    ax[0, 0].set_xlabel("Lead (months)")
    ax[0, 0].set_ylabel("RPC")
    ax[0, 0].set_title("(a) AR1 framework: lines analytic, circles Monte Carlo\n(30 members, bias-corrected RPC)", fontsize=10)
    ax[0, 0].legend(fontsize=8)

    # (a2) ACC vs perfect-model ACC for the less persistent model
    th = mk.ar1_skill(a_o, s_o, 0.88, cases[1][2], leads, n_members=10)
    ax[0, 1].plot(leads, th["acc_inf"], "k", label="Ensemble mean vs obs, infinite ensemble")
    ax[0, 1].plot(leads, th["acc"], "k--", label="Ensemble mean vs obs, 10 members")
    ax[0, 1].plot(leads, th["acc_perfect_inf"], "tab:red", label="Model vs itself, infinite ensemble")
    ax[0, 1].plot(leads, th["acc_perfect"], "tab:red", ls="--", label="Model vs itself, 10 members")
    ax[0, 1].set_xlabel("Lead (months)")
    ax[0, 1].set_ylabel("Correlation")
    ax[0, 1].set_title("(b) Less persistent model: it predicts the real world\nbetter than it predicts itself", fontsize=10)
    ax[0, 1].legend(fontsize=8)

    # (a3) RPC map over model persistence and noise at a 6-month lead
    am_vals = np.linspace(0.80, 0.97, 60)
    sg_vals = np.linspace(0.25, 0.65, 60)
    R = mk.rpc_map(a_o, s_o, am_vals, sg_vals, 6)
    cs = ax[1, 0].contourf(sg_vals, am_vals, R, levels=np.arange(0.4, 2.01, 0.1), cmap="RdBu_r", extend="both")
    ax[1, 0].contour(sg_vals, am_vals, R, levels=[1.0], colors="k", linewidths=1.5)
    ax[1, 0].plot(s_o, a_o, "k*", ms=12, label="Observations")
    ax[1, 0].set_xlabel("Model noise amplitude (innovation std)")
    ax[1, 0].set_ylabel("Model persistence (lag-1 autocorrelation)")
    ax[1, 0].set_title("(c) RPC at 6-month lead; black line RPC = 1\nSNP (red) for weaker persistence or more noise", fontsize=10)
    ax[1, 0].legend(fontsize=8, loc="lower right")
    fig.colorbar(cs, ax=ax[1, 0], label="RPC")

    # (b) recharge oscillator: RPC over model growth rate and period
    g_o, p_o = -0.02, 42.0
    A_o = rp.matrix_from_growth_period(g_o, p_o, F1=7.48)
    S = np.diag([0.18 ** 2, 0.007 ** 2])
    growths = np.linspace(-0.06, -0.004, 25)  # growth 0 has no stationary state
    periods = np.linspace(28, 60, 25)
    RR = np.empty((growths.size, periods.size))
    for i, g in enumerate(growths):
        for j, p in enumerate(periods):
            RR[i, j] = rp.ro_skill(A_o, S, rp.matrix_from_growth_period(g, p, F1=7.48), S, [12],
                                   match_variance=True)["rpc"][0]
    cs = ax[1, 1].contourf(periods / 12, growths, RR, levels=np.arange(0.4, 2.01, 0.1), cmap="RdBu_r", extend="both")
    ax[1, 1].contour(periods / 12, growths, RR, levels=[1.0], colors="k", linewidths=1.5)
    ax[1, 1].plot(p_o / 12, g_o, "k*", ms=12, label="Observed oscillator")
    ax[1, 1].set_xlabel("Model ENSO period (years)")
    ax[1, 1].set_ylabel("Model ENSO growth rate (1/month)")
    ax[1, 1].set_title("(d) Linear RO (no seasonal cycle), Nino-3.4 RPC at 12 months\nmodel variance matched; SNP for a more damped model", fontsize=10)
    ax[1, 1].legend(fontsize=8, loc="lower right")
    fig.colorbar(cs, ax=ax[1, 1], label="RPC")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "02_snp_markov.png"), dpi=150)

    for name, (th, mc) in obs_mc.items():
        print(f"{name:<34} RPC at 9 months: analytic {th['rpc'][8]:.2f}, Monte Carlo {mc[2]:.2f}")
    i0 = np.argmin(abs(growths - g_o))
    j0 = np.argmin(abs(periods - p_o))
    print(f"RO: RPC at observed growth/period = {RR[i0, j0]:.2f}; "
          f"more damped model (growth -0.05) = {RR[np.argmin(abs(growths + 0.05)), j0]:.2f}; "
          f"longer-period model (5 yr) = {RR[i0, np.argmin(abs(periods - 60))]:.2f}")
    print(f"Figure written to {outdir}/")


if __name__ == "__main__":
    main()
