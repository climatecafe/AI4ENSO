"""Conceptual ENSO dynamics: recharge oscillators and the delayed oscillator.

1. Linear recharge oscillator (Burgers et al. 2005 form): period, growth rate, T-h phase.
2. Stochastic recharge oscillator with seasonal growth rate and state-dependent noise
   (Bai et al. 2026 form): phase locking to boreal winter.
3. Delayed oscillator (Suarez & Schopf 1988): self-sustained oscillation.
4. Multi-year El Nino frequency versus the thermocline noise amplitude sigma_h, the
   control identified by Bai et al. (2026) with an interactive-ensemble GCM.
5. Recovering recharge-oscillator parameters from simulated T and h.

Run: python examples/03_enso_dynamics.py   (about 1 minute)
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
from ai4enso.dynamics import (LinearRO, DelayedOscillator, SROParams, simulate, fit_ro,
                              summarize, dominant_period)


def multiyear_fraction(T, months):
    """Fraction of El Nino winters (DJF > 0.5 sd) followed by a second warm winter (> 0.4 sd)."""
    djf, _ = I.winter_means(T, months)
    z = djf / djf.std()
    first = z[:-1] > 0.5
    return float(np.mean(z[1:][first] > 0.4))


def main(outdir=os.path.join(HERE, "figures")):
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.5))

    # 1. linear RO
    ro = LinearRO()
    T, h = ro.simulate(12 * 200, seed=1)
    print(f"Linear RO: growth {ro.growth_rate():.3f}/month, period {ro.period() / 12:.2f} yr "
          f"(simulated spectral peak {dominant_period(T) / 12:.2f} yr)")
    ax[0, 0].plot(np.arange(240) / 12, T[:240], "tab:red", label="T (Nino 3.4, C)")
    ax2 = ax[0, 0].twinx()
    ax2.plot(np.arange(240) / 12, h[:240] * 100, "tab:blue", lw=1, label="h (cm)")
    ax[0, 0].set_xlabel("Year")
    ax[0, 0].set_ylabel("T (C)", color="tab:red")
    ax2.set_ylabel("h (cm)", color="tab:blue")
    ax[0, 0].set_title(f"Linear RO: h leads T (period {ro.period() / 12:.1f} yr)", fontsize=10)
    lags = np.arange(-24, 25)
    cc = [np.corrcoef(T[max(0, -l):len(T) - max(0, l)], h[max(0, l):len(h) - max(0, -l)])[0, 1] for l in lags]
    ax[0, 1].plot(lags, cc, "k")
    ax[0, 1].axvline(0, color="0.7", lw=1)
    ax[0, 1].set_xlabel("Lag (months; negative = h leads T)")
    ax[0, 1].set_ylabel("corr(T(t), h(t + lag))")
    ax[0, 1].set_title("Recharge: heat content leads SST", fontsize=10)

    # 2. stochastic RO with seasonal growth rate: phase locking
    p = SROParams(sigma_T=0.18, sigma_h=0.007)
    Ts, hs, ms = simulate(p, 12 * 300, n_runs=1, seed=2, spinup_months=240)
    Ts, hs = Ts[0], hs[0]
    sd = [Ts[ms == m].std() for m in range(1, 13)]
    ax[0, 2].bar(range(1, 13), sd, color="0.5")
    ax[0, 2].set_xticks(range(1, 13))
    ax[0, 2].set_xticklabels(list("JFMAMJJASOND"))
    ax[0, 2].set_ylabel("Std of T (C)")
    ax[0, 2].set_title("Stochastic RO: phase locking (peak in boreal winter)", fontsize=10)

    # 3. delayed oscillator
    do = DelayedOscillator()
    t, x = do.simulate(3000)
    ax[1, 0].plot(t, x, "k", lw=1)
    ax[1, 0].set_xlabel("Nondimensional time")
    ax[1, 0].set_ylabel("T")
    ax[1, 0].set_title(f"Delayed oscillator (alpha={do.alpha}, delta={do.delta})", fontsize=10)

    # 4. multi-year El Nino frequency vs sigma_h
    sig_h = np.linspace(0.003, 0.011, 9)
    frac = []
    for s in sig_h:
        Tr, _, mr = simulate(p.but(sigma_h=s), 12 * 400, n_runs=4, seed=3, spinup_months=240)
        frac.append(np.mean([multiyear_fraction(Tr[i], mr) for i in range(Tr.shape[0])]))
    ax[1, 1].plot(sig_h * 1000, frac, "ko-")
    ax[1, 1].set_xlabel("sigma_h, thermocline noise (mm/month)")
    ax[1, 1].set_ylabel("Fraction of El Nino winters\nfollowed by a second warm winter")
    ax[1, 1].set_title("Stronger thermocline noise, more multi-year El Nino\n(toy SRO, modest effect; cf. Bai et al. 2026)", fontsize=10)
    print("Multi-year El Nino fraction vs sigma_h:", " ".join(f"{s * 1000:.0f}:{f:.2f}" for s, f in zip(sig_h, frac)))

    # 5. parameter recovery
    f = summarize(fit_ro(Ts, hs, ms))
    truth = {"R_mean": p.R0, "F1": p.F1, "F2": p.F2}
    names = list(truth)
    ax[1, 2].bar(np.arange(3) - 0.2, [truth[k] / truth[k] for k in names], 0.4, color="0.6", label="true")
    ax[1, 2].bar(np.arange(3) + 0.2, [f[k] / truth[k] for k in names], 0.4, color="k", label="fitted")
    ax[1, 2].set_xticks(range(3))
    ax[1, 2].set_xticklabels(names)
    ax[1, 2].set_ylabel("Fitted / true")
    ax[1, 2].set_title("RO fit from monthly T and h\nF1, F2 recovered; R biased by monthly means", fontsize=10)
    ax[1, 2].legend(fontsize=8)
    print("RO fit:", {k: round(v, 4) for k, v in f.items()}, "truth:", truth)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "03_enso_dynamics.png"), dpi=150)
    print(f"Figure written to {outdir}/")


if __name__ == "__main__":
    main()
