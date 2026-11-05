"""ENSO forecast benchmarks and an explainable ML predictor.

On stochastic recharge oscillator data ("observations": 600 years, first 400 for training):
- damped persistence, a linear inverse model (LIM), model analogs drawn from a separate
  1000-year run of the same oscillator, and a neural-network (MLP) predictor;
- explainable AI for the MLP: which inputs (SST history, thermocline history, calendar
  month) matter at short and long leads? Recharge theory predicts that heat content
  dominates at long leads.

Run: python examples/04_forecast_benchmarks.py   (about 1-2 minutes)
"""
import os
import sys
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from ai4enso.dynamics import SROParams, simulate
from ai4enso.forecast import LIM, ModelAnalog, build_library, lag_autocorrelation
from ai4enso import xai

warnings.filterwarnings("ignore", category=UserWarning)
LEADS = np.arange(1, 25)
ML_LEADS = [3, 6, 12, 18, 24]
N_LAG = 3


def features(T, h, months, t):
    """Inputs at time indices t: T and h at lags 0..N_LAG-1, plus calendar month."""
    cols = [T[t - k] for k in range(N_LAG)] + [h[t - k] for k in range(N_LAG)]
    ang = 2 * np.pi * months[t] / 12
    return np.column_stack(cols + [np.sin(ang), np.cos(ang)])


GROUPS = {"SST history": list(range(N_LAG)), "Thermocline history": list(range(N_LAG, 2 * N_LAG)),
          "Calendar month": [2 * N_LAG, 2 * N_LAG + 1]}


def main(outdir=os.path.join(HERE, "figures")):
    os.makedirs(outdir, exist_ok=True)
    p = SROParams(sigma_T=0.18, sigma_h=0.007)
    T, h, m = simulate(p, 12 * 600, n_runs=1, seed=21, spinup_months=240)
    T, h = T[0], h[0]
    sT, sh = T.std(), h.std()
    Tn, hn = T / sT, h / sh
    n_train = 12 * 400
    t_tr = np.arange(N_LAG, n_train - LEADS[-1])
    t_te = np.arange(n_train + N_LAG, len(T) - LEADS[-1])
    target = lambda t, L: Tn[t + L]
    skill = {}

    # damped persistence
    r = lag_autocorrelation(Tn[:n_train], LEADS[-1])
    skill["Damped persistence"] = [np.corrcoef(Tn[t_te] * r[L - 1], target(t_te, L))[0, 1] for L in LEADS]

    # LIM on (T, h)
    lim = LIM().fit(np.column_stack([Tn[:n_train], hn[:n_train]]))
    fc = lim.forecast(np.column_stack([Tn[t_te], hn[t_te]]), LEADS)[..., 0]
    skill["LIM (T, h)"] = [np.corrcoef(fc[:, i], target(t_te, L))[0, 1] for i, L in enumerate(LEADS)]

    # model analogs from an independent long run, matched on (T, h) and calendar month
    TL, hL, mL = simulate(p, 12 * 100, n_runs=10, seed=99, spinup_months=240)
    preds = np.zeros((len(t_te), len(LEADS)))
    for cm in range(1, 13):
        lib_s, lib_f = [], []
        for i in range(TL.shape[0]):
            s_, f_ = build_library(np.column_stack([TL[i] / sT, hL[i] / sh]), mL, LEADS[-1], start_month=cm % 12 + 1)
            lib_s.append(s_)
            lib_f.append(f_)
        an = ModelAnalog(k=30).fit(np.concatenate(lib_s), np.concatenate(lib_f))
        sel = np.where(m[t_te] == cm)[0]
        ens, _ = an.forecast(np.column_stack([Tn[t_te[sel]], hn[t_te[sel]]]))
        preds[sel] = ens[..., 0].mean(axis=1)
    skill["Model analogs"] = [np.corrcoef(preds[:, i], target(t_te, L))[0, 1] for i, L in enumerate(LEADS)]

    # MLP predictor, one network per lead, with XAI
    Xtr, Xte = features(Tn, hn, m, t_tr), features(Tn, hn, m, t_te)
    ml, imp, ig = [], {}, {}
    for L in ML_LEADS:
        net = MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-3, max_iter=500,
                           early_stopping=True, random_state=0).fit(Xtr, target(t_tr, L))
        yhat = net.predict(Xte)
        ml.append(np.corrcoef(yhat, target(t_te, L))[0, 1])
        if L in (3, 12):
            imp[L] = xai.permutation_importance(net.predict, Xte, target(t_te, L), groups=GROUPS, n_repeats=5)
            a = np.abs(xai.integrated_gradients(net.predict, Xte[:500], steps=16))
            ig[L] = {g: float(a[:, c].sum(axis=1).mean()) for g, c in GROUPS.items()}
    skill["Neural network (MLP)"] = ml

    print("Correlation skill on 200 test years:")
    print(f"{'lead (months)':<22}" + "".join(f"{L:>7}" for L in ML_LEADS))
    for k, v in skill.items():
        vals = v if k.startswith("Neural") else [v[L - 1] for L in ML_LEADS]
        print(f"{k:<22}" + "".join(f"{x:7.2f}" for x in vals))
    print("\nPermutation importance (drop in correlation when shuffled):")
    for L in (3, 12):
        print(f"  lead {L:2d}: " + ", ".join(f"{g} {v:.2f}" for g, v in imp[L].items()))
    print("Mean |integrated gradients| per input group:")
    for L in (3, 12):
        print(f"  lead {L:2d}: " + ", ".join(f"{g} {v:.2f}" for g, v in ig[L].items()))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    styles = {"Damped persistence": ("0.6", "-"), "LIM (T, h)": ("tab:blue", "-"),
              "Model analogs": ("tab:green", "-"), "Neural network (MLP)": ("tab:red", "o")}
    for k, v in skill.items():
        col, ls = styles[k]
        if k.startswith("Neural"):
            ax[0].plot(ML_LEADS, v, "o-", color=col, label=k)
        else:
            ax[0].plot(LEADS, v, ls, color=col, label=k)
    ax[0].set_xlabel("Lead (months)")
    ax[0].set_ylabel("Correlation with Nino 3.4")
    ax[0].set_title("Benchmark forecasts (synthetic recharge-oscillator ENSO)", fontsize=10)
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    g = list(GROUPS)
    x = np.arange(len(g))
    ax[1].bar(x - 0.2, [imp[3][k] for k in g], 0.4, color="0.6", label="Lead 3 months")
    ax[1].bar(x + 0.2, [imp[12][k] for k in g], 0.4, color="k", label="Lead 12 months")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(g)
    ax[1].set_ylabel("Drop in correlation when shuffled")
    ax[1].set_title("Explainable AI: what the network relies on", fontsize=10)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "04_forecast_benchmarks.png"), dpi=150)
    print(f"Figure written to {outdir}/")


if __name__ == "__main__":
    main()
