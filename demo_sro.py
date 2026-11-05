"""End-to-end demo on synthetic data: diagnose and correct for a signal-to-noise paradox.

1. "Observations": a long stochastic recharge oscillator (SRO) run.
2. "Physics model": an SRO with biased parameters (weaker persistence, stronger noise),
   run as 20-member November-initialized hindcasts. This produces RPC > 1, i.e. an SNP.
3. Train a Gaussian (benchmark) and a diffusion emulator on the 20-member hindcasts.
4. Generate 1000-member ensembles for held-out start years.
5. Compare with a 1000-member physics ensemble (possible only in the toy setting):
   skill, RPC, model SNR, ROM fingerprint and multi-year El Nino statistics.
6. Correct for the SNP with an RPC-based rescaling estimated on training years.

Run:  python demo_sro.py            (about 5-10 minutes on a laptop CPU)
      python demo_sro.py --quick    (smaller, about 1-2 minutes)
"""
import argparse
import json
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ensosnp.sro import SROParams, simulate, make_hindcasts
from ensosnp.rofit import fit_ro, summarize
from ensosnp.emulator import GaussianEmulator, DiffusionEmulator
from ensosnp import metrics as M


def main(quick=False, outdir="results"):
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    n_years = 120 if quick else 220
    lead = 36
    start_month = 11
    n_members = 20
    n_big = 300 if quick else 1000

    # ---- 1. observations -------------------------------------------------------
    p_obs = SROParams(sigma_T=0.18, sigma_h=0.007)
    T_obs, h_obs, months = simulate(p_obs, n_years * 12, start_month=1, n_runs=1,
                                    seed=11, spinup_months=600)
    T_obs, h_obs = T_obs[0], h_obs[0]
    nov = np.where(months == start_month)[0]
    nov = nov[(nov >= 1) & (nov + lead <= T_obs.size)]
    T_init, h_init = T_obs[nov - 1], h_obs[nov - 1]  # October state initializes Nov 1
    obs_T = np.stack([T_obs[i:i + lead] for i in nov])  # (n_starts, lead)
    n_starts = len(nov)
    n_train = int(0.65 * n_starts)
    tr, te = np.arange(n_train), np.arange(n_train, n_starts)
    print(f"[obs] {n_years} yr, {n_starts} start dates ({n_train} train / {len(te)} test), "
          f"Nino3.4 std = {T_obs.std():.2f} C")

    # ---- 2. biased physics model hindcasts ------------------------------------------
    # weaker persistence (more damping) and stronger noise than "observations"
    p_mod = p_obs.but(R0=-0.08, sigma_T=0.28, sigma_h=0.010)
    hT, hh = make_hindcasts(p_mod, T_init, h_init, start_month, n_members, lead, seed=2)
    bigT, bigh = make_hindcasts(p_mod, T_init[te], h_init[te], start_month, n_big, lead, seed=3)
    print(f"[model] hindcasts: {hT.shape}, reference big ensemble on test years: {bigT.shape}")

    # ---- 3. train emulators on the 20-member training hindcasts ---------------------
    X = np.concatenate([hT[tr], hh[tr]], axis=-1).reshape(-1, 2 * lead)
    C = np.repeat(np.stack([T_init[tr], h_init[tr]], axis=1), n_members, axis=0)
    t1 = time.time()
    gauss = GaussianEmulator(n_components=48).fit(X, C)
    diff = DiffusionEmulator(n_components=48, K=40, repeats=6 if quick else 10,
                             max_iter=150 if quick else 300).fit(X, C)
    print(f"[emulator] trained in {time.time() - t1:.0f} s; PCA keeps "
          f"{diff.red_.explained * 100:.1f}% of trajectory variance")

    # ---- 4. generate large ensembles for test years -----------------------------------
    Cte = np.stack([T_init[te], h_init[te]], axis=1)
    out = {}
    for name, em in (("gaussian", gauss), ("diffusion", diff)):
        s = em.sample(Cte, n_big, seed=5)
        out[name] = (s[..., :lead], s[..., lead:])

    # ---- 5. diagnostics ---------------------------------------------------------------
    sd = T_obs.std()
    ens = {
        "physics_20": (hT[te], hh[te]),
        f"physics_{n_big}": (bigT, bigh),
        f"gaussian_{n_big}": out["gaussian"],
        f"diffusion_{n_big}": out["diffusion"],
    }
    table = {}
    for w in (0, 1, 2):
        ob = M.winter_mean(obs_T[te], start_month, w)
        row = {}
        for name, (eT, _) in ens.items():
            e = M.winter_mean(eT, start_month, w)
            row[name] = {"acc": M.acc(e, ob), "rpc": M.rpc(e, ob), "snr": M.model_snr(e),
                         "spread_skill": M.spread_skill(e, ob)}
        table[f"winter_{w + 1}"] = row

    print("\nWinter (DJF) Nino-3.4 skill on test years")
    print(f"{'ensemble':<16}" + "".join(f"{'W' + str(w + 1) + ' ACC':>9}{'RPC':>7}{'SNR':>7}" for w in range(3)))
    for name in ens:
        line = f"{name:<16}"
        for w in range(3):
            r = table[f"winter_{w + 1}"][name]
            line += f"{r['acc']:9.2f}{r['rpc']:7.2f}{r['snr']:7.2f}"
        print(line)

    # ROM fingerprints (same estimator for every dataset)
    mon_lead = (start_month - 1 + np.arange(lead)) % 12 + 1
    fps = {"observations": summarize(fit_ro(T_obs, h_obs, months))}
    for name, (eT, eh) in ens.items():
        fps[name] = summarize(fit_ro(eT, eh, mon_lead))
    print("\nROM fingerprint")
    keys = ["R_mean", "R_range", "F1", "F2", "sigma_T", "sigma_h"]
    print(f"{'dataset':<16}" + "".join(f"{k:>10}" for k in keys))
    for name, f in fps.items():
        print(f"{name:<16}" + "".join(f"{f[k]:10.4f}" for k in keys))

    # multi-year El Nino persistence (Bai et al. thresholds)
    myen = {}
    for name, (eT, _) in ens.items():
        w1 = M.winter_mean(eT, start_month, 0)
        w2 = M.winter_mean(eT, start_month, 1)
        myen[name] = M.multiyear_elnino_fraction(w1, w2, sd)
    print("\nFraction of first-winter El Nino events that persist into winter 2:")
    for name, v in myen.items():
        print(f"  {name:<16} {v:.2f}")

    # state-dependent spread: El Nino starts should be noisier (multiplicative noise, B > 0)
    warm = T_init[te] > 0.5 * sd
    cold = T_init[te] < -0.5 * sd
    sdep = {}
    for name, (eT, _) in ens.items():
        w1 = M.winter_mean(eT, start_month, 0)
        spread = w1.std(axis=1)
        sdep[name] = float(spread[warm].mean() / spread[cold].mean()) if warm.any() and cold.any() else float("nan")
    print("\nWinter-1 spread, warm starts / cold starts (state-dependent noise):")
    for name, v in sdep.items():
        print(f"  {name:<16} {v:.2f}")

    # ---- 6. correct for the SNP (winter 2 example) ---------------------------------------
    ob_tr = M.winter_mean(obs_T[tr], start_month, 1)
    ob_te = M.winter_mean(obs_T[te], start_month, 1)
    e_tr = M.winter_mean(hT[tr], start_month, 1)
    e_te = M.winter_mean(out["diffusion"][0], start_month, 1)
    em_adj, scale = M.rpc_adjust(e_te, ob_tr, e_tr)
    raw = e_te.mean(axis=1)
    corr = {
        "scale_factor": scale,
        "raw_rmse": float(np.sqrt(np.mean((raw - ob_te) ** 2))),
        "adjusted_rmse": float(np.sqrt(np.mean((em_adj - ob_te) ** 2))),
        "raw_signal_std": float(raw.std()),
        "adjusted_signal_std": float(em_adj.std()),
        "obs_std": float(ob_te.std()),
    }
    print(f"\nSNP correction (winter 2): signal scaled by {scale:.2f}; "
          f"ensemble-mean std {corr['raw_signal_std']:.2f} -> {corr['adjusted_signal_std']:.2f} C "
          f"(obs {corr['obs_std']:.2f}); RMSE {corr['raw_rmse']:.2f} -> {corr['adjusted_rmse']:.2f}")

    # ---- figures ------------------------------------------------------------------------
    sizes = [5, 10, 20, 50, 100, 200, n_big]
    ob2 = M.winter_mean(obs_T[te], start_month, 1)
    conv = {k: M.convergence(M.winter_mean(v[0], start_month, 1), ob2, sizes, n_boot=50)
            for k, v in ens.items() if k != "physics_20"}
    conv20 = M.convergence(M.winter_mean(hT[te], start_month, 1), ob2, [5, 10, 20], n_boot=50)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    styles = {f"physics_{n_big}": ("k", "o", "Physics model"),
              f"gaussian_{n_big}": ("tab:blue", "s", "Gaussian emulator"),
              f"diffusion_{n_big}": ("tab:red", "^", "Diffusion emulator")}
    for k, c in conv.items():
        col, mk, lab = styles[k]
        ax[0].plot(c["size"], c["acc"], color=col, marker=mk, label=lab)
        ax[1].plot(c["size"], c["rpc"], color=col, marker=mk, label=lab)
    ax[0].plot(conv20["size"], conv20["acc"], "k--", lw=1, label="Physics, 20 members")
    for a, yl in zip(ax, ("ACC, winter-2 Nino-3.4", "RPC, winter-2 Nino-3.4")):
        a.set_xscale("log")
        a.set_xlabel("Ensemble size (members)")
        a.set_ylabel(yl)
        a.grid(alpha=0.3)
    ax[1].axhline(1, color="gray", lw=1)
    ax[0].legend(fontsize=8)
    fig.suptitle("Skill and RPC converge only with large ensembles (synthetic SRO test)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_convergence.png"), dpi=150)

    fig, axs = plt.subplots(1, 4, figsize=(12, 3.2))
    names = list(fps.keys())
    cols = ["0.5", "k", "0.3", "tab:blue", "tab:red"][:len(names)]
    for a, k in zip(axs, ["R_mean", "F1", "sigma_T", "sigma_h"]):
        a.bar(range(len(names)), [fps[n][k] for n in names], color=cols)
        a.set_xticks(range(len(names)))
        a.set_xticklabels([n.replace("_", " ") for n in names], fontsize=7, rotation=45, ha="right")
        a.set_title(k)
    fig.suptitle("Recharge-oscillator fingerprint: emulators should match the physics model")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2_rom_fingerprint.png"), dpi=150)

    i = int(np.argmax(np.abs(obs_T[te, 1])))  # an eventful test start
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    for a, (k, lab) in zip(ax, [(f"physics_{n_big}", "Physics model"),
                                (f"gaussian_{n_big}", "Gaussian emulator"),
                                (f"diffusion_{n_big}", "Diffusion emulator")]):
        eT = ens[k][0][i]
        q = np.percentile(eT, [5, 25, 50, 75, 95], axis=0)
        x = np.arange(lead)
        a.fill_between(x, q[0], q[4], color="0.85", label="5-95%")
        a.fill_between(x, q[1], q[3], color="0.6", label="25-75%")
        a.plot(x, q[2], "k", lw=1, label="median")
        a.plot(x, obs_T[te][i], "tab:orange", lw=2, label="observed")
        a.set_title(lab)
        a.set_xlabel("Lead (months from Nov 1)")
    ax[0].set_ylabel("Nino-3.4 (C)")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig3_plumes.png"), dpi=150)

    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump({"skill": table, "rom_fingerprint": fps, "multiyear_elnino_fraction": myen,
                   "spread_ratio_warm_over_cold": sdep,
                   "snp_correction_winter2": corr,
                   "params_obs": p_obs.__dict__, "params_model": p_mod.__dict__}, f, indent=2)
    print(f"\nDone in {time.time() - t0:.0f} s. Figures and results.json written to {outdir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller run for a fast check")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    main(quick=args.quick, outdir=args.outdir)
