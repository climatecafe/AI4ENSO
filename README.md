# enso-snp-emulator

A prototype for building a generative AI emulator of multi-year ENSO hindcasts and using
it to diagnose and correct for the signal-to-noise paradox (SNP). This is the
"diagnose → explain → correct" workflow proposed for NOAA CPO FY27 Priority Area 1.

The emulator does **not** replace the full physics model. It learns one thing: given the
tropical Pacific state at initialization, what range of multi-year ENSO trajectories
(Niño-3.4 SST `T` and equatorial thermocline/SSH `h`) can follow. Once trained on a
20-member physics ensemble, it can generate 1000+ members in seconds.

Everything runs on a laptop CPU with numpy and scikit-learn. The demo uses a stochastic
recharge oscillator (SRO) as synthetic "observations" and "model", so every result can be
checked against a true 1000-member physics ensemble.

## Quick start

```bash
pip install -r requirements.txt
python demo_sro.py --quick   # about 15 s
python demo_sro.py           # about 1 min; writes figures and results.json to results/
```

## What is in the package

| File | Purpose | Proposal objective |
|---|---|---|
| `ensosnp/sro.py` | Stochastic recharge oscillator (Bai et al., 2026 form, with seasonal growth rate and state-dependent noise); long runs and initialized ensemble hindcasts | Synthetic test bed |
| `ensosnp/emulator.py` | `GaussianEmulator` (benchmark) and `DiffusionEmulator` (conditional DDPM) in a reduced PCA space, conditioned on the initial state | 1. Diagnose (generative ensembles) |
| `ensosnp/rofit.py` | Recharge-oscillator fingerprint: fits growth rate R(month), feedbacks F1, F2 and noise amplitudes σT, σh from monthly T and h | Process-fidelity test; 2. Explain |
| `ensosnp/metrics.py` | ACC, RPC, model SNR, spread–skill ratio, ensemble-size convergence, multi-year El Niño persistence, RPC-based SNP adjustment | 1. Diagnose; 3. Correct |
| `ensosnp/data_prep.py` | **Untested template** for turning SPEAR or CESM2 hindcast files into emulator inputs | Real-data step |
| `demo_sro.py` | End-to-end synthetic experiment and figures | All |

## Demo design

- **Observations:** a long SRO run (220 years).
- **Physics model:** an SRO with weaker persistence (more damping) and stronger noise than
  the observations, run as 20-member November-initialized hindcasts for 36 months. These
  biases create an SNP.
- **Emulators:** trained only on the 20-member hindcasts from 141 training start years.
- **Test:** 76 held-out start years. Emulators generate 1000 members, compared with a true
  1000-member physics ensemble.

## Results (full run, winter-mean Niño-3.4 on test years)

| Ensemble | Winter 2 ACC | Winter 2 RPC | Winter 3 ACC | Winter 3 RPC |
|---|---|---|---|---|
| Physics, 20 members | 0.68 | 1.13 | 0.42 | 1.08 |
| Physics, 1000 members (reference) | 0.72 | 1.26 | 0.52 | 1.53 |
| Gaussian emulator, 1000 members | 0.72 | 1.24 | 0.53 | 1.52 |
| Diffusion emulator, 1000 members | 0.71 | 1.22 | 0.51 | 1.43 |

What this shows:

1. **The SNP is real in the test bed.** With 1000 members, RPC reaches 1.26 in winter 2 and
   1.53 in winter 3. The model predicts the observations better than it predicts itself.
2. **Small ensembles hide skill and understate the SNP.** With 20 members, winter-3 skill is
   0.42 instead of 0.52, and the RPC is underestimated too (`results/fig1_convergence.png`).
3. **Emulators trained on 20 members recover the 1000-member answer.** Both emulators
   reproduce the large-ensemble skill and RPC from held-out initial states.
4. **The ROM fingerprint catches fidelity errors.** With only 16 principal components, both
   emulators matched skill and RPC but underestimated the noise amplitudes σT and σh by about
   one third, because PCA truncation smoothed month-to-month variability. With 48 components
   (99.8% of variance) the fingerprint matches the physics model
   (`results/fig2_rom_fingerprint.png`). Skill scores alone would not have revealed this.
5. **State-dependent noise remains a challenge.** In the physics model, winter-1 spread is
   1.7 times larger after warm starts than after cold starts. The Gaussian benchmark cannot
   represent this (ratio 1.00 by construction). The diffusion emulator learns some of it
   (1.13) but not all. This is the main target for improvement (larger network, PyTorch,
   more training data).
6. **Correction step.** An RPC-based rescaling estimated on training years raises the
   winter-2 ensemble-mean signal from 0.84 to 1.08 °C, closer to the amplitude implied by the
   observed predictable signal. In this toy case it does not change RMSE (1.02 → 1.02),
   as expected for a variance-matching (not MSE-minimizing) adjustment. Its value is in
   realistic signal amplitude and calibrated probabilities, which still need to be scored.

## Caveats

- The SRO is a two-variable toy. Its first-winter skill (ACC ≈ 0.97) is far higher than in
  reality, because initial states are known almost perfectly.
- The SRO parameter values follow Bai et al. (2026) in magnitude, but the seasonal growth
  rate R(m) is an assumed cosine, not a fit to observations.
- The diffusion denoiser is a scikit-learn MLP. For SPEAR or CESM2 fields (more variables,
  spatial patterns), use a PyTorch network with the same interface.
- `data_prep.py` has not been run on real files.

## Next steps with real data

1. Extract Niño-3.4 SST and equatorial SSH (or 20 °C isotherm depth) from SPEAR decadal
   hindcasts, CESM2 SMYLE and CESM2-DP, with lead-dependent drift correction.
2. Fit ROM fingerprints for observations (ORAS5, 20CR), SPEAR and CESM2 to locate the
   source of the SNP (persistence, growth rate, period or noise).
3. Train the emulator on SPEAR hindcasts, check its fingerprint, then generate 1000-member
   ensembles for every start date.
4. Add explainable-AI attribution to a deep-learning multi-year ENSO predictor, and test it on
   the CCSM4 control vs. interactive-ensemble pair.

## References

- Bai, K., Jin, Y., Zhang, W., Lin, X., & Kirtman, B. (2026). Atmospheric noise forcing increases multiyear El Niño events. *GRL*, 53(14), e2026GL123553.
- Eade, R., et al. (2014). Do seasonal-to-decadal climate predictions underestimate the predictability of the real world? *GRL*, 41, 5620–5628.
- Jin, Y. (2022). The signal-to-noise paradox in ENSO prediction: Role of ENSO growth rate and period. *GRL*, 49(12), e2022GL097965.
- Vialard, J., et al. (2025). The ENSO recharge oscillator conceptual model: Achievements and future prospects. *Reviews of Geophysics*, 63(1), e2024RG000843.
- Vijayeta, A., & Dommenget, D. (2018). An evaluation of ENSO dynamics in CMIP simulations in the framework of the recharge oscillator model. *Climate Dynamics*, 51, 1753–1771.
- Zhang, W., & Kirtman, B. (2019). Understanding the signal-to-noise paradox with a simple Markov model. *GRL*, 46(22), 13308–13317.
