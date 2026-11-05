# Generative emulators of initialized ENSO hindcasts

The emulator does **not** replace the full physics model. It learns one thing: given the
tropical Pacific state at initialization, what range of multi-year ENSO trajectories
(Niño-3.4 SST `T` and equatorial thermocline/SSH `h`) can follow. Once trained on a
20-member physics ensemble, it can generate 1000+ members in seconds.

Everything runs on a laptop CPU with numpy and scikit-learn. The demo uses a stochastic
recharge oscillator (SRO) as synthetic "observations" and "model", so every result can be
checked against a true 1000-member physics ensemble.

## Run

```bash
python examples/05_emulator_snp.py --quick   # about 20 s
python examples/05_emulator_snp.py           # about 2 min; figures in examples/figures_emulator/
```

## Components

| Module | Role |
|---|---|
| `ai4enso.dynamics.stochastic_ro` | Synthetic "observations" and biased "physics model" hindcasts |
| `ai4enso.emulator` | `GaussianEmulator` (benchmark) and `DiffusionEmulator` (conditional DDPM in a reduced PCA space, conditioned on the initial state) |
| `ai4enso.dynamics.rofit` | Recharge-oscillator fingerprint used as a process-fidelity test |
| `ai4enso.snp`, `ai4enso.verification` | Skill, perfect-model skill, RPC, calibration, CRPS |

## Demo design

- **Observations:** a long SRO run (220 years).
- **Physics model:** an SRO with weaker persistence (more damping) and stronger noise than
  the observations, run as 20-member November-initialized hindcasts for 36 months. These
  biases create an SNP.
- **Emulators:** trained only on the 20-member hindcasts from 141 training start years.
- **Test:** 76 held-out start years. Emulators generate 1000 members, compared with a true
  1000-member physics ensemble.

## Results (full run, winter-mean Niño-3.4 on test years)

ACC = ensemble mean vs observations; perfect-model ACC = the model predicting itself (one
member as truth, the rest as forecast); RPC = bias-corrected ratio of predictable components.
An SNP means ACC > perfect-model ACC and RPC > 1.

| Ensemble | W2 ACC | W2 perfect-model ACC | W2 RPC | W3 ACC | W3 perfect-model ACC | W3 RPC |
|---|---|---|---|---|---|---|
| Physics, 20 members | 0.68 | 0.55 | 1.25 | 0.42 | 0.31 | 1.53 |
| Physics, 1000 members (reference) | 0.72 | 0.55 | 1.26 | 0.52 | 0.33 | 1.54 |
| Gaussian emulator, 1000 members | 0.72 | 0.61 | 1.24 | 0.53 | 0.40 | 1.54 |
| Diffusion emulator, 1000 members | 0.71 | 0.61 | 1.22 | 0.51 | 0.33 | 1.44 |

What this shows:

1. **The SNP is real in the test bed.** The model predicts the observations (ACC 0.72 in
   winter 2) better than it predicts itself (0.55). RPC is 1.26 in winter 2 and 1.54 in
   winter 3.
2. **Large ensembles reduce the damage the SNP does to skill, but not the SNP itself.**
   Because each member carries a weak signal, many members are needed to average out the
   noise: winter-3 skill rises from 0.42 with 20 members to 0.52 with 1000. Perfect-model skill
   rises too, so the gap between the two curves stays (`examples/figures_emulator/fig1_convergence.png`, left).
   The SNP comes from the model's physics (here, too much damping and noise); more members
   estimate it more precisely but do not remove it. That is why the correction step is needed.
3. **Naive RPC is biased low for small ensembles.** Using the ensemble-mean variance as the
   signal variance, and the finite-ensemble ACC, both understate the SNP when there are few
   members: the naive winter-2 RPC rises from 0.91 at 5 members to 1.26 at 1000. Correcting
   both terms for residual noise (`ai4enso.snp.rpc_corrected`) gives a nearly constant RPC,
   between 1.25 and 1.30 at every ensemble size (`examples/figures_emulator/fig1_convergence.png`, right).
4. **Emulators trained on 20 members recover the 1000-member answer.** Both emulators
   reproduce the large-ensemble skill and RPC from held-out initial states. The Gaussian
   emulator's members are slightly too predictable (perfect-model ACC 0.61 vs 0.55 in winter 2
   and 0.40 vs 0.33 in winter 3); the diffusion emulator matches winter 3 better.
5. **The ROM fingerprint catches fidelity errors.** With only 16 principal components, both
   emulators matched skill and RPC but underestimated the noise amplitudes σT and σh by about
   one third, because PCA truncation smoothed month-to-month variability. With 48 components
   (99.8% of variance) the fingerprint matches the physics model
   (`examples/figures_emulator/fig2_rom_fingerprint.png`). Skill scores alone would not have revealed this.
6. **State-dependent noise remains a challenge.** In the physics model, winter-1 spread is
   1.7 times larger after warm starts than after cold starts. The Gaussian benchmark cannot
   represent this (ratio 1.00 by construction). The diffusion emulator learns some of it
   (1.13) but not all. This is the main target for improvement (larger network, PyTorch,
   more training data).
7. **Correction step.** `calibrate_ensemble` follows Eade et al. (2014): it rescales the
   ensemble-mean signal so its variance matches the observed predictable variance, and
   rescales member deviations so the spread matches the unpredictable part. For the ensemble
   mean, this factor equals the regression slope of observations on the ensemble mean, so it
   is the same as the ratio-of-predictable-signals (RPS) scaling used by Smith et al. (2025).
   Factors are estimated on training years and applied to test years (winter 2, diffusion
   emulator, 1000 members):

   | | Signal std (°C) | RMSE | Spread/skill | CRPS |
   |---|---|---|---|---|
   | Raw | 0.84 | 1.02 | 1.15 | 0.586 |
   | Calibrated | 1.22 | 1.04 | 1.04 | 0.596 |

   Calibration makes the ensemble more reliable (spread/skill closer to 1) and restores a
   realistic signal amplitude, but CRPS and RMSE do not improve on the test years. The signal
   factor estimated from 141 training years (×1.46) is larger than the test years would
   support (about ×1.2). Like the RPS scaling in Smith et al. (2025), this purely statistical
   correction is sensitive to sampling. Physically based corrections, informed by the ROM
   fingerprint, are the next step.

## Caveats

- The SRO is a two-variable toy. Its first-winter skill (ACC ≈ 0.97) is far higher than in
  reality, because initial states are known almost perfectly.
- The SRO parameter values follow Bai et al. (2026) in magnitude, but the seasonal growth
  rate R(m) is an assumed cosine, not a fit to observations.
- The diffusion denoiser is a scikit-learn MLP. For SPEAR or CESM2 fields (more variables,
  spatial patterns), use a PyTorch network with the same interface.
- `ai4enso/io.py` is a template and has not been run on real files.

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

See the [main README](../README.md#references).
