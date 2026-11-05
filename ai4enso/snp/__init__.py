"""Signal-to-noise paradox: ensemble diagnostics, Markov (AR1) framework, and
recharge-oscillator predictability."""
from .metrics import (noise_variance, signal_variance, model_snr, predictable_fraction,
                      acc_infinite, perfect_model_acc, perfect_model_acc_infinite, rpc,
                      rpc_corrected, rsc, rsc_corrected, summary, convergence, calibrate_ensemble)
from . import markov, ro_predictability
