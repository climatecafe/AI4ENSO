"""Reference and benchmark forecast methods: LIM, model analogs, persistence."""
from .lim import LIM
from .analogs import ModelAnalog, build_library
from .baselines import persistence, damped_persistence, lag_autocorrelation
