"""Linear inverse model (LIM; Penland & Sardeshmukh 1995).

Assumes the state x (e.g. leading PCs of tropical SST and heat content) evolves as
dx/dt = L x + white noise. From lag covariances:
    G(tau0) = C(tau0) C(0)^-1,   L = log(G(tau0)) / tau0,
    Q = -(L C(0) + C(0) L^T)     (fluctuation-dissipation relation).
Forecasts at lead tau are x(t + tau) = expm(L tau) x(t); the expected forecast error
covariance is C(0) - G(tau) C(0) G(tau)^T. LIM is a standard benchmark for ENSO prediction
and for optimal-growth (precursor) analysis.
"""
import numpy as np
from scipy.linalg import expm, logm


class LIM:
    def fit(self, X, tau0=1):
        """X : array (n_times, n_vars), anomalies, regularly spaced in time."""
        X = np.asarray(X, dtype=float)
        X = X - X.mean(axis=0)
        n = X.shape[0]
        C0 = X[:n - tau0].T @ X[:n - tau0] / (n - tau0)
        Ct = X[tau0:].T @ X[:n - tau0] / (n - tau0)
        G = Ct @ np.linalg.inv(C0)
        self.L_ = np.real(logm(G)) / tau0
        self.C0_ = C0
        self.Q_ = -(self.L_ @ C0 + C0 @ self.L_.T)
        self.tau0_ = tau0
        return self

    def propagator(self, tau):
        return expm(self.L_ * tau)

    def forecast(self, x0, leads):
        """x0 : (n_starts, n_vars). Returns (n_starts, len(leads), n_vars)."""
        x0 = np.atleast_2d(x0)
        return np.stack([x0 @ self.propagator(t).T for t in leads], axis=1)

    def error_covariance(self, tau):
        G = self.propagator(tau)
        return self.C0_ - G @ self.C0_ @ G.T

    def eigenmodes(self):
        """Eigenvalues of L: real part = decay rate, imaginary part = frequency."""
        lam, vec = np.linalg.eig(self.L_)
        order = np.argsort(-lam.real)
        return lam[order], vec[:, order]

    def optimal_growth(self, tau, norm=None):
        """Initial pattern that maximizes amplitude growth over tau (in the given norm).

        Returns (growth factor, initial pattern, evolved pattern).
        """
        G = self.propagator(tau)
        N = np.eye(G.shape[0]) if norm is None else norm
        w, v = np.linalg.eigh(G.T @ N @ G)
        k = np.argmax(w)
        p = v[:, k]
        return float(np.sqrt(w[k])), p, G @ p

    def noise_check(self):
        """Q should be positive semi-definite; returns its eigenvalues."""
        return np.linalg.eigvalsh(0.5 * (self.Q_ + self.Q_.T))
