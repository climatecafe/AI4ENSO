"""Conditional generative emulators of initialized ENSO hindcasts.

Task: given the initial state of the tropical Pacific (condition c), generate plausible
multi-year trajectories x (e.g. monthly Nino-3.4 T and thermocline h over 36 months),
as many as wanted, so that a 20-member physics ensemble can be expanded to 1000+
members.

Design (works the same for SRO toy data, SPEAR or CESM2 hindcasts):
1. Trajectories are flattened, standardized and compressed with PCA (reduced space).
2. A conditional-mean network predicts the predictable part mu(c).
3. The residual (x - mu) is modelled either by
   - GaussianEmulator: a multivariate normal with fixed covariance (benchmark), or
   - DiffusionEmulator: a conditional denoising-diffusion model (DDPM) that can learn
     state-dependent, non-Gaussian spread (e.g. stronger noise during El Nino).

Only numpy and scikit-learn are required, so the prototype runs on a laptop. For SPEAR-
scale data, swap the scikit-learn MLPs for a PyTorch network; the interface stays the same.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor


class _Reducer:
    """Standardize, then PCA, then whiten the PCs."""

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-12
        Z = (X - self.mean_) / self.std_
        self.pca_ = PCA(n_components=self.n_components).fit(Z)
        P = self.pca_.transform(Z)
        self.pc_std_ = P.std(axis=0) + 1e-12
        return self

    def transform(self, X):
        Z = (X - self.mean_) / self.std_
        return self.pca_.transform(Z) / self.pc_std_

    def inverse(self, P):
        Z = self.pca_.inverse_transform(P * self.pc_std_)
        return Z * self.std_ + self.mean_

    @property
    def explained(self):
        return float(self.pca_.explained_variance_ratio_.sum())


class _Base:
    def __init__(self, n_components=16, mean_hidden=(64, 64), seed=0):
        self.n_components = n_components
        self.mean_hidden = mean_hidden
        self.seed = seed

    def _fit_common(self, X, C):
        """X: (n_samples, n_features) trajectories; C: (n_samples, n_cond) conditions."""
        self.red_ = _Reducer(self.n_components).fit(X)
        P = self.red_.transform(X)
        self.c_mean_ = C.mean(axis=0)
        self.c_std_ = C.std(axis=0) + 1e-12
        Cs = (C - self.c_mean_) / self.c_std_
        self.mean_net_ = MLPRegressor(hidden_layer_sizes=self.mean_hidden, alpha=1e-3,
                                      max_iter=2000, early_stopping=True,
                                      random_state=self.seed).fit(Cs, P)
        R = P - self.mean_net_.predict(Cs)
        return P, Cs, R

    def _cond(self, C):
        return (np.atleast_2d(C) - self.c_mean_) / self.c_std_


class GaussianEmulator(_Base):
    """Benchmark: conditional mean plus Gaussian residuals with fixed covariance."""

    def fit(self, X, C):
        _, _, R = self._fit_common(X, C)
        self.cov_ = np.cov(R, rowvar=False)
        return self

    def sample(self, C, n_members, seed=0):
        rng = np.random.default_rng(seed)
        Cs = self._cond(C)
        mu = self.mean_net_.predict(Cs)  # (n_starts, k)
        L = np.linalg.cholesky(self.cov_ + 1e-9 * np.eye(self.cov_.shape[0]))
        eps = rng.standard_normal((Cs.shape[0], n_members, mu.shape[1]))
        P = mu[:, None, :] + eps @ L.T
        out = self.red_.inverse(P.reshape(-1, P.shape[-1]))
        return out.reshape(Cs.shape[0], n_members, -1)


def _cosine_alpha_bar(K, s=0.008):
    t = np.arange(K + 1) / K
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    ab = f / f[0]
    return np.clip(ab, 1e-5, 1.0)


def _time_embed(t, K, n_freq=4):
    x = (t / K)[:, None]
    freqs = 2.0 ** np.arange(n_freq)[None, :] * np.pi
    return np.concatenate([np.sin(freqs * x), np.cos(freqs * x)], axis=1)


class DiffusionEmulator(_Base):
    """Conditional DDPM on residuals in the reduced (PCA) space.

    The denoiser sees (x_t, condition, diffusion time) and predicts the added noise.
    Because it is conditioned on the initial state, the generated spread can depend on
    the state, as in state-dependent (multiplicative) noise.
    """

    def __init__(self, n_components=16, mean_hidden=(64, 64), hidden=(256, 256),
                 K=50, repeats=10, max_iter=300, seed=0):
        super().__init__(n_components, mean_hidden, seed)
        self.hidden = hidden
        self.K = K
        self.repeats = repeats
        self.max_iter = max_iter

    def fit(self, X, C):
        _, Cs, R = self._fit_common(X, C)
        self.res_std_ = R.std(axis=0) + 1e-12
        R0 = R / self.res_std_
        rng = np.random.default_rng(self.seed)
        ab = _cosine_alpha_bar(self.K)
        self.ab_ = ab
        n, k = R0.shape
        idx = np.repeat(np.arange(n), self.repeats)
        t = rng.integers(1, self.K + 1, size=idx.size)
        eps = rng.standard_normal((idx.size, k))
        xt = np.sqrt(ab[t])[:, None] * R0[idx] + np.sqrt(1 - ab[t])[:, None] * eps
        feats = np.concatenate([xt, Cs[idx], _time_embed(t, self.K)], axis=1)
        self.denoiser_ = MLPRegressor(hidden_layer_sizes=self.hidden, alpha=1e-4,
                                      learning_rate_init=1e-3, batch_size=256,
                                      max_iter=self.max_iter, early_stopping=True,
                                      n_iter_no_change=15, random_state=self.seed)
        self.denoiser_.fit(feats, eps)
        return self

    def sample(self, C, n_members, seed=0):
        rng = np.random.default_rng(seed)
        Cs = self._cond(C)
        n_starts = Cs.shape[0]
        mu = self.mean_net_.predict(Cs)
        k = mu.shape[1]
        ab = self.ab_
        Cr = np.repeat(Cs, n_members, axis=0)
        x = rng.standard_normal((n_starts * n_members, k))
        for t in range(self.K, 0, -1):
            tt = np.full(x.shape[0], t)
            eps_hat = self.denoiser_.predict(np.concatenate([x, Cr, _time_embed(tt, self.K)], axis=1))
            x0_hat = (x - np.sqrt(1 - ab[t]) * eps_hat) / np.sqrt(ab[t])
            x0_hat = np.clip(x0_hat, -6, 6)
            if t > 1:
                beta = 1 - ab[t] / ab[t - 1]
                c0 = np.sqrt(ab[t - 1]) * beta / (1 - ab[t])
                ct = np.sqrt(1 - beta) * (1 - ab[t - 1]) / (1 - ab[t])
                var = beta * (1 - ab[t - 1]) / (1 - ab[t])
                x = c0 * x0_hat + ct * x + np.sqrt(var) * rng.standard_normal(x.shape)
            else:
                x = x0_hat
        P = mu[:, None, :] + (x * self.res_std_).reshape(n_starts, n_members, k)
        out = self.red_.inverse(P.reshape(-1, k))
        return out.reshape(n_starts, n_members, -1)
