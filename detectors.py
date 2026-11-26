"""Generic fault-detection building blocks, independent of any one dataset.

- PCAMonitor: standardise + PCA fitted on reference data, with SPE (Q) and Hotelling T^2 and percentile
  thresholds.
- persistence: alarm only after N consecutive exceedances.
- cusum, page_hinkley: sequential change detectors on a (standardised) series.
- classify_change: label an offline change point as a step or a ramp.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCAMonitor:
    """PCA process monitor.

    Fit on reference data: standardise each variable, keep enough components to explain `var_explained`
    (or exactly `n_components`). SPE (Q) is the squared reconstruction error outside the retained
    components; T^2 is the Mahalanobis distance inside them. Thresholds are the `percentile` of each
    statistic on the reference data.
    """

    def __init__(self, n_components=None, var_explained=0.9, percentile=99.0):
        self.n_components, self.var_explained, self.percentile = n_components, var_explained, percentile

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.scaler_ = StandardScaler().fit(X)
        Z = self.scaler_.transform(X)
        self.full_pca_ = PCA().fit(Z)
        k = self.n_components or int(np.searchsorted(np.cumsum(self.full_pca_.explained_variance_ratio_),
                                                     self.var_explained) + 1)
        self.k_ = min(k, X.shape[1])
        self.components_ = self.full_pca_.components_[: self.k_]
        self.eigvals_ = self.full_pca_.explained_variance_[: self.k_]
        spe, t2 = self.statistics(X)
        self.spe_limit_ = np.percentile(spe, self.percentile)
        self.t2_limit_ = np.percentile(t2, self.percentile)
        return self

    def scores(self, X):
        return self.scaler_.transform(np.asarray(X, dtype=float)) @ self.components_.T

    def statistics(self, X):
        """(SPE, T^2) per row."""
        Z = self.scaler_.transform(np.asarray(X, dtype=float))
        S = Z @ self.components_.T
        resid = Z - S @ self.components_
        return (resid ** 2).sum(1), (S ** 2 / self.eigvals_).sum(1)

    def alarms(self, X):
        """(SPE > limit, T^2 > limit) per row."""
        spe, t2 = self.statistics(X)
        return spe > self.spe_limit_, t2 > self.t2_limit_


def persistence(exceed, n, groups=None):
    """True from the n-th consecutive True onwards, within each group (e.g. turbine). Order = input order."""
    exceed = np.asarray(exceed, dtype=bool)
    if n <= 1:
        return exceed.copy()
    groups = np.zeros(len(exceed), dtype=int) if groups is None else np.asarray(groups)
    # Run length of consecutive Trues, restarting at False and at group boundaries.
    new_run = np.r_[True, (groups[1:] != groups[:-1])] | ~exceed
    run_id = np.cumsum(new_run)
    run_len = pd.Series(exceed.astype(int)).groupby(run_id).cumsum().to_numpy()
    return exceed & (run_len >= n)


def alarm_onsets(alarm, groups=None):
    """True where an alarm episode starts (rising edge), within each group."""
    alarm = np.asarray(alarm, dtype=bool)
    groups = np.zeros(len(alarm), dtype=int) if groups is None else np.asarray(groups)
    prev = np.r_[False, alarm[:-1]] & np.r_[False, groups[1:] == groups[:-1]]
    return alarm & ~prev


def cusum(x, k=0.5, h=5.0, reset=True):
    """Two-sided tabular CUSUM on a standardised series (mean 0, sd 1 when in control).

    Returns (alarm, s_pos, s_neg). NaNs are skipped (statistics carried over). After an alarm both sums
    reset to 0 when `reset`.
    """
    x = np.asarray(x, dtype=float)
    sp = sn = 0.0
    s_pos, s_neg, alarm = np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x), dtype=bool)
    for i, v in enumerate(x):
        if not np.isnan(v):
            sp = max(0.0, sp + v - k)
            sn = max(0.0, sn - v - k)
            if sp > h or sn > h:
                alarm[i] = True
                if reset:
                    sp = sn = 0.0
        s_pos[i], s_neg[i] = sp, sn
    return alarm, s_pos, s_neg


def page_hinkley(x, delta=0.5, lam=5.0, reset=True):
    """Two-sided Page-Hinkley test against the running mean since the last reset. Returns alarm array."""
    x = np.asarray(x, dtype=float)
    alarm = np.zeros(len(x), dtype=bool)
    n, mean, m_up, m_dn, min_up, max_dn = 0, 0.0, 0.0, 0.0, 0.0, 0.0
    for i, v in enumerate(x):
        if np.isnan(v):
            continue
        n += 1
        mean += (v - mean) / n
        m_up += v - mean - delta
        m_dn += v - mean + delta
        min_up, max_dn = min(min_up, m_up), max(max_dn, m_dn)
        if m_up - min_up > lam or max_dn - m_dn > lam:
            alarm[i] = True
            if reset:
                n, mean, m_up, m_dn, min_up, max_dn = 0, 0.0, 0.0, 0.0, 0.0, 0.0
    return alarm


def classify_change(y, cp, window):
    """Step or ramp at index `cp` of series y, judged on y[cp-window : cp+window] (NaNs dropped).

    step: different constant mean before and after cp.
    ramp: constant before cp, then a linear trend after it (continuous at cp).
    The model with the lower BIC wins (both have 3 parameters, so this is lower residual sum of squares).
    Returns (label, size): size is the mean shift (step) or the change over the window (ramp).
    """
    y = np.asarray(y, dtype=float)
    lo, hi = max(0, cp - window), min(len(y), cp + window)
    idx = np.arange(lo, hi)
    seg = y[lo:hi]
    ok = ~np.isnan(seg)
    idx, seg = idx[ok], seg[ok]
    before, after = idx < cp, idx >= cp
    if before.sum() < 3 or after.sum() < 3:
        return "unknown", np.nan
    # Step
    step_fit = np.where(before, seg[before].mean(), seg[after].mean())
    rss_step = ((seg - step_fit) ** 2).sum()
    # Ramp: y = a + b * max(0, t - cp)
    A = np.c_[np.ones(len(idx)), np.maximum(0, idx - cp)]
    coef, *_ = np.linalg.lstsq(A, seg, rcond=None)
    rss_ramp = ((seg - A @ coef) ** 2).sum()
    if rss_step <= rss_ramp:
        return "step", seg[after].mean() - seg[before].mean()
    return "ramp", coef[1] * (idx.max() - cp)
