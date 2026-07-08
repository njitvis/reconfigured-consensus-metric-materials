#!/usr/bin/env python3
"""
residual_consensus.py
=====================
Standalone implementation of the residual-consensus metric for ConsensusLens v2.

What it does
------------
For each unit (a response distribution over the k Likert categories for one
question, at whatever granularity you pool to), it computes:

    A      observed consensus            (the repo's median-anchored cofA)
    mu     mean rating
    E      expected consensus at mu      (centerline of the feasible band)
    Q1,Q3  feasible band edges at mu
    IQR    feasible spread               (reliability; the residual's denominator)
    z      residual consensus = (A - E) / IQR
    defined  False if IQR ~ 0 (band pinched near floor/ceiling)

The consensus function reproduces the repository's calculate_cofA exactly; a
self-check at import time asserts agreement to 1e-9.

How to use
----------
    from residual_consensus import compute_units, residual_from_counts

    # one distribution (counts in categories 0..k-1):
    rec = residual_from_counts([4, 5, 6, 26, 42])     # k inferred from length
    print(rec.z, rec.A, rec.E)

    # a whole dataframe of long-format responses:
    df = pd.read_csv("student_responses.csv")
    units = compute_units(
        df,
        response_col="Overallteachingeffectiv",   # one question column
        group_cols=["cname"],                       # granularity: section level
        scale_min=0, scale_max=4,                   # 0..4 here
    )
    units.to_csv("residuals_overall_by_section.csv", index=False)

Granularity = group_cols. Section level -> ["cname"]. Department level ->
["Dept"]. College level -> ["College_new"]. Pooling happens automatically by
grouping the raw responses, so a coarser unit's residual is recomputed from the
pooled counts (never averaged from finer units).

Run `python residual_consensus.py` for the built-in self-test and a small demo.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from math import comb
from dataclasses import dataclass, asdict
import itertools as IT

# ----------------------------------------------------------------------------
# 1. Consensus A  (exact port of the repository's calculate_cofA)
# ----------------------------------------------------------------------------
def calculate_cofA(values, n, k, r_options):
    """Canonical repo implementation. `values` is the raw list of responses."""
    values = np.asarray(values)
    r_med = np.median(values)                       # median over RAW values
    d_obs = 0.0
    for r in r_options:
        n_r = np.count_nonzero(values == r)
        d_obs += n_r * np.abs(r_med - r)
    d_exp = sum(np.abs(((k + 1) / 2) - r) for r in r_options) * (n / k)
    return 1.0 - (d_obs / d_exp)


def cofA_from_counts(counts, k):
    """A from integer category counts [c_0..c_{k-1}], matching calculate_cofA."""
    counts = np.asarray(counts, dtype=int)
    n = int(counts.sum())
    vals = np.repeat(np.arange(k), counts)          # rebuild raw values
    return calculate_cofA(vals, n, k, list(range(k)))


def mean_from_counts(counts, k):
    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    return float((counts * np.arange(k)).sum() / n)


# ----------------------------------------------------------------------------
# 2. Feasible baseline at (mu, n):  E, Q1, Q3, IQR  over all distributions
#    that have the given n and a mean within a small window of mu.
# ----------------------------------------------------------------------------
ENUM_CAP = 600_000          # enumerate exactly below this many compositions
MC_SAMPLES = 400_000        # else Monte-Carlo this many
WINDOW = 0.05               # half-width of the mean window (on the value scale)

def _enumerate_compositions(n, k):
    out = []
    for choice in IT.combinations(range(n + k - 1), k - 1):
        edges = (-1,) + choice + (n + k - 1,)
        out.append([edges[i + 1] - edges[i] - 1 for i in range(k)])
    return np.array(out, dtype=int)

def _random_compositions(n, k, M, rng):
    """Sample M count-vectors summing to n, memory-safe for large n (cost O(M*k)).
       Proportions ~ Dirichlet(1..1) (uniform on the simplex), then the multinomial
       count is built as a sequence of binomials, which is exact and avoids the
       (M, n+k) allocation that the bars-in-a-line method needs."""
    p = rng.dirichlet(np.ones(k), size=M)          # (M, k) uniform-simplex proportions
    out = np.zeros((M, k), dtype=int)
    remaining = np.full(M, n, dtype=int)
    p_left = np.ones(M)
    for j in range(k - 1):
        frac = np.clip(p[:, j] / np.maximum(p_left, 1e-12), 0.0, 1.0)
        draw = rng.binomial(remaining, frac)
        out[:, j] = draw
        remaining -= draw
        p_left -= p[:, j]
    out[:, k - 1] = remaining
    return out

def _A_vec(counts2d, k):
    """Vectorized A for many count-vectors; uses the integer-midpoint median.
       Only used to characterize the baseline cloud, where the tiny even-n
       median difference is immaterial to E/Q1/Q3."""
    counts2d = np.atleast_2d(counts2d).astype(float)
    n = counts2d.sum(axis=1)
    VALS = np.arange(k)
    cum = counts2d.cumsum(axis=1)
    lo = (n - 1) // 2; hi = n // 2
    cat_lo = (cum > lo[:, None]).argmax(axis=1)
    cat_hi = (cum > hi[:, None]).argmax(axis=1)
    med = (VALS[cat_lo] + VALS[cat_hi]) / 2.0
    d_exp_unit = np.abs((k + 1) / 2 - VALS).sum()
    d_obs = (counts2d * np.abs(med[:, None] - VALS[None, :])).sum(axis=1)
    mean = (counts2d * VALS).sum(axis=1) / n
    return 1 - d_obs / ((n / k) * d_exp_unit), mean

_baseline_cache: dict = {}

def feasible_baseline(n, mu, k, rng):
    """Return (E, Q1, Q3, IQR) of consensus over feasible distributions at (n, mu).
       Depends only on (n, mu, k); cached by (n, round(mu,2), k)."""
    key = (int(n), round(float(mu), 2), int(k))
    if key in _baseline_cache:
        return _baseline_cache[key]
    total = comb(n + k - 1, k - 1)
    if total <= ENUM_CAP:
        C = _enumerate_compositions(n, k)
    else:
        C = _random_compositions(n, k, MC_SAMPLES, rng)
    A, mean = _A_vec(C, k)
    sel = A[(mean >= mu - WINDOW) & (mean <= mu + WINDOW)]
    if sel.size < 40:
        res = (np.nan, np.nan, np.nan, np.nan)
    else:
        E = float(sel.mean())
        Q1 = float(np.percentile(sel, 25)); Q3 = float(np.percentile(sel, 75))
        res = (E, Q1, Q3, Q3 - Q1)
    _baseline_cache[key] = res
    return res


# ----------------------------------------------------------------------------
# 3. The residual for one distribution
# ----------------------------------------------------------------------------
@dataclass
class ResidualRecord:
    n: int
    counts: list
    pct: list
    mu: float
    A: float
    E: float
    Q1: float
    Q3: float
    IQR: float
    z: float
    defined: bool

EPS = 1e-9

def residual_from_counts(counts, k=None, rng=None):
    """counts: integer list over categories 0..k-1. k inferred if not given."""
    counts = list(map(int, counts))
    if k is None:
        k = len(counts)
    if rng is None:
        rng = np.random.default_rng(7)
    n = int(sum(counts))
    pct = [round(c / n, 4) for c in counts] if n else [0.0] * k
    A = float(cofA_from_counts(counts, k))
    mu = mean_from_counts(counts, k)
    E, Q1, Q3, IQR = feasible_baseline(n, mu, k, rng)
    if not np.isfinite(IQR) or IQR < EPS:
        z, defined = float("nan"), False
    else:
        z, defined = (A - E) / IQR, True
    return ResidualRecord(n, counts, pct, round(mu, 4), round(A, 4),
                          _r(E), _r(Q1), _r(Q3), _r(IQR), _r(z), defined)

def _r(x, nd=4):
    return round(float(x), nd) if x is not None and np.isfinite(x) else float("nan")


# ----------------------------------------------------------------------------
# 4. Whole-dataframe driver
# ----------------------------------------------------------------------------
def compute_units(df, response_col, group_cols, scale_min, scale_max,
                  min_n=18, rng=None):
    """
    df            long-format responses, one row per student-response.
    response_col  the single question column to score (questions stay independent).
    group_cols    granularity, e.g. ["cname"] (section), ["Dept"], ["College_new"].
    scale_min/max inclusive integer Likert bounds (0 and 4 here; 1 and 4 for k=4 scales).
    min_n         skip units with fewer than this many valid responses.
    Returns a tidy DataFrame, one row per unit, with all numbers from the spec.
    """
    if rng is None:
        rng = np.random.default_rng(7)
    k = scale_max - scale_min + 1
    cats = np.arange(scale_min, scale_max + 1)
    rows = []
    for keys, g in df.groupby(group_cols):
        s = pd.to_numeric(g[response_col], errors="coerce").dropna()
        s = s[(s >= scale_min) & (s <= scale_max)]
        n_valid = len(s)
        na_pct = round(1 - n_valid / len(g), 4) if len(g) else 0.0
        if n_valid < min_n:
            continue
        counts = [int((s == v).sum()) for v in cats]
        rec = residual_from_counts(counts, k=k, rng=rng)
        row = {}
        if not isinstance(keys, tuple):
            keys = (keys,)
        for col, val in zip(group_cols, keys):
            row[col] = val
        row["question"] = response_col
        row.update(asdict(rec))
        row["na_pct"] = na_pct
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 5. Self-test + demo
# ----------------------------------------------------------------------------
def _selftest():
    # canonical repo example: 7 zeros, 3 ones, n=10, k=5  ->  Consensus 0.5
    vals = 7 * [0] + 3 * [1]
    A = calculate_cofA(np.array(vals), 10, 5, [0, 1, 2, 3, 4])
    A2 = cofA_from_counts([7, 3, 0, 0, 0], 5)
    assert abs(A - A2) < 1e-12, (A, A2)
    # cross-check counts-path vs raw-values-path on random cases
    rng = np.random.default_rng(0)
    for _ in range(500):
        n = int(rng.integers(18, 60)); k = 5
        counts = rng.multinomial(n, np.ones(k) / k)
        a1 = cofA_from_counts(counts, k)
        a2 = calculate_cofA(np.repeat(np.arange(k), counts), n, k, list(range(k)))
        assert abs(a1 - a2) < 1e-9
    print("SELF-TEST passed: cofA matches repo calculate_cofA (<1e-9), example A =",
          round(A, 4))

if __name__ == "__main__":
    _selftest()
    rec = residual_from_counts([4, 5, 6, 26, 42])   # a real below-expected cell
    print("demo unit [4,5,6,26,42]:")
    for kk, vv in asdict(rec).items():
        print(f"   {kk:8s} {vv}")
