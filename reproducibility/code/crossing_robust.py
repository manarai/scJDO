"""
Robust crossing detection.

The naive "first τ where curve > 0" detector triggers on one noise-driven
positive point from the early-time SDE cloud, which produces an
apparent-crossing at grid[0] that is *not* a real transition. Three
fixes bundled here:

  1. Sustained sign change — the curve must remain > threshold for at
     least ``min_run`` consecutive grid points after the first crossing.
  2. Bootstrap band — resample cells with replacement, recompute the
     kernel-aggregated curve, redetect the crossing; report the [2.5, 97.5]
     percentile band.
  3. Explicit no-crossing return — if the curve never satisfies the run
     condition, return NaN and a `code = "no_crossing"` diagnostic
     instead of a spurious grid[0].
"""

from __future__ import annotations

import numpy as np


def robust_crossing(curve, grid, threshold=0.0, min_run=5, min_run_neg=None):
    """
    Genuine sign-change crossing detection.

    A crossing at index i requires:
      (a) ``min_run`` consecutive valid points  >  threshold  starting at i, AND
      (b) at least ``min_run_neg`` consecutive valid points  <  threshold
          somewhere in ``grid[:i]``  (i.e. the curve was genuinely below
          threshold BEFORE the crossing — this is what distinguishes a
          "crossing" from a curve that was always positive).

    Returns ``(tau, code)`` with code in
      {"crossing",           genuine sign change satisfying both (a) and (b)
       "always_positive",    curve is > threshold everywhere; NO crossing
       "always_negative",    curve is < threshold everywhere; NO crossing
       "no_crossing",        neither always positive nor always negative,
                             but no i satisfies both (a) and (b)
       "insufficient_data"}.

    ``min_run_neg`` defaults to ``min_run``.
    """
    if min_run_neg is None:
        min_run_neg = min_run
    valid = ~np.isnan(curve)
    if valid.sum() < min_run + min_run_neg:
        return float("nan"), "insufficient_data"

    valid_curve = curve[valid]
    if np.all(valid_curve > threshold):
        return float("nan"), "always_positive"
    if np.all(valid_curve < threshold):
        return float("nan"), "always_negative"

    N = len(curve)
    # Precompute a running-count of consecutive-negative points from the left.
    # Then a crossing at i requires:
    #   (a) curve[i:i+min_run] all valid AND > threshold
    #   (b) somewhere in curve[:i], at least min_run_neg consecutive valid
    #       points < threshold.
    below = valid & (curve < threshold)
    # Run lengths of below-threshold up to each index
    cum_neg = np.zeros(N, dtype=int)
    running = 0
    for k in range(N):
        if below[k]:
            running += 1
        else:
            running = 0
        cum_neg[k] = running
    ever_negative_run_up_to = np.zeros(N, dtype=int)
    m = 0
    for k in range(N):
        if cum_neg[k] > m:
            m = cum_neg[k]
        ever_negative_run_up_to[k] = m

    for i in range(N - min_run + 1):
        # (a)
        if not np.all(valid[i:i + min_run]):
            continue
        if not np.all(curve[i:i + min_run] > threshold):
            continue
        # (b) — the strict "before" check.  If i == 0, no chance.
        if i == 0:
            continue
        if ever_negative_run_up_to[i - 1] < min_run_neg:
            continue
        return float(grid[i]), "crossing"
    return float("nan"), "no_crossing"


def bootstrap_crossing(scalar_per_cell, t_per_cell, grid, curve_fn,
                       n_boot=200, seed=0, threshold=0.0, min_run=5,
                       agg_kwargs=None):
    """
    Bootstrap the crossing τ̂₀ from a per-cell scalar and a curve-building
    function. ``curve_fn(scalar_per_cell, t_per_cell, grid, **agg_kwargs)``
    is called on resamples.

    Returns dict with:
      point   : point estimate on the full sample
      mean    : mean of bootstrap crossings (NaNs = no-crossing)
      lo, hi  : 2.5 / 97.5 percentile band
      p_no_crossing : fraction of bootstrap resamples that returned no crossing
      n_boot  : number of bootstrap replicates
    """
    if agg_kwargs is None:
        agg_kwargs = {}
    rng = np.random.default_rng(seed)
    N = scalar_per_cell.shape[0]

    curve_full = curve_fn(scalar_per_cell, t_per_cell, grid, **agg_kwargs)
    tau_point, code_point = robust_crossing(curve_full, grid,
                                              threshold=threshold,
                                              min_run=min_run)

    taus = np.full(n_boot, np.nan)
    codes = []
    for b in range(n_boot):
        idx = rng.integers(0, N, N)
        curve_b = curve_fn(scalar_per_cell[idx], t_per_cell[idx], grid,
                           **agg_kwargs)
        t_b, c_b = robust_crossing(curve_b, grid, threshold=threshold,
                                    min_run=min_run)
        taus[b] = t_b
        codes.append(c_b)

    finite = np.isfinite(taus)
    p_no_cross = float(1.0 - finite.mean())
    if finite.sum() < 3:
        return {"point": tau_point, "point_code": code_point,
                "mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p_no_crossing": p_no_cross, "n_boot": n_boot}
    return {"point": tau_point, "point_code": code_point,
            "mean": float(np.nanmean(taus[finite])),
            "lo": float(np.percentile(taus[finite], 2.5)),
            "hi": float(np.percentile(taus[finite], 97.5)),
            "p_no_crossing": p_no_cross,
            "n_boot": n_boot}
