"""
FOLLOWUP5 (#1a, #1b) — non-determinism diagnostic and fix.

Diagnostic
----------
Run the same 4-seed panel {42, 0, 1, 2} in two orders:
   A: [42, 0, 1, 2]     (as in FOLLOWUP4)
   B: [2, 1, 0, 42]     (reversed)
Also run each seed as a STANDALONE process (fresh interpreter) by
subprocess for baseline comparison.

If seed 42 gives:
  - A: X_forward, B: Y_reverse, standalone: Z — with X ≠ Y ≠ Z:
        carried state (Python-level or torch-level global)
  - A == B (position independent), but ≠ standalone:
        kernel / thread non-determinism

Fix candidates, cheapest first
------------------------------
  F0: no fix (baseline).
  F1: torch.use_deterministic_algorithms(True) +
      CUBLAS_WORKSPACE_CONFIG=:4096:8 in env +
      torch.backends.cudnn.deterministic = True +
      torch.backends.cudnn.benchmark    = False.
  F2: On top of F1, add explicit Python random.seed(seed) and force a
      fresh torch.Generator per fit (so we do not depend on any global
      state that may leak between calls).
  F3: On top of F2, drop the DriftField's `X_ref` / `V_ref` buffer
      caches per fit (module-level state suspicion).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/Users/terooatt/Downloads/scJDO")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reproducibility" / "code"))


def _run_one_fit(seed, fix_level=0):
    """Fit fresh, return argmax(R1) — mirroring FOLLOWUP4 scoring."""
    if fix_level >= 1:
        import torch
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            print(f"    (note: torch.use_deterministic_algorithms failed: {e})")
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    if fix_level >= 2:
        import random
        random.seed(seed)
        # Also reset numpy global
        np.random.seed(seed)

    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)

    from FOLLOWUP2_trained_probe import _build_adata
    from scjdo.tl import fit_drift

    adata, Z_lat, tau = _build_adata(seed=seed)
    t0 = time.time()
    model = fit_drift(
        adata, rep="X_pca", time_key="pseudotime",
        n_epochs=800, n_archetypes=4, n_eff_min=20.0, n_boot=10,
        grid_size=200, seed=seed, verbose=False,
    )
    dt = time.time() - t0
    r2 = float(adata.uns["scjdo"]["r2"])
    lam = np.asarray(adata.uns["scjdo"]["max_real_eig"])
    grid = np.asarray(adata.uns["scjdo"]["t_centers"])
    h_auto = adata.uns["scjdo"].get("bandwidth")
    m = (grid >= 0.05) & (grid <= 0.95) & ~np.isnan(lam)
    tau_argmax = float(grid[np.where(m)[0][np.argmax(lam[m])]]) if m.any() else float("nan")
    # A bit-identity signature: hash of the eigenvalue curve
    import hashlib
    sig = hashlib.md5(lam.tobytes()).hexdigest()[:12]
    return {"seed": seed, "argmax": tau_argmax, "r2": r2,
            "h_auto": h_auto, "fit_time": dt, "sig": sig}


def order_test(order, fix_level=0, label=""):
    print(f"\n== {label}  order={order}  fix_level={fix_level} ==")
    results = []
    for s in order:
        r = _run_one_fit(int(s), fix_level=fix_level)
        r["order_position"] = order.index(int(s))
        results.append(r)
        print(f"   seed={r['seed']:>3d}  pos={r['order_position']}  "
              f"fit={r['fit_time']:.1f}s  R²={r['r2']:.4f}  h_auto={r['h_auto']}  "
              f"argmax={r['argmax']:.4f}  sig={r['sig']}")
    return results


def standalone_test(seed, fix_level=0, n_repeats=3):
    """Run same seed in fresh subprocesses to isolate cross-fit contamination."""
    py = sys.executable
    print(f"\n== standalone (fresh interpreter) seed={seed}  fix_level={fix_level}, "
          f"{n_repeats} runs ==")
    results = []
    for run in range(n_repeats):
        cmd = [
            py, "-c",
            f"import sys; sys.path.insert(0, '{REPO / 'reproducibility' / 'code'}'); "
            f"sys.path.insert(0, '{REPO}'); "
            f"from FOLLOWUP5_determinism import _run_one_fit; "
            f"import json; "
            f"print('__RESULT__' + json.dumps(_run_one_fit({seed}, {fix_level})))"
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=200)
        out = proc.stdout.strip().splitlines()
        result_line = [l for l in out if l.startswith("__RESULT__")]
        if not result_line:
            print(f"   run {run}: no result. stderr[-200:]: {proc.stderr[-200:]}")
            continue
        r = json.loads(result_line[0][len("__RESULT__"):])
        results.append(r)
        print(f"   run {run}: seed={r['seed']}  argmax={r['argmax']:.4f}  sig={r['sig']}")
    return results


def main():
    out_dir = REPO / "reproducibility"
    all_out = {"fix_level_0": {}, "fix_level_1": {}, "fix_level_2": {}}

    print("=" * 74)
    print("(1a) DIAGNOSTIC — reversed-order multi-seed vs standalone")
    print("=" * 74)

    forward = order_test([42, 0, 1, 2], fix_level=0, label="forward")
    reversed_ = order_test([2, 1, 0, 42], fix_level=0, label="reversed")
    solo = standalone_test(42, fix_level=0, n_repeats=3)

    all_out["fix_level_0"]["forward"] = forward
    all_out["fix_level_0"]["reversed"] = reversed_
    all_out["fix_level_0"]["standalone_42"] = solo

    # Extract seed=42 values
    argmax_42_forward = next(r["argmax"] for r in forward if r["seed"] == 42)
    sig_42_forward = next(r["sig"] for r in forward if r["seed"] == 42)
    argmax_42_reversed = next(r["argmax"] for r in reversed_ if r["seed"] == 42)
    sig_42_reversed = next(r["sig"] for r in reversed_ if r["seed"] == 42)
    argmax_42_solo = solo[0]["argmax"] if solo else float("nan")
    sig_42_solo = solo[0]["sig"] if solo else "N/A"
    print("\n" + "=" * 74)
    print("(1a) DIAGNOSTIC SUMMARY (seed 42):")
    print(f"     forward multi-seed (pos {forward[0]['order_position']}):   "
          f"argmax = {argmax_42_forward:.4f}   sig = {sig_42_forward}")
    print(f"     reversed multi-seed (pos {next(r['order_position'] for r in reversed_ if r['seed']==42)}): "
          f"argmax = {argmax_42_reversed:.4f}   sig = {sig_42_reversed}")
    print(f"     standalone fresh process:                            "
          f"argmax = {argmax_42_solo:.4f}   sig = {sig_42_solo}")
    if abs(argmax_42_forward - argmax_42_reversed) < 1e-4:
        print("     → position-independent within a process, but ≠ standalone.")
        print("       Points to KERNEL/THREADING nondeterminism (fix candidate F1).")
    else:
        print("     → position-dependent within a process.")
        print("       Points to CARRIED GLOBAL STATE (fix candidate F2/F3).")

    print("\n" + "=" * 74)
    print("(1b) TRIAL FIX F1  (torch.use_deterministic_algorithms, cudnn flags)")
    print("=" * 74)
    forward_f1 = order_test([42, 0, 1, 2], fix_level=1, label="forward-F1")
    reversed_f1 = order_test([2, 1, 0, 42], fix_level=1, label="reversed-F1")
    solo_f1 = standalone_test(42, fix_level=1, n_repeats=3)
    all_out["fix_level_1"]["forward"] = forward_f1
    all_out["fix_level_1"]["reversed"] = reversed_f1
    all_out["fix_level_1"]["standalone_42"] = solo_f1

    print("\n" + "=" * 74)
    print("(1b) TRIAL FIX F2  (add python random + fresh generators)")
    print("=" * 74)
    forward_f2 = order_test([42, 0, 1, 2], fix_level=2, label="forward-F2")
    reversed_f2 = order_test([2, 1, 0, 42], fix_level=2, label="reversed-F2")
    solo_f2 = standalone_test(42, fix_level=2, n_repeats=3)
    all_out["fix_level_2"]["forward"] = forward_f2
    all_out["fix_level_2"]["reversed"] = reversed_f2
    all_out["fix_level_2"]["standalone_42"] = solo_f2

    (out_dir / "data" / "FOLLOWUP5_determinism.json").write_text(
        json.dumps(all_out, indent=2, default=lambda x: float(x))
    )
    print(f"\nSaved: {out_dir}/data/FOLLOWUP5_determinism.json")


if __name__ == "__main__":
    main()
