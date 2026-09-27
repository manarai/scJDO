"""
T3 — Sign and labelling audit for the toggle-switch bifurcation readouts.

Verifications:
  1. At the symmetric fixed point (s(α), s(α)), the eigenvector of
     J = [[-1, -c], [-c, -1]] associated with eigenvalue λ = -1 + c is
     the antisymmetric direction (1, -1)/sqrt(2), for a sweep of α values.
  2. The corresponding eigenvector for λ = -1 - c is the symmetric direction
     (1, 1)/sqrt(2).
  3. Along a small perturbation off the symmetric line — (s + ε, s - ε) —
     the drift's antisymmetric component grows past τ_crit and decays
     before, i.e. the SIGN of ẇ · w flips at τ_crit exactly as it should.
  4. The buggy fixed-point iteration in saddle_readouts._toggle_sym_fp
     diverges for α > α_crit — recorded here so any future code that
     reuses that helper past the pitchfork gets a warning.

These are cheap sanity checks. They do not require any trained model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import toggle_truth as gt


def test_antisymmetric_eigenvector():
    failures = []
    for a in [0.5, 1.0, 1.05, 1.2, 1.5, 2.0, 3.0, 4.0]:
        s = gt.sym_fp(a)
        J = gt.toggle_jacobian_at(s, s, a)
        c = 4.0 * a * s ** 3 / (1.0 + s ** 4) ** 2
        eigvals, eigvecs = np.linalg.eig(J)
        # Sort by real part descending so [0] is the leading eigenvalue.
        order = np.argsort(-eigvals.real)
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        # Analytic expectations
        lam_perp_expected = -1.0 + c
        lam_par_expected = -1.0 - c
        # Antisymmetric mode = (1,-1)/sqrt(2)
        v_perp = np.array([1.0, -1.0]) / np.sqrt(2.0)
        v_par = np.array([1.0, 1.0]) / np.sqrt(2.0)
        assert np.isclose(eigvals[0].real, lam_perp_expected, atol=1e-10), \
            f"leading eigenvalue mismatch at α={a}: got {eigvals[0]}, expected {lam_perp_expected}"
        assert np.isclose(eigvals[1].real, lam_par_expected, atol=1e-10), \
            f"trailing eigenvalue mismatch at α={a}"
        # Eigenvectors up to sign
        vv = eigvecs[:, 0].real
        vv /= np.linalg.norm(vv)
        # Force positive first component for a canonical comparison
        if vv[0] < 0:
            vv = -vv
        # Antisymmetric expectation, canonical positive first component
        if not (np.isclose(abs(vv @ v_perp), 1.0, atol=1e-8)):
            failures.append(
                f"α={a}: leading eigenvector is {vv}, expected ±(1,-1)/sqrt(2)={v_perp}"
            )
        vv2 = eigvecs[:, 1].real
        vv2 /= np.linalg.norm(vv2)
        if vv2[0] < 0:
            vv2 = -vv2
        if not (np.isclose(abs(vv2 @ v_par), 1.0, atol=1e-8)):
            failures.append(
                f"α={a}: trailing eigenvector is {vv2}, expected ±(1,1)/sqrt(2)={v_par}"
            )
    return failures


def test_transverse_flow_sign():
    """
    Along the perturbation (s + ε, s − ε), the transverse coordinate
    w = (x − y)/sqrt(2) and its drift are

        ẇ = (fx − fy) / sqrt(2)  ≈  (-1 + c) · w  = λ_⊥ · w.

    So ẇ · w has the same sign as λ_⊥. It is negative (restoring) for
    α < α_crit and positive (destabilizing) for α > α_crit.
    """
    failures = []
    eps = 1e-3
    for a in [0.5, 1.0, 1.05, 1.2, 2.0, 4.0]:
        s = gt.sym_fp(a)
        z = np.array([[s + eps, s - eps]])
        f = gt.toggle_drift(z, a)[0]
        w = (z[0, 0] - z[0, 1]) / np.sqrt(2.0)
        wdot = (f[0] - f[1]) / np.sqrt(2.0)
        c = 4.0 * a * s ** 3 / (1.0 + s ** 4) ** 2
        lam_perp = -1.0 + c
        sign_flow = np.sign(wdot * w)
        sign_pred = np.sign(lam_perp)
        if sign_flow != sign_pred:
            failures.append(f"α={a}: ẇ·w sign {sign_flow} != sign λ_⊥ ({sign_pred})")
    return failures


def test_saddle_readouts_fp_iteration_diverges():
    """
    Reproduce the divergence of ``analysis/supp_saddle_localization/
    saddle_readouts._toggle_sym_fp`` for α > α_crit and record the
    diverging value against the correct Newton-solved value.
    """
    from importlib import import_module
    import sys
    sys.path.insert(0, "/Users/terooatt/Downloads/scJDO")
    mod = import_module("analysis.supp_saddle_localization.saddle_readouts")
    buggy = mod._toggle_sym_fp

    rows = []
    for a in [0.5, 1.0, 1.013, 1.1, 1.5, 2.0, 4.0]:
        b = buggy(a)
        n = gt.sym_fp(a)
        rows.append(
            {"alpha": a, "newton": n, "buggy_iteration": b, "residual_newton": abs(n + n**5 - a),
             "residual_buggy": abs(b + b**5 - a)}
        )
    return rows


def main():
    out_dir = Path("/Users/terooatt/Downloads/scJDO/reproducibility")
    report = {}

    print("[T3.1] antisymmetric-eigenvector test")
    f1 = test_antisymmetric_eigenvector()
    report["antisymmetric_eigenvector_failures"] = f1
    print(f"  {'OK' if not f1 else 'FAIL: ' + '; '.join(f1)}")

    print("[T3.2] transverse flow sign test")
    f2 = test_transverse_flow_sign()
    report["transverse_flow_sign_failures"] = f2
    print(f"  {'OK' if not f2 else 'FAIL: ' + '; '.join(f2)}")

    print("[T3.3] saddle_readouts._toggle_sym_fp iteration behaviour")
    rows = test_saddle_readouts_fp_iteration_diverges()
    report["fp_iteration_table"] = rows
    for r in rows:
        print(
            f"  α={r['alpha']:.3f}  Newton={r['newton']:.4f} (res={r['residual_newton']:.2e})  "
            f"buggy={r['buggy_iteration']:.4f} (res={r['residual_buggy']:.2e})"
        )

    (out_dir / "data" / "T3_sign_audit.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved: {out_dir}/data/T3_sign_audit.json")


if __name__ == "__main__":
    main()
