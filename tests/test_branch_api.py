"""
Unit tests for the new per-branch / per-perturbation convenience APIs:

    sjd.tl.infer_regulators_branches
    sjd.tl.fit_bridge_branches
    sjd.pl.branch_regulator_panels

These are the encapsulations of the per-branch idiom that was hand-written
(and bug-prone) in Figure3_FA.ipynb and Figure5.ipynb. Each test exercises
the specific bug class that motivated the API.

The drift-branches tests use the saved marrow AnnData at
    examples/results/palantir_driftfield/marrow_scjdo.h5ad
which carries pre-fit ``scqdiff_DC/Ery/Mono`` branch results. If that file
is not present, those tests are skipped (so the suite stays portable).
"""
from __future__ import annotations

import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest

import scjdo as sjd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_REPO     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MARROW   = os.path.join(
    _REPO, "examples", "results", "palantir_driftfield", "marrow_scjdo.h5ad"
)


@pytest.fixture(scope="module")
def marrow_adata():
    """Pre-fit marrow AnnData with scqdiff_DC/Ery/Mono branch uns entries."""
    if not os.path.exists(_MARROW):
        pytest.skip(f"marrow_scjdo.h5ad not found at {_MARROW}; "
                    "skipping branch-API tests")
    import anndata as ad
    return ad.read_h5ad(_MARROW)


# ---------------------------------------------------------------------------
# infer_regulators_branches
# ---------------------------------------------------------------------------

def test_infer_regulators_branches_copies_uns_back(marrow_adata):
    """The canonical bug: per-branch infer_regulators writes to a *subset*
    AnnData's uns, which is then discarded. The wrapper must copy that uns
    entry back onto the full AnnData so plotters can find it."""
    branch_models = {"DC": None, "Ery": None, "Mono": None}  # only keys matter
    out = sjd.tl.infer_regulators_branches(
        marrow_adata, branch_models,
        key_prefix="scqdiff",                  # matches the saved keys
        organism="human", network_source="builtin",
        min_targets=1, n_top=10, verbose=False,
    )
    assert set(out.keys()) == {"DC", "Ery", "Mono"}, \
        "Should return a dict keyed by every branch model"
    for name in ("DC", "Ery", "Mono"):
        # Returned DataFrame is non-empty
        assert isinstance(out[name], pd.DataFrame)
        assert len(out[name]) > 0, f"{name}: empty regulator table"
        # uns was copied back to the FULL AnnData (this is the bug-fix)
        reg_key = f"scjdo_regulators_{name}"
        assert reg_key in marrow_adata.uns, \
            f"{reg_key} not in adata.uns -- copy-back failed"
        assert "tables" in marrow_adata.uns[reg_key]


def test_infer_regulators_branches_skips_missing_source(marrow_adata):
    """Branches whose source key is missing should warn and skip, not raise."""
    branch_models = {"DC": None, "NotARealBranch": None}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = sjd.tl.infer_regulators_branches(
            marrow_adata, branch_models, key_prefix="scqdiff",
            organism="human", network_source="builtin",
            min_targets=1, n_top=5, verbose=False,
        )
    assert "DC" in out
    assert "NotARealBranch" not in out
    assert any("not in adata.uns" in str(wi.message) for wi in w)


def test_infer_regulators_branches_csv_writeout(marrow_adata, tmp_path):
    """save_csv_dir should produce one CSV per branch / direction."""
    branch_models = {"DC": None, "Ery": None}
    sjd.tl.infer_regulators_branches(
        marrow_adata, branch_models, key_prefix="scqdiff",
        organism="human", network_source="builtin",
        min_targets=1, n_top=5, verbose=False,
        save_csv_dir=str(tmp_path),
    )
    for name in ("DC", "Ery"):
        sub = tmp_path / name
        assert sub.is_dir(), f"missing branch dir for {name}"
        csvs = list(sub.glob("regulators_*.csv"))
        assert len(csvs) >= 1, f"no regulators CSV in {sub}"
        df = pd.read_csv(csvs[0])
        assert "regulator" in df.columns and len(df) > 0


# ---------------------------------------------------------------------------
# branch_regulator_panels
# ---------------------------------------------------------------------------

def test_branch_regulator_panels_writes_pdfs(marrow_adata, tmp_path):
    """The plot harness must emit one file per requested panel per branch,
    and tolerate per-panel failures without aborting the run."""
    import matplotlib
    matplotlib.use("Agg")

    # First populate regulator results via the new wrapper.
    branch_models = {"DC": None, "Ery": None, "Mono": None}
    sjd.tl.infer_regulators_branches(
        marrow_adata, branch_models, key_prefix="scqdiff",
        organism="human", network_source="builtin",
        min_targets=1, n_top=10, verbose=False,
    )

    # Use just the two safest panels to keep the test fast and robust.
    summary = sjd.pl.branch_regulator_panels(
        marrow_adata, branch_models, str(tmp_path),
        panels=("barplot", "scatter"),
        key_prefix="scqdiff", file_ext="pdf", verbose=False,
    )
    for name in ("DC", "Ery", "Mono"):
        d = tmp_path / name
        assert d.is_dir(), f"missing branch dir for {name}"
        files = {p.name for p in d.glob("reg_*.pdf")}
        # At least one panel succeeded per branch
        assert len(files) >= 1, f"no panels written for {name} (summary: {summary})"


def test_branch_regulator_panels_skips_when_no_table(marrow_adata, tmp_path):
    """Branches without a regulators key should be skipped cleanly, not raise."""
    import matplotlib
    matplotlib.use("Agg")
    # Branch name with no corresponding scjdo_regulators_* uns entry
    summary = sjd.pl.branch_regulator_panels(
        marrow_adata, {"GhostBranch": None}, str(tmp_path),
        panels=("barplot",), verbose=False,
    )
    assert summary == {"GhostBranch": []}, \
        "Missing regulator key should produce an empty panel list, not error"


# ---------------------------------------------------------------------------
# fit_bridge_branches
# ---------------------------------------------------------------------------

def _synthetic_perturb_adata(n_src=200, n_tgt_each=80, n_genes=50, seed=0):
    """Tiny synthetic perturb-seq-like AnnData: one source + two targets.
    Source ~ N(0, I); targets shifted along distinct axes. Loadings set so
    the bridge has a valid X_fa rep and varm['PCs']."""
    import anndata as ad
    rng = np.random.default_rng(seed)
    n_latent = 8
    src = rng.normal(0, 1, (n_src, n_latent))
    tA  = rng.normal(0, 1, (n_tgt_each, n_latent)); tA[:, 0] += 4.0
    tB  = rng.normal(0, 1, (n_tgt_each, n_latent)); tB[:, 1] -= 4.0
    X_lat = np.vstack([src, tA, tB]).astype(np.float32)

    # gene loadings — random orthonormal-ish
    W = rng.standard_normal((n_genes, n_latent)).astype(np.float32)
    X_gene = (X_lat @ W.T).astype(np.float32)

    labels = (["CTRL"] * n_src + ["TGT_A"] * n_tgt_each + ["TGT_B"] * n_tgt_each)
    A = ad.AnnData(X=X_gene)
    A.var_names = [f"g{i}" for i in range(n_genes)]
    A.obsm["X_fa"] = X_lat
    A.varm["PCs"]  = W
    A.obs["target"] = pd.Categorical(labels, categories=["CTRL","TGT_A","TGT_B"])
    return A


def test_fit_bridge_branches_runs_and_stores_uns():
    """One control vs two targets — both bridges must train and produce
    the standard uns layout (max_eig_fwd, df_fwd, t_vals, _bridge, …)."""
    A = _synthetic_perturb_adata()

    out = sjd.tl.fit_bridge_branches(
        A,
        groupby="target",
        src_group="CTRL",
        tgt_groups=["TGT_A", "TGT_B"],
        rep="X_fa",
        # Keep cheap so the test runs in seconds
        epsilon=0.5,
        max_iterations=2,
        n_score_steps=40,
        n_traj=60,
        steps=40,
        n_archetypes=3,
        n_genes=10,
        seed=0,
        verbose=False,
    )
    assert set(out.keys()) == {"TGT_A", "TGT_B"}
    # auto_time_key created the obs column
    assert "bridge_t" in A.obs.columns
    assert set(A.obs["bridge_t"].unique()) <= {0.0, 1.0}
    # uns entries exist with the expected structure
    for tgt in ("TGT_A", "TGT_B"):
        uns_key = f"scjdo_bridge_{tgt}"
        assert uns_key in A.uns
        res = A.uns[uns_key]
        for k in ("max_eig_fwd", "max_eig_bwd", "df_fwd", "df_bwd",
                  "t_vals", "_bridge", "src_mask", "tgt_mask"):
            assert k in res, f"{uns_key}: missing '{k}'"
        # populations match the requested groupby labels
        assert int(res["src_mask"].sum()) == 200      # CTRL
        assert int(res["tgt_mask"].sum()) == 80       # TGT_A or TGT_B


def test_fit_bridge_branches_missing_src_raises():
    A = _synthetic_perturb_adata()
    with pytest.raises(ValueError, match="src_group"):
        sjd.tl.fit_bridge_branches(
            A, groupby="target", src_group="NotPresent",
            tgt_groups=["TGT_A"], rep="X_fa",
            max_iterations=1, n_score_steps=10, verbose=False,
        )


def test_fit_bridge_branches_missing_groupby_raises():
    A = _synthetic_perturb_adata()
    with pytest.raises(KeyError, match="groupby"):
        sjd.tl.fit_bridge_branches(
            A, groupby="not_a_column", src_group="CTRL",
            tgt_groups=["TGT_A"], rep="X_fa",
            max_iterations=1, n_score_steps=10, verbose=False,
        )


def test_fit_bridge_branches_skips_missing_target():
    """A nonexistent target should warn (not raise) and be omitted from out."""
    A = _synthetic_perturb_adata()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = sjd.tl.fit_bridge_branches(
            A, groupby="target", src_group="CTRL",
            tgt_groups=["TGT_A", "NotPresent"], rep="X_fa",
            epsilon=0.5, max_iterations=2, n_score_steps=20,
            n_traj=40, steps=30, n_archetypes=3, n_genes=5,
            seed=0, verbose=False,
        )
    assert "TGT_A" in out and "NotPresent" not in out
    assert any("has 0 cells" in str(wi.message) for wi in w)
