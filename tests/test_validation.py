"""
Tests for scjdo.validation — null models, robustness, identifiability.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from scjdo.validation.identifiability import (
    archetype_cosine_similarity,
    instability_peak_overlap,
    model_sensitivity_report,
)
from scjdo.validation.null_models import (
    continuous_control_null,
    run_null_comparison,
    temporal_shuffle_null,
)
from scjdo.validation.robustness import (
    gene_overlap_across_pseudotimes,
    pseudotime_sensitivity_report,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_jacobian_tensor():
    """A (20, 4, 4) random Jacobian tensor for fast testing."""
    torch.manual_seed(42)
    return torch.randn(20, 4, 4)


@pytest.fixture()
def structured_jacobian_tensor():
    """
    A (40, 4, 4) Jacobian tensor with two clear sequential regimes:
    archetype A dominates t=0..19, archetype B dominates t=20..39.
    This should produce a strong sequential handoff signal.
    """
    torch.manual_seed(0)
    A = torch.randn(4, 4)
    B = torch.randn(4, 4)
    frames = []
    for t in range(40):
        alpha = max(0.0, 1.0 - t / 20.0)
        beta = min(1.0, t / 20.0)
        frames.append(alpha * A + beta * B + 0.05 * torch.randn(4, 4))
    return torch.stack(frames)  # (40, 4, 4)


# ---------------------------------------------------------------------------
# null_models
# ---------------------------------------------------------------------------

class TestTemporalShuffleNull:
    def test_returns_expected_keys(self, small_jacobian_tensor):
        result = temporal_shuffle_null(
            small_jacobian_tensor, rank=2, n_shuffles=5, seed=0
        )
        for key in ("observed", "null_mean", "null_std", "null_all",
                    "p_sequential", "p_concurrent", "summary"):
            assert key in result

    def test_p_values_in_range(self, small_jacobian_tensor):
        result = temporal_shuffle_null(
            small_jacobian_tensor, rank=2, n_shuffles=10, seed=1
        )
        assert 0.0 <= result["p_sequential"] <= 1.0
        assert 0.0 <= result["p_concurrent"] <= 1.0

    def test_null_all_length(self, small_jacobian_tensor):
        n = 8
        result = temporal_shuffle_null(
            small_jacobian_tensor, rank=2, n_shuffles=n, seed=2
        )
        assert len(result["null_all"]) == n

    def test_summary_is_string(self, small_jacobian_tensor):
        result = temporal_shuffle_null(
            small_jacobian_tensor, rank=2, n_shuffles=5, seed=3
        )
        assert isinstance(result["summary"], str)
        assert "sequential" in result["summary"].lower()

    def test_structured_tensor_lower_p(self, structured_jacobian_tensor):
        """Structured tensor should have lower shuffle p-value than random."""
        result = temporal_shuffle_null(
            structured_jacobian_tensor, rank=3, n_shuffles=50, seed=0
        )
        # Observed sequential fraction should be higher than null mean
        assert result["observed"]["sequential_frac"] >= result["null_mean"]["sequential_frac"] - 0.3


class TestContinuousControlNull:
    def test_returns_expected_keys(self):
        result = continuous_control_null(T=20, d=4, rank=2, n_replicates=3, seed=0)
        for key in ("null_mean", "null_std", "null_all", "summary"):
            assert key in result

    def test_null_all_length(self):
        n = 5
        result = continuous_control_null(T=20, d=4, rank=2, n_replicates=n, seed=1)
        assert len(result["null_all"]) == n

    def test_fractions_in_range(self):
        result = continuous_control_null(T=30, d=4, rank=3, n_replicates=5, seed=2)
        for key in ("sequential_frac", "concurrent_frac"):
            assert 0.0 <= result["null_mean"][key] <= 1.0


class TestRunNullComparison:
    def test_combined_output(self, small_jacobian_tensor):
        result = run_null_comparison(
            small_jacobian_tensor, rank=2, n_shuffles=5, n_continuous=3, seed=0
        )
        assert "shuffle" in result
        assert "continuous" in result
        assert "summary" in result
        assert "NULL MODEL COMPARISON" in result["summary"]


# ---------------------------------------------------------------------------
# robustness
# ---------------------------------------------------------------------------

class TestGeneOverlapAcrossPseudotimes:
    @pytest.fixture()
    def gene_rankings(self):
        genes = [f"gene_{i}" for i in range(200)]
        rng = np.random.default_rng(0)
        return {
            "DPT": list(rng.permutation(genes)),
            "Palantir": list(rng.permutation(genes)),
            "Slingshot": list(rng.permutation(genes)),
        }

    def test_returns_expected_keys(self, gene_rankings):
        result = gene_overlap_across_pseudotimes(gene_rankings, top_k_values=[50, 100])
        for key in ("jaccard_by_k", "mean_jaccard", "summary"):
            assert key in result

    def test_jaccard_range(self, gene_rankings):
        result = gene_overlap_across_pseudotimes(gene_rankings, top_k_values=[50])
        for pair_val in result["jaccard_by_k"][50].values():
            assert 0.0 <= pair_val <= 1.0

    def test_identical_rankings_give_jaccard_one(self):
        genes = [f"gene_{i}" for i in range(100)]
        rankings = {"A": genes, "B": genes}
        result = gene_overlap_across_pseudotimes(rankings, top_k_values=[50])
        assert result["jaccard_by_k"][50]["A vs B"] == pytest.approx(1.0)

    def test_disjoint_rankings_give_jaccard_zero(self):
        genes_a = [f"gene_a_{i}" for i in range(100)]
        genes_b = [f"gene_b_{i}" for i in range(100)]
        rankings = {"A": genes_a, "B": genes_b}
        result = gene_overlap_across_pseudotimes(rankings, top_k_values=[50])
        assert result["jaccard_by_k"][50]["A vs B"] == pytest.approx(0.0)

    def test_spearman_with_score_arrays(self, gene_rankings):
        rng = np.random.default_rng(1)
        score_arrays = {m: rng.random(200) for m in gene_rankings}
        result = gene_overlap_across_pseudotimes(
            gene_rankings, top_k_values=[50], score_arrays=score_arrays
        )
        assert len(result["spearman_by_pair"]) > 0
        for r in result["spearman_by_pair"].values():
            assert -1.0 <= r <= 1.0

    def test_summary_contains_method_names(self, gene_rankings):
        result = gene_overlap_across_pseudotimes(gene_rankings, top_k_values=[50])
        for method in gene_rankings:
            assert method in result["summary"]


class TestPseudotimeSensitivityReport:
    def test_returns_string(self):
        genes = [f"gene_{i}" for i in range(100)]
        rng = np.random.default_rng(0)
        rankings = {
            "DPT": list(rng.permutation(genes)),
            "Palantir": list(rng.permutation(genes)),
        }
        report = pseudotime_sensitivity_report(rankings, top_k_values=[50])
        assert isinstance(report, str)
        assert len(report) > 0


# ---------------------------------------------------------------------------
# identifiability
# ---------------------------------------------------------------------------

class TestArchetypeCosineSimilarity:
    def test_identical_archetypes_give_one(self):
        arch = torch.randn(4, 6, 6)
        result = archetype_cosine_similarity(arch, arch)
        assert result["median"] == pytest.approx(1.0, abs=1e-4)

    def test_random_archetypes_below_one(self):
        torch.manual_seed(0)
        a = torch.randn(4, 6, 6)
        b = torch.randn(4, 6, 6)
        result = archetype_cosine_similarity(a, b)
        assert result["median"] < 1.0

    def test_returns_expected_keys(self):
        a = torch.randn(3, 4, 4)
        b = torch.randn(3, 4, 4)
        result = archetype_cosine_similarity(a, b)
        for key in ("per_archetype", "median", "min", "summary"):
            assert key in result

    def test_per_archetype_length(self):
        K = 5
        a = torch.randn(K, 4, 4)
        b = torch.randn(K, 4, 4)
        result = archetype_cosine_similarity(a, b)
        assert len(result["per_archetype"]) == K

    def test_mismatched_K_uses_min(self):
        a = torch.randn(5, 4, 4)
        b = torch.randn(3, 4, 4)
        result = archetype_cosine_similarity(a, b)
        assert len(result["per_archetype"]) == 3


class TestInstabilityPeakOverlap:
    def test_identical_curves_full_overlap(self):
        curve = np.zeros(50)
        curve[25] = 1.0
        result = instability_peak_overlap([curve, curve, curve])
        assert result["overlap_frac"] == pytest.approx(1.0)

    def test_distant_peaks_no_overlap(self):
        c1 = np.zeros(100)
        c1[5] = 1.0
        c2 = np.zeros(100)
        c2[95] = 1.0
        result = instability_peak_overlap([c1, c2], peak_window=0.05)
        assert result["overlap_frac"] == pytest.approx(0.0)

    def test_returns_expected_keys(self):
        curves = [np.random.rand(30) for _ in range(4)]
        result = instability_peak_overlap(curves)
        for key in ("peak_locations", "peak_std", "overlap_frac", "summary"):
            assert key in result

    def test_peak_locations_in_unit_interval(self):
        curves = [np.random.rand(50) for _ in range(5)]
        result = instability_peak_overlap(curves)
        for loc in result["peak_locations"]:
            assert 0.0 <= loc <= 1.0


class TestModelSensitivityReport:
    def test_returns_string(self):
        configs = {
            "depth=4, hidden=256": {
                "archetypes": torch.randn(3, 4, 4),
                "instability_curve": np.random.rand(30),
                "auroc": 0.72,
            },
            "depth=6, hidden=128": {
                "archetypes": torch.randn(3, 4, 4),
                "instability_curve": np.random.rand(30),
                "auroc": 0.70,
            },
        }
        report = model_sensitivity_report(configs)
        assert isinstance(report, str)
        assert "Invariant" in report
        assert "Non-invariant" in report
        assert "AUROC" in report

    def test_no_auroc_key(self):
        configs = {
            "A": {
                "archetypes": torch.randn(3, 4, 4),
                "instability_curve": np.random.rand(30),
            },
            "B": {
                "archetypes": torch.randn(3, 4, 4),
                "instability_curve": np.random.rand(30),
            },
        }
        report = model_sensitivity_report(configs)
        assert "AUROC" not in report
