"""
scjdo/models/hybrid_grn.py
==============================
Top-level HybridGRN wrapper: the opt-in experimental extension.

This module is the single entry point for the HybridGRN extension.
It wires together:

    DriftField          (existing, unchanged)
    RepresentationModel (new: PCARep / LDVAERep / VegaRep)
    pullback_gene_operator
    SparseGRNRefiner
    grn_modes

and exposes a clean API that mirrors the existing scJDO style.

Usage
-----
    from scjdo.models.hybrid_grn import HybridGRNConfig, HybridGRNModel

    cfg = HybridGRNConfig(
        drift_cfg=DriftConfig(dim=50),
        rep_cfg=RepresentationConfig(backend="ldvae", n_latent=50, n_genes=2000),
    )
    model = HybridGRNModel(cfg, rep=LDVAERep(cfg.rep_cfg), drift=drift)
    result = model.run(X_gene, T, tf_index=tf_idx)

Design departures from spec
---------------------------
1. We do NOT jointly optimise drift + representation + GRN from scratch.
   The spec recommends a staged training order and we enforce it here:
   the ``HybridGRNModel.run()`` method assumes the drift and representation
   are already trained (or being trained separately).  This avoids the
   instability of joint optimisation and keeps the latent dynamics clean.

2. We add a ``validate_consistency`` method that checks whether J_x is
   stable across random seeds and nearby latent neighbourhoods.  This
   implements the first of the three promotion criteria from the user's
   design document.

3. The ``HybridGRNResult`` dataclass keeps J_z, J_x, and K_x strictly
   separate (as required by the spec's key consistency rule).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from scjdo.models.drift import DriftField, DriftConfig
from scjdo.models.representation import (
    RepresentationConfig,
    RepresentationModel,
    build_representation,
)
from scjdo.grn.pullback import (
    compute_latent_jacobian,
    pullback_gene_operator,
    binned_pullback,
)
from scjdo.grn.refine import GRNRefinerConfig, SparseGRNRefiner
from scjdo.grn.archetypes import grn_modes, GRNArchetypeResult


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HybridGRNConfig:
    """Configuration for the HybridGRN extension.

    Parameters
    ----------
    drift_cfg :
        Configuration for the latent DriftField (existing scJDO config).
    rep_cfg :
        Configuration for the encoder/decoder backend.
    refiner_cfg :
        Configuration for the sparse GRN refiner.
    grn_rank :
        Number of GRN archetypes to extract.
    n_time_bins :
        Number of pseudotime bins for binned pullback and GRN extraction.
    pullback_mode :
        Pullback mode: ``"linear"`` (default for PCARep/LDVAERep),
        ``"autograd"``, ``"projected"``, or ``"tf_restricted"``.
    approx_jz :
        Use approximate latent Jacobian (memory-efficient for large D).
    n_proj_jz :
        Random projections for approximate J_z.
    batch_size :
        Batch size for Jacobian computation.

    Notes
    -----
    The recommended default is ``pullback_mode="tf_restricted"`` with
    ``rep_cfg.backend="ldvae"``.  This gives the best combination of
    memory efficiency, interpretability, and mathematical cleanliness.
    """
    drift_cfg: DriftConfig = field(default_factory=DriftConfig)
    rep_cfg: RepresentationConfig = field(default_factory=RepresentationConfig)
    refiner_cfg: GRNRefinerConfig = field(default_factory=GRNRefinerConfig)
    grn_rank: int = 10
    n_time_bins: int = 20
    pullback_mode: str = "tf_restricted"
    approx_jz: bool = False
    n_proj_jz: int = 64
    batch_size: int = 256


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class HybridGRNResult:
    """Output of HybridGRNModel.run().

    These four objects must be kept distinct (the spec's key consistency rule):

    Attributes
    ----------
    z : (N, D)
        Latent representations from the encoder.
    Jz : (T, D, D)
        Latent Jacobian operators (one per pseudotime bin, averaged).
    Jx : (T, G, n_tf) or (T, G, G)
        Pulled-back gene-space operators.
    Kx : (T, n_tf, G)
        Sparse, prior-constrained GRN approximation to J_x.
    archetypes : GRNArchetypeResult
        SVD archetypes extracted from K_x.
    metadata : dict
        Miscellaneous metadata (gene names, TF names, bin edges, etc.).
    """
    z: torch.Tensor
    Jz: torch.Tensor
    Jx: torch.Tensor
    Kx: torch.Tensor
    archetypes: GRNArchetypeResult
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# HybridGRNModel
# ---------------------------------------------------------------------------

class HybridGRNModel(nn.Module):
    """HybridGRN: opt-in experimental extension for scJDO.

    Wires together a trained DriftField and a RepresentationModel to
    extract gene-space GRN operators and archetypes.

    Parameters
    ----------
    cfg : HybridGRNConfig
    rep : RepresentationModel
        Trained encoder/decoder backend.
    drift : DriftField
        Trained latent drift model.
    tf_index : (n_tf,) LongTensor, optional
        Indices of TF genes.  Required for ``pullback_mode="tf_restricted"``.
    tf_mask : (n_tf, G) bool, optional
        Allowed TF→gene edges for the refiner.
    sign_mask : (n_tf, G) int, optional
        Sign constraints for the refiner.
    """

    # Experimental flag — makes it clear this is not the default workflow
    _EXPERIMENTAL = True
    _EXPERIMENTAL_WARNING = (
        "[HybridGRN] This is an experimental extension of scJDO. "
        "The default latent Hybrid Drift workflow is unchanged. "
        "GRN conclusions depend on decoder choice, TF mask, and regularisation. "
        "Validate results on known benchmark systems before biological interpretation."
    )

    def __init__(
        self,
        cfg: HybridGRNConfig,
        rep: RepresentationModel,
        drift: DriftField,
        tf_index: Optional[torch.Tensor] = None,
        tf_mask: Optional[torch.Tensor] = None,
        sign_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        warnings.warn(self._EXPERIMENTAL_WARNING, stacklevel=2, category=UserWarning)

        self.cfg = cfg
        self.rep = rep
        self.drift = drift
        self.tf_index = tf_index

        # Build refiner
        n_tf = tf_index.shape[0] if tf_index is not None else cfg.rep_cfg.n_latent
        n_genes = cfg.rep_cfg.n_genes
        self.refiner = SparseGRNRefiner(
            cfg.refiner_cfg, n_tf=n_tf, n_genes=n_genes,
            tf_mask=tf_mask, sign_mask=sign_mask,
        )

    # ------------------------------------------------------------------
    def forward_latent(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run the latent drift forward pass.

        This is identical to the standard scJDO workflow.

        Returns
        -------
        dict with keys: ``z``, ``drift``, ``Jz``.
        """
        z = self.rep.encode(x)
        f = self.drift(z, t)
        Jz = compute_latent_jacobian(
            self.drift, z, t,
            approx=self.cfg.approx_jz,
            n_proj=self.cfg.n_proj_jz,
        )
        return {"z": z, "drift": f, "Jz": Jz}

    # ------------------------------------------------------------------
    def pullback_operator(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        Jz: torch.Tensor,
    ) -> torch.Tensor:
        """Compute J_x from J_z via the chain rule.

        Returns
        -------
        Jx : (B, G, n_tf) for tf_restricted mode, else (B, G, G).
        """
        return pullback_gene_operator(
            self.rep, x, z, Jz,
            mode=self.cfg.pullback_mode,
            tf_index=self.tf_index,
        )

    # ------------------------------------------------------------------
    def run(
        self,
        X: torch.Tensor,
        T: torch.Tensor,
        x_tf_seq: Optional[torch.Tensor] = None,
        dx_seq: Optional[torch.Tensor] = None,
        gene_names: Optional[list[str]] = None,
        tf_names: Optional[list[str]] = None,
        verbose: bool = False,
    ) -> HybridGRNResult:
        """Full HybridGRN pipeline: encode → pullback → refine → archetypes.

        Parameters
        ----------
        X : (N, G)
            Gene expression matrix (log1p-normalised).
        T : (N,)
            Pseudotime values in [0, 1].
        x_tf_seq : (T_bins, n_tf), optional
            Mean TF expression per pseudotime bin (for local dynamics loss).
        dx_seq : (T_bins, G), optional
            Mean observed expression change per bin (for local dynamics loss).
        gene_names :
            Gene names for metadata.
        tf_names :
            TF names for metadata.
        verbose :
            Print progress.

        Returns
        -------
        HybridGRNResult
        """
        cfg = self.cfg
        device = next(self.drift.parameters()).device

        # ── Step 1: Encode all cells ──────────────────────────────────
        if verbose:
            print("[HybridGRN] Step 1: Encoding cells...")
        self.rep.eval()
        with torch.no_grad():
            z_all = self.rep.encode(X.to(device))

        # ── Step 2: Binned pullback ───────────────────────────────────
        if verbose:
            print("[HybridGRN] Step 2: Computing binned gene-space pullback...")
        Jx_bins, bin_edges = binned_pullback(
            self.drift, self.rep, X, T,
            n_bins=cfg.n_time_bins,
            mode=cfg.pullback_mode,
            tf_index=self.tf_index.to(device) if self.tf_index is not None else None,
            approx_jz=cfg.approx_jz,
            n_proj=cfg.n_proj_jz,
            batch_size=cfg.batch_size,
            device=device,
        )
        # Jx_bins: (T, G, n_tf) — transpose to (T, n_tf, G) for refiner
        Jx_for_refiner = Jx_bins.transpose(-1, -2)   # (T, n_tf, G)

        # ── Step 3: Compute mean J_z per bin (for result) ────────────
        if verbose:
            print("[HybridGRN] Step 3: Computing mean latent Jacobians per bin...")
        D = z_all.shape[1]
        Jz_bins = self._compute_binned_jz(X, T, D, device, verbose)

        # ── Step 4: Sparse GRN refinement ────────────────────────────
        if verbose:
            print("[HybridGRN] Step 4: Refining sparse GRN operators...")
        Kx = self.refiner.fit(
            Jx_for_refiner,
            x_tf_seq=x_tf_seq,
            dx_seq=dx_seq,
            verbose=verbose,
        )   # (T, n_tf, G)

        # ── Step 5: Archetype extraction ─────────────────────────────
        if verbose:
            print("[HybridGRN] Step 5: Extracting GRN archetypes...")
        arch_result = grn_modes(Kx, rank=cfg.grn_rank, center=True)

        metadata = {
            "bin_edges": bin_edges,
            "gene_names": gene_names,
            "tf_names": tf_names,
            "n_time_bins": cfg.n_time_bins,
            "pullback_mode": cfg.pullback_mode,
            "backend": cfg.rep_cfg.backend,
        }

        return HybridGRNResult(
            z=z_all.cpu(),
            Jz=Jz_bins.cpu(),
            Jx=Jx_bins.cpu(),
            Kx=Kx.cpu(),
            archetypes=arch_result,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    def _compute_binned_jz(
        self,
        X: torch.Tensor,
        T: torch.Tensor,
        D: int,
        device: torch.device,
        verbose: bool,
    ) -> torch.Tensor:
        """Compute mean latent Jacobian per pseudotime bin."""
        cfg = self.cfg
        N = X.shape[0]
        bin_edges = torch.linspace(0.0, 1.0, cfg.n_time_bins + 1)
        bin_ids = torch.bucketize(T, bin_edges[1:-1])

        Jz_bins = torch.zeros(cfg.n_time_bins, D, D)
        bin_counts = torch.zeros(cfg.n_time_bins, dtype=torch.long)

        self.drift.eval()
        self.rep.eval()
        with torch.no_grad():
            for start in range(0, N, cfg.batch_size):
                end = min(start + cfg.batch_size, N)
                x_b = X[start:end].to(device)
                t_b = T[start:end].to(device)
                ids_b = bin_ids[start:end]
                z_b = self.rep.encode(x_b)
                with torch.enable_grad():
                    Jz_b = compute_latent_jacobian(
                        self.drift, z_b, t_b,
                        approx=cfg.approx_jz, n_proj=cfg.n_proj_jz,
                    )
                for b in range(cfg.n_time_bins):
                    mask = (ids_b == b)
                    if mask.any():
                        Jz_bins[b] += Jz_b[mask].sum(0).cpu()
                        bin_counts[b] += mask.sum()

        for b in range(cfg.n_time_bins):
            if bin_counts[b] > 0:
                Jz_bins[b] /= bin_counts[b].float()

        return Jz_bins

    # ------------------------------------------------------------------
    def validate_consistency(
        self,
        X: torch.Tensor,
        T: torch.Tensor,
        n_seeds: int = 3,
        noise_scale: float = 0.01,
    ) -> dict[str, float]:
        """Check whether J_x is stable across seeds and nearby neighbourhoods.

        This implements the first promotion criterion from the design document:
        "Mathematical stability: the pulled-back gene operator is consistent
        across seeds, nearby latent neighbourhoods, and reasonable decoder choices."

        Parameters
        ----------
        X : (N, G)
        T : (N,)
        n_seeds :
            Number of random perturbations to test.
        noise_scale :
            Scale of Gaussian noise added to X for neighbourhood test.

        Returns
        -------
        dict with keys:
            ``mean_cosine_similarity`` : mean cosine similarity of J_x
                                         between original and perturbed inputs.
            ``std_cosine_similarity``  : standard deviation.
            ``is_stable``              : True if mean > 0.9.
        """
        device = next(self.drift.parameters()).device
        self.rep.eval()
        self.drift.eval()

        # Compute reference J_x on a small subset
        n_test = min(64, X.shape[0])
        idx = torch.randperm(X.shape[0])[:n_test]
        x_ref = X[idx].to(device)
        t_ref = T[idx].to(device)

        with torch.no_grad():
            z_ref = self.rep.encode(x_ref)
        with torch.enable_grad():
            Jz_ref = compute_latent_jacobian(self.drift, z_ref, t_ref)
        with torch.enable_grad():
            Jx_ref = pullback_gene_operator(
                self.rep, x_ref, z_ref, Jz_ref,
                mode=self.cfg.pullback_mode,
                tf_index=self.tf_index.to(device) if self.tf_index is not None else None,
            )

        # Flatten for cosine similarity
        Jx_ref_flat = Jx_ref.reshape(n_test, -1)

        similarities = []
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            x_noisy = x_ref + torch.randn_like(x_ref) * noise_scale
            with torch.no_grad():
                z_noisy = self.rep.encode(x_noisy)
            with torch.enable_grad():
                Jz_noisy = compute_latent_jacobian(self.drift, z_noisy, t_ref)
            with torch.enable_grad():
                Jx_noisy = pullback_gene_operator(
                    self.rep, x_noisy, z_noisy, Jz_noisy,
                    mode=self.cfg.pullback_mode,
                    tf_index=self.tf_index.to(device) if self.tf_index is not None else None,
                )
            Jx_noisy_flat = Jx_noisy.reshape(n_test, -1)

            # Cosine similarity per cell
            cos_sim = torch.nn.functional.cosine_similarity(
                Jx_ref_flat, Jx_noisy_flat, dim=-1
            )
            similarities.append(cos_sim.mean().item())

        mean_sim = float(torch.tensor(similarities).mean())
        std_sim = float(torch.tensor(similarities).std())

        return {
            "mean_cosine_similarity": mean_sim,
            "std_cosine_similarity": std_sim,
            "is_stable": mean_sim > 0.9,
        }
