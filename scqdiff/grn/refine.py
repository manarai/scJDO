"""
scqdiff/grn/refine.py
======================
Sparse GRN refinement: fit a structured operator K_x that stays close
to the pulled-back J_x but obeys biological constraints.

Objective
---------
For each local bin or neighbourhood:

    L_grn = ||K_x - J_x||_F²
          + λ_sparse  · ||K_x||_1
          + λ_prior   · Ω_prior(K_x)
          + λ_local   · Ω_local(K_x, x, dx)
          + λ_temporal· Ω_temporal(K_x_seq)
          + λ_stability · Ω_stability(K_x)

Parameterisation
----------------
K_x has shape (B_or_T, n_tf, G) — TF→gene, not full gene→gene.

This is a deliberate departure from the spec's (B, G, G) suggestion.
Reasons:
  1. TF→gene is far more identifiable (n_tf << G, typically 50–500 vs 2000+).
  2. It matches the biological prior: TFs regulate target genes, not the
     reverse.  Gene→gene edges are mostly indirect.
  3. It reduces the parameter count by G/n_tf (10–50×) and makes the
     L1 sparsity penalty much more meaningful.
  4. It aligns with CellOracle / SCENIC style, which the spec endorses.

The full gene→gene matrix can be recovered if needed by multiplying
K_x (n_tf, G) by a TF indicator matrix, but we do not expose that by
default.

Optimisation
------------
We solve the refinement problem per-bin using a small number of gradient
steps (default 200) with Adam.  This is fast because K_x is small
(n_tf × G) and the objective is convex in K_x when J_x is fixed.

Alternatively, for the L1 + Frobenius objective without the local
dynamics term, there is a closed-form soft-thresholding solution which
we provide as ``refine_closed_form`` for speed.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GRNRefinerConfig:
    """Configuration for SparseGRNRefiner.

    Parameters
    ----------
    lambda_sparse :
        Weight for L1 sparsity on K_x entries.
    lambda_prior :
        Weight for prior mask penalty (penalise edges outside allowed support).
    lambda_local :
        Weight for local dynamics consistency: ||K_x x_tf - dx||².
    lambda_temporal :
        Weight for temporal smoothness: ||K_t - K_{t-1}||_F².
    lambda_stability :
        Weight for stability regulariser: penalise large positive eigenvalues.
    eig_clip :
        Eigenvalue threshold for stability regulariser.
    n_steps :
        Number of gradient steps per bin.
    lr :
        Learning rate for Adam optimiser.
    positive_targets_only :
        If True, clamp K_x to non-negative values (activation only).
    use_closed_form :
        If True and no local/temporal terms, use fast soft-thresholding.
    """
    lambda_sparse: float = 1e-3
    lambda_prior: float = 1e-2
    lambda_local: float = 0.1
    lambda_temporal: float = 1e-2
    lambda_stability: float = 1e-3
    eig_clip: float = 0.5
    n_steps: int = 200
    lr: float = 1e-2
    positive_targets_only: bool = False
    use_closed_form: bool = False


# ---------------------------------------------------------------------------
# Individual loss terms
# ---------------------------------------------------------------------------

def loss_pullback(Kx: torch.Tensor, Jx: torch.Tensor) -> torch.Tensor:
    """||K_x - J_x||_F² — keep K_x close to the pulled-back operator.

    J_x is detached so the GRN head does not destabilise drift training.
    """
    return F.mse_loss(Kx, Jx.detach())


def loss_sparse(Kx: torch.Tensor) -> torch.Tensor:
    """L1 sparsity on K_x entries."""
    return Kx.abs().mean()


def loss_prior(
    Kx: torch.Tensor,
    tf_mask: Optional[torch.Tensor] = None,
    sign_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Prior mask penalty.

    Parameters
    ----------
    Kx : (B_or_T, n_tf, G)
    tf_mask : (n_tf, G) bool
        Allowed TF→gene edges.  Edges outside this mask are penalised.
    sign_mask : (n_tf, G) int {-1, 0, +1}
        Optional sign constraint.  +1 → activator (penalise negative K),
        -1 → repressor (penalise positive K), 0 → unconstrained.
    """
    loss = Kx.new_zeros(())

    if tf_mask is not None:
        # Penalise edges outside allowed support
        forbidden = ~tf_mask.bool()   # (n_tf, G)
        if Kx.dim() == 3:
            forbidden = forbidden.unsqueeze(0)
        loss = loss + Kx[forbidden].pow(2).mean()

    if sign_mask is not None:
        sm = sign_mask.float()
        if Kx.dim() == 3:
            sm = sm.unsqueeze(0)
        # Penalise activators being negative
        act_mask = (sm > 0)
        if act_mask.any():
            loss = loss + torch.relu(-Kx[act_mask]).pow(2).mean()
        # Penalise repressors being positive
        rep_mask = (sm < 0)
        if rep_mask.any():
            loss = loss + torch.relu(Kx[rep_mask]).pow(2).mean()

    return loss


def loss_local_dynamics(
    Kx: torch.Tensor,
    x_tf: torch.Tensor,
    dx_obs: torch.Tensor,
) -> torch.Tensor:
    """Local dynamics consistency.

    Predicts gene expression changes as K_x · x_tf and compares to
    observed changes dx_obs.

    Parameters
    ----------
    Kx : (B, n_tf, G)
    x_tf : (B, n_tf)
        TF expression values.
    dx_obs : (B, G)
        Observed expression changes (e.g. x_next - x_now).

    Returns
    -------
    Scalar loss.
    """
    # dx_pred[b, g] = sum_tf K_x[b, tf, g] * x_tf[b, tf]
    dx_pred = torch.einsum("btg,bt->bg", Kx, x_tf)   # (B, G)
    return F.mse_loss(dx_pred, dx_obs)


def loss_temporal_smoothness(K_seq: torch.Tensor) -> torch.Tensor:
    """Temporal smoothness: sum_t ||K_t - K_{t-1}||_F².

    Parameters
    ----------
    K_seq : (T, n_tf, G)
    """
    if K_seq.shape[0] < 2:
        return K_seq.new_zeros(())
    diff = K_seq[1:] - K_seq[:-1]   # (T-1, n_tf, G)
    return (diff ** 2).mean()


def loss_stability(
    Kx: torch.Tensor,
    eig_clip: float = 0.5,
) -> torch.Tensor:
    """Penalise large positive eigenvalues of K_x.

    This is a soft numerical-sanity regulariser, not a biological prior.
    It discourages runaway instability in the GRN operator.

    Parameters
    ----------
    Kx : (B_or_T, n_tf, G)
        If n_tf == G (square), compute eigenvalues directly.
        Otherwise skip (non-square operators have singular values, not
        eigenvalues in the usual sense).
    eig_clip :
        Eigenvalues above this threshold are penalised.
    """
    n_tf, G = Kx.shape[-2], Kx.shape[-1]
    if n_tf != G:
        return Kx.new_zeros(())
    try:
        eig = torch.linalg.eigvals(Kx.float()).real   # (..., n_tf)
        return torch.relu(eig - eig_clip).mean()
    except Exception:
        return Kx.new_zeros(())


# ---------------------------------------------------------------------------
# Closed-form soft-thresholding (fast path)
# ---------------------------------------------------------------------------

def refine_closed_form(
    Jx: torch.Tensor,
    lambda_sparse: float,
    tf_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fast closed-form solution for L1 + Frobenius objective.

    argmin_K  ||K - J_x||_F²  +  λ ||K||_1

    Solution: K* = sign(J_x) · max(|J_x| - λ/2, 0)  (soft thresholding)

    Then apply tf_mask to zero out forbidden edges.

    Parameters
    ----------
    Jx : (T, n_tf, G) or (B, n_tf, G)
    lambda_sparse : float
    tf_mask : (n_tf, G) bool, optional

    Returns
    -------
    Kx : same shape as Jx
    """
    threshold = lambda_sparse / 2.0
    Kx = Jx.sign() * torch.relu(Jx.abs() - threshold)
    if tf_mask is not None:
        if Kx.dim() == 3:
            Kx = Kx * tf_mask.float().unsqueeze(0)
        else:
            Kx = Kx * tf_mask.float()
    return Kx


# ---------------------------------------------------------------------------
# SparseGRNRefiner
# ---------------------------------------------------------------------------

class SparseGRNRefiner(nn.Module):
    """Refine a sequence of pulled-back operators into sparse GRN operators.

    This module takes a time-binned tensor of pulled-back gene operators
    J_x (shape T × n_tf × G) and fits a sparse, prior-constrained
    approximation K_x of the same shape.

    The refinement is performed jointly over all time bins so that the
    temporal smoothness penalty can be applied.

    Parameters
    ----------
    cfg : GRNRefinerConfig
    n_tf : int
        Number of TF genes.
    n_genes : int
        Number of target genes.
    tf_mask : (n_tf, G) bool, optional
        Allowed TF→gene edges from prior knowledge.
    sign_mask : (n_tf, G) int, optional
        Sign constraints from prior knowledge.
    """

    def __init__(
        self,
        cfg: GRNRefinerConfig,
        n_tf: int,
        n_genes: int,
        tf_mask: Optional[torch.Tensor] = None,
        sign_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.n_tf = n_tf
        self.n_genes = n_genes

        if tf_mask is not None:
            self.register_buffer("tf_mask", tf_mask.bool())
        else:
            self.tf_mask = None

        if sign_mask is not None:
            self.register_buffer("sign_mask", sign_mask)
        else:
            self.sign_mask = None

    # ------------------------------------------------------------------
    def fit(
        self,
        Jx: torch.Tensor,
        x_tf_seq: Optional[torch.Tensor] = None,
        dx_seq: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Fit K_x to approximate J_x under biological constraints.

        Parameters
        ----------
        Jx : (T, n_tf, G)
            Pulled-back gene operators (one per pseudotime bin).
        x_tf_seq : (T, n_tf), optional
            Mean TF expression per bin (for local dynamics loss).
        dx_seq : (T, G), optional
            Mean observed expression change per bin (for local dynamics loss).
        verbose :
            Print loss every 50 steps.

        Returns
        -------
        Kx : (T, n_tf, G)
            Refined sparse GRN operators.
        """
        cfg = self.cfg

        # Fast path: closed-form soft thresholding
        if cfg.use_closed_form or (
            cfg.lambda_local == 0.0 and cfg.lambda_temporal == 0.0
            and cfg.lambda_stability == 0.0 and cfg.lambda_prior == 0.0
        ):
            return refine_closed_form(
                Jx, cfg.lambda_sparse, self.tf_mask
            )

        # Gradient-based refinement
        T, n_tf, G = Jx.shape
        Kx = nn.Parameter(Jx.clone().detach())
        opt = torch.optim.Adam([Kx], lr=cfg.lr)

        for step in range(cfg.n_steps):
            opt.zero_grad()

            total = loss_pullback(Kx, Jx)

            if cfg.lambda_sparse > 0:
                total = total + cfg.lambda_sparse * loss_sparse(Kx)

            if cfg.lambda_prior > 0:
                total = total + cfg.lambda_prior * loss_prior(
                    Kx, self.tf_mask, self.sign_mask
                )

            if cfg.lambda_local > 0 and x_tf_seq is not None and dx_seq is not None:
                total = total + cfg.lambda_local * loss_local_dynamics(
                    Kx, x_tf_seq, dx_seq
                )

            if cfg.lambda_temporal > 0:
                total = total + cfg.lambda_temporal * loss_temporal_smoothness(Kx)

            if cfg.lambda_stability > 0:
                total = total + cfg.lambda_stability * loss_stability(
                    Kx, cfg.eig_clip
                )

            total.backward()
            opt.step()

            if cfg.positive_targets_only:
                with torch.no_grad():
                    Kx.clamp_(min=0.0)

            if verbose and (step % 50 == 0 or step == cfg.n_steps - 1):
                print(f"  [GRNRefiner] step {step:4d}  loss={total.item():.4f}")

        # Apply hard mask after optimisation
        with torch.no_grad():
            if self.tf_mask is not None:
                Kx.data = Kx.data * self.tf_mask.float().unsqueeze(0)

        return Kx.detach()

    # ------------------------------------------------------------------
    def forward(
        self,
        Jx: torch.Tensor,
        x_tf_seq: Optional[torch.Tensor] = None,
        dx_seq: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Alias for ``fit``."""
        return self.fit(Jx, x_tf_seq, dx_seq, verbose=verbose)
