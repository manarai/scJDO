"""
scqdiff/grn/perturb.py
=======================
In-silico TF perturbation scoring using the refined GRN operators.

This module provides two complementary perturbation analyses:

    1. ``knockout_score``   — simulate TF knockout by zeroing out a TF's
                              row in K_x and measuring the predicted
                              change in target gene expression.

    2. ``control_score``    — compute the minimum control energy required
                              to drive a cell from its current state to a
                              target state via the learned drift field.
                              This is the scQDiff-native perturbation metric.

Design note
-----------
The spec listed these under ``grn/perturb.py`` and ``analysis/grn_scores.py``.
We merge the perturbation logic here and keep ``analysis/grn_scores.py``
for higher-level summary statistics (regulator centrality, branch scores).

The key distinction we maintain:
    - Perturbation scores here are *local* (per-cell or per-bin).
    - Summary scores in grn_scores.py are *global* (aggregated over the
      full pseudotime trajectory or cell type).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# TF knockout score
# ---------------------------------------------------------------------------

def knockout_score(
    Kx: torch.Tensor,
    x_tf: torch.Tensor,
    tf_index: int,
    x_gene: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Predict the effect of knocking out a single TF.

    Simulates TF knockout by zeroing out the TF's row in K_x and
    computing the difference in predicted gene expression change.

    Parameters
    ----------
    Kx : (T_or_B, n_tf, G)
        Refined GRN operators.
    x_tf : (T_or_B, n_tf)
        TF expression values (mean per bin or per cell).
    tf_index :
        Index of the TF to knock out (row index in n_tf dimension).
    x_gene : (T_or_B, G), optional
        Full gene expression.  If provided, the score is the predicted
        change in gene expression; otherwise it is the change in drift.

    Returns
    -------
    ko_effect : (T_or_B, G)
        Predicted change in gene expression (or drift) due to TF knockout.
        Positive values indicate genes that are *upregulated* after KO
        (i.e. the TF was repressing them); negative values indicate
        genes that are *downregulated* after KO (i.e. the TF was activating them).
    """
    # Wild-type predicted change
    dx_wt = torch.einsum("btg,bt->bg", Kx, x_tf)   # (T_or_B, G)

    # Knockout: zero out TF row
    Kx_ko = Kx.clone()
    Kx_ko[:, tf_index, :] = 0.0

    # Knockout predicted change
    x_tf_ko = x_tf.clone()
    x_tf_ko[:, tf_index] = 0.0
    dx_ko = torch.einsum("btg,bt->bg", Kx_ko, x_tf_ko)   # (T_or_B, G)

    return dx_ko - dx_wt   # (T_or_B, G)


def batch_knockout_scores(
    Kx: torch.Tensor,
    x_tf: torch.Tensor,
) -> torch.Tensor:
    """Compute knockout scores for all TFs simultaneously.

    Parameters
    ----------
    Kx : (T, n_tf, G)
    x_tf : (T, n_tf)

    Returns
    -------
    scores : (n_tf, T, G)
        Knockout effect for each TF at each time bin for each gene.
    """
    n_tf = Kx.shape[1]
    scores = torch.stack(
        [knockout_score(Kx, x_tf, i) for i in range(n_tf)],
        dim=0,
    )   # (n_tf, T, G)
    return scores


# ---------------------------------------------------------------------------
# Control energy score
# ---------------------------------------------------------------------------

def control_energy_score(
    drift,
    z_start: torch.Tensor,
    z_target: torch.Tensor,
    t_start: float = 0.0,
    t_end: float = 1.0,
    n_steps: int = 50,
) -> torch.Tensor:
    """Estimate the control energy to drive cells from z_start to z_target.

    Uses the learned drift field to simulate forward trajectories and
    computes the integrated squared control input needed to reach z_target.

    This is the scQDiff-native perturbation metric: it measures how
    "far" a cell is from a target state in terms of the regulatory
    effort required, not just Euclidean distance.

    Parameters
    ----------
    drift :
        Trained DriftField model.
    z_start : (B, D)
        Starting latent states.
    z_target : (B, D)
        Target latent states.
    t_start, t_end :
        Pseudotime range for integration.
    n_steps :
        Number of Euler integration steps.

    Returns
    -------
    energy : (B,)
        Estimated control energy per cell.
    """
    dt = (t_end - t_start) / n_steps
    z = z_start.clone()
    energy = torch.zeros(z.shape[0], device=z.device)

    drift.eval()
    with torch.no_grad():
        for step in range(n_steps):
            t = torch.full((z.shape[0],), t_start + step * dt, device=z.device)
            # Natural drift
            f_nat = drift(z, t)
            # Residual control needed to stay on path to target
            z_desired = z_start + (z_target - z_start) * ((step + 1) / n_steps)
            u_control = (z_desired - z) / dt - f_nat
            # Accumulate squared control energy
            energy += (u_control ** 2).sum(dim=-1) * dt
            # Step forward
            z = z + f_nat * dt

    return energy


# ---------------------------------------------------------------------------
# Perturbation direction score
# ---------------------------------------------------------------------------

def perturbation_direction_score(
    Kx: torch.Tensor,
    tf_index: int,
    archetype_directions: torch.Tensor,
) -> torch.Tensor:
    """Score how much a TF perturbation aligns with archetype directions.

    Parameters
    ----------
    Kx : (T, n_tf, G)
        Refined GRN operators.
    tf_index :
        Index of the TF to perturb.
    archetype_directions : (rank, G)
        Gene-space archetype directions (gene_scores from GRNArchetypeResult).

    Returns
    -------
    scores : (T, rank)
        Alignment of TF perturbation with each archetype at each time bin.
        High positive score = TF perturbation drives cells along archetype k.
    """
    # TF's regulatory programme at each time: K_x[:, tf_index, :]  (T, G)
    tf_programme = Kx[:, tf_index, :]   # (T, G)

    # Cosine similarity with each archetype direction
    tf_norm = F.normalize(tf_programme, dim=-1)   # (T, G)
    arch_norm = F.normalize(archetype_directions, dim=-1)   # (rank, G)

    scores = tf_norm @ arch_norm.T   # (T, rank)
    return scores
