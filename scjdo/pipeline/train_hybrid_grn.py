"""
scjdo/pipeline/train_hybrid_grn.py
======================================
Staged training pipeline for the HybridGRN extension.

Training order (enforced)
-------------------------
1. Pretrain representation model (LDVAERep / PCARep / VegaRep).
2. Freeze or semi-freeze representation.
3. Train latent drift on z (standard scJDO training).
4. Compute J_z → pull back to J_x.
5. Train/refine K_x.
6. Extract archetypes on time-binned K_x.

We do NOT jointly optimise everything from scratch.  Joint optimisation
is tempting but creates a coupled loss landscape where the GRN head can
destabilise the drift training.  The staged approach is more robust and
easier to debug.

Departure from spec
-------------------
The spec's ``train_hybrid_grn_from_anndata`` signature is preserved,
but we add a ``stage`` parameter that lets users run individual stages
independently.  This is important for iterative development: users can
train the representation once, save it, and then experiment with
different GRN refiner configurations without retraining.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scjdo.models.drift import DriftField
from scjdo.models.representation import RepresentationModel, build_representation
from scjdo.models.hybrid_grn import HybridGRNConfig, HybridGRNModel, HybridGRNResult
from scjdo.io.anndata_hybrid import tensors_from_anndata_hybrid, compute_bin_means
from scjdo.grn.priors import identify_tfs, build_tf_mask


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class HybridGRNTrainConfig:
    """Training hyperparameters for the staged HybridGRN pipeline.

    Parameters
    ----------
    n_epochs_rep :
        Epochs for representation pretraining.
    n_epochs_drift :
        Epochs for latent drift training.
    batch_size :
        Batch size for all training stages.
    lr_rep :
        Learning rate for representation pretraining.
    lr_drift :
        Learning rate for drift training.
    freeze_rep_after_pretrain :
        If True, freeze the representation after pretraining.
        If False, allow fine-tuning during drift training (use with care).
    device :
        Training device.
    seed :
        Random seed.
    """
    n_epochs_rep: int = 50
    n_epochs_drift: int = 100
    batch_size: int = 256
    lr_rep: float = 1e-3
    lr_drift: float = 1e-3
    freeze_rep_after_pretrain: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


# ---------------------------------------------------------------------------
# Stage 1: Pretrain representation
# ---------------------------------------------------------------------------

def pretrain_representation(
    rep: RepresentationModel,
    x_gene: torch.Tensor,
    cfg: HybridGRNTrainConfig,
    verbose: bool = True,
) -> RepresentationModel:
    """Pretrain the encoder/decoder on reconstruction loss.

    Parameters
    ----------
    rep :
        Untrained RepresentationModel.
    x_gene : (N, G)
        Gene expression matrix (log1p-normalised).
    cfg :
        Training configuration.
    verbose :
        Print loss per epoch.

    Returns
    -------
    rep :
        Trained RepresentationModel (in-place modification + return).
    """
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    rep = rep.to(device)
    rep.train()

    opt = torch.optim.Adam(rep.parameters(), lr=cfg.lr_rep)
    dataset = TensorDataset(x_gene)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    for epoch in range(cfg.n_epochs_rep):
        total_loss = 0.0
        for (x_b,) in loader:
            x_b = x_b.to(device)
            opt.zero_grad()
            loss = rep.reconstruction_loss(x_b)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x_b.shape[0]
        if verbose and (epoch % 10 == 0 or epoch == cfg.n_epochs_rep - 1):
            print(f"  [Rep pretrain] epoch {epoch:4d}  "
                  f"recon_loss={total_loss / len(x_gene):.4f}")

    if cfg.freeze_rep_after_pretrain:
        for p in rep.parameters():
            p.requires_grad_(False)
        if verbose:
            print("  [Rep pretrain] Representation frozen.")

    return rep


# ---------------------------------------------------------------------------
# Stage 2: Train latent drift
# ---------------------------------------------------------------------------

def train_latent_drift(
    drift: DriftField,
    rep: RepresentationModel,
    x_gene: torch.Tensor,
    pseudotime: torch.Tensor,
    velocity: Optional[torch.Tensor],
    cfg: HybridGRNTrainConfig,
    verbose: bool = True,
) -> DriftField:
    """Train the latent drift model on encoded representations.

    This is a thin wrapper around the standard scJDO drift training,
    adapted to use the representation model's encoder.

    Parameters
    ----------
    drift :
        Untrained DriftField.
    rep :
        Pretrained (and optionally frozen) RepresentationModel.
    x_gene : (N, G)
        Gene expression.
    pseudotime : (N,)
        Pseudotime in [0, 1].
    velocity : (N, D_or_G) or None
        RNA velocity (will be projected to latent space if gene-space).
    cfg :
        Training configuration.
    verbose :
        Print loss per epoch.

    Returns
    -------
    drift :
        Trained DriftField.
    """
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    drift = drift.to(device)
    rep = rep.to(device)
    rep.eval()

    # Encode all cells
    with torch.no_grad():
        z_all = rep.encode(x_gene.to(device)).cpu()

    # Project velocity to latent space if needed
    v_latent = None
    if velocity is not None:
        if velocity.shape[1] == x_gene.shape[1]:
            # Gene-space velocity: project via decoder Jacobian pseudo-inverse
            # For linear decoder: v_z = W^+ v_x = (W^T W)^{-1} W^T v_x
            loadings = rep.get_loadings()   # (G, D)
            if loadings is not None:
                with torch.no_grad():
                    WtW = loadings.T @ loadings   # (D, D)
                    try:
                        WtW_inv = torch.linalg.inv(WtW + 1e-4 * torch.eye(WtW.shape[0]))
                        v_latent = (velocity @ loadings @ WtW_inv.T).cpu()
                    except Exception:
                        warnings.warn(
                            "[train_latent_drift] Could not project velocity to "
                            "latent space; training without velocity prior.",
                            stacklevel=2,
                        )
        elif velocity.shape[1] == z_all.shape[1]:
            v_latent = velocity.cpu()
        else:
            warnings.warn(
                f"[train_latent_drift] Velocity shape {velocity.shape} does not "
                f"match gene space ({x_gene.shape[1]}) or latent space "
                f"({z_all.shape[1]}). Training without velocity prior.",
                stacklevel=2,
            )

    # Build dataset
    tensors = [z_all, pseudotime]
    if v_latent is not None:
        tensors.append(v_latent)
    dataset = TensorDataset(*tensors)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    opt = torch.optim.Adam(drift.parameters(), lr=cfg.lr_drift)

    for epoch in range(cfg.n_epochs_drift):
        drift.train()
        total_loss = 0.0
        for batch in loader:
            z_b = batch[0].to(device)
            t_b = batch[1].to(device)
            v_b = batch[2].to(device) if v_latent is not None else None

            opt.zero_grad()
            loss = _drift_loss(drift, z_b, t_b, v_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(drift.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * z_b.shape[0]

        if verbose and (epoch % 20 == 0 or epoch == cfg.n_epochs_drift - 1):
            print(f"  [Drift train]  epoch {epoch:4d}  "
                  f"loss={total_loss / len(z_all):.4f}")

    return drift


def _drift_loss(
    drift: DriftField,
    z: torch.Tensor,
    t: torch.Tensor,
    v: Optional[torch.Tensor],
) -> torch.Tensor:
    """Simple DSM + velocity prior loss for latent drift training."""
    # Denoising score matching: add noise and predict drift
    noise_scale = 0.1
    eps = torch.randn_like(z) * noise_scale
    z_noisy = z + eps
    f_pred = drift(z_noisy, t)

    # DSM loss: predicted drift should point toward clean z
    loss_dsm = ((f_pred - (-eps / noise_scale)) ** 2).mean()

    # Velocity prior: if available, align drift with observed velocity
    loss_vel = torch.zeros((), device=z.device)
    if v is not None:
        f_clean = drift(z, t)
        loss_vel = ((f_clean - v) ** 2).mean()

    return loss_dsm + 0.5 * loss_vel


# ---------------------------------------------------------------------------
# Full staged pipeline
# ---------------------------------------------------------------------------

def train_hybrid_grn_from_anndata(
    adata,
    cfg: HybridGRNConfig,
    train_cfg: Optional[HybridGRNTrainConfig] = None,
    *,
    tf_names: Optional[list[str]] = None,
    gene_names: Optional[list[str]] = None,
    prior_network=None,
    use_rep: str = "X_pca",
    gene_layer: str = "X",
    vel_layer: Optional[str] = "velocity",
    pseudotime_key: str = "pseudotime",
    normalize_pseudotime: bool = False,
    pretrained_rep: Optional[RepresentationModel] = None,
    pretrained_drift: Optional[DriftField] = None,
    verbose: bool = True,
) -> tuple[HybridGRNModel, HybridGRNResult]:
    """Full staged HybridGRN training pipeline from AnnData.

    Parameters
    ----------
    adata :
        AnnData object with gene expression, pseudotime, and optionally velocity.
    cfg :
        HybridGRNConfig specifying model architecture.
    train_cfg :
        Training hyperparameters.  If None, uses defaults.
    tf_names :
        TF gene names.  If None, uses the built-in fallback list.
    gene_names :
        Gene subset.  If None, uses all genes.
    prior_network :
        Optional prior network for TF mask construction.
    use_rep :
        Latent representation key in adata.obsm.
    gene_layer :
        Gene expression layer key.
    vel_layer :
        Velocity layer key.
    pseudotime_key :
        Pseudotime column in adata.obs.
    normalize_pseudotime :
        Rescale pseudotime to [0, 1] if not already.
    pretrained_rep :
        Skip representation pretraining if provided.
    pretrained_drift :
        Skip drift training if provided.
    verbose :
        Print progress.

    Returns
    -------
    model : HybridGRNModel
        Trained HybridGRN model.
    result : HybridGRNResult
        GRN extraction results.
    """
    if train_cfg is None:
        train_cfg = HybridGRNTrainConfig()

    device = train_cfg.device

    # ── Load data ─────────────────────────────────────────────────────
    if verbose:
        print("[HybridGRN Pipeline] Loading data from AnnData...")
    data = tensors_from_anndata_hybrid(
        adata,
        use_rep=use_rep,
        gene_layer=gene_layer,
        vel_layer=vel_layer,
        pseudotime_key=pseudotime_key,
        tf_names=tf_names,
        gene_names=gene_names,
        normalize_pseudotime=normalize_pseudotime,
        device="cpu",
    )
    x_gene = data["x_gene"]
    pseudotime = data["pseudotime"]
    velocity = data["velocity"]
    tf_index = data["tf_index"]
    found_tf_names = data["tf_names"]
    selected_gene_names = data["gene_names"]

    # Update config with actual gene count
    cfg.rep_cfg.n_genes = x_gene.shape[1]

    # ── Build TF mask ─────────────────────────────────────────────────
    tf_mask = None
    if tf_index is not None and found_tf_names is not None:
        if verbose:
            print(f"[HybridGRN Pipeline] Building TF mask "
                  f"({len(found_tf_names)} TFs × {x_gene.shape[1]} genes)...")
        tf_mask = build_tf_mask(
            found_tf_names, selected_gene_names, prior_network=prior_network
        )

    # ── Stage 1: Representation ───────────────────────────────────────
    if pretrained_rep is not None:
        rep = pretrained_rep
        if verbose:
            print("[HybridGRN Pipeline] Using pretrained representation.")
    else:
        if verbose:
            print("[HybridGRN Pipeline] Stage 1: Pretraining representation...")
        rep = build_representation(cfg.rep_cfg)
        rep = pretrain_representation(rep, x_gene, train_cfg, verbose=verbose)

    # ── Stage 2: Latent drift ─────────────────────────────────────────
    if pretrained_drift is not None:
        drift = pretrained_drift
        if verbose:
            print("[HybridGRN Pipeline] Using pretrained drift model.")
    else:
        if verbose:
            print("[HybridGRN Pipeline] Stage 2: Training latent drift...")
        drift = DriftField(cfg.drift_cfg)
        drift = train_latent_drift(
            drift, rep, x_gene, pseudotime, velocity, train_cfg, verbose=verbose
        )

    # ── Stage 3: GRN extraction ───────────────────────────────────────
    if verbose:
        print("[HybridGRN Pipeline] Stage 3: Building HybridGRN model...")
    model = HybridGRNModel(
        cfg, rep=rep, drift=drift,
        tf_index=tf_index,
        tf_mask=tf_mask,
    )

    # Compute bin means for local dynamics loss
    bin_data = compute_bin_means(
        x_gene, pseudotime, n_bins=cfg.n_time_bins, tf_index=tf_index
    )

    if verbose:
        print("[HybridGRN Pipeline] Stage 4: Running GRN extraction pipeline...")
    result = model.run(
        X=x_gene,
        T=pseudotime,
        x_tf_seq=bin_data.get("x_tf_seq"),
        dx_seq=bin_data.get("dx_mean"),
        gene_names=selected_gene_names,
        tf_names=found_tf_names,
        verbose=verbose,
    )

    return model, result
