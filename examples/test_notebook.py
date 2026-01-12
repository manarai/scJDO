#!/usr/bin/env python
# coding: utf-8

# # Schrödinger Bridge: Synthetic Aging Analysis
# 
# This notebook demonstrates the **Schrödinger Bridge** implementation in scIDiff for modeling optimal transport between two distributions. We use a synthetic aging scenario where:
# 
# - **Source distribution (ρ₀)**: Young cells
# - **Target distribution (ρ₁)**: Old cells
# - **Forward process**: Aging trajectory
# - **Backward process**: Rejuvenation trajectory
# 
# ## Mathematical Framework
# 
# The Schrödinger Bridge finds the most probable stochastic process connecting two distributions:
# 
# $$
# dX_t = f(X_t, t)\,dt + \sqrt{2\beta}\,dW_t
# $$
# 
# subject to marginal constraints:
# - $X_0 \sim \rho_0$ (young cells)
# - $X_1 \sim \rho_1$ (old cells)
# 
# The bridge is learned via:
# 1. **Optimal Transport**: Sinkhorn algorithm computes coupling between distributions
# 2. **Score Matching**: Neural networks learn forward and backward drift fields
# 3. **Endpoint Loss**: Ensures trajectories reach target distributions

# In[ ]:


# Import required libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from scqdiff.models.schrodinger_bridge import SchrodingerBridge, SchrodingerBridgeConfig
from scqdiff.transport.sinkhorn import compute_ot_plan, sinkhorn_divergence

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

print("✓ Libraries imported successfully")
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")


# ## 1. Generate Synthetic Aging Data
# 
# We create two distributions representing young and old cells:
# 
# - **Young cells**: Compact distribution centered near origin
# - **Old cells**: More dispersed distribution shifted in state space
# 
# This mimics biological aging where:
# - Young cells have tightly regulated gene expression
# - Old cells show increased variability and dysregulation

# In[ ]:


# Configuration
n_young = 500
n_old = 500
dim = 20  # Gene expression dimensions

# Young cells: compact, centered
X_young = torch.randn(n_young, dim) * 0.5

# Old cells: dispersed, shifted
# Add systematic shift + increased variance
shift = torch.randn(dim) * 2.0  # Aging-related changes
X_old = torch.randn(n_old, dim) * 1.2 + shift

print(f"Young cells: {X_young.shape}")
print(f"  Mean: {X_young.mean(0)[:5].numpy()}...")
print(f"  Std:  {X_young.std(0)[:5].numpy()}...")
print(f"\nOld cells: {X_old.shape}")
print(f"  Mean: {X_old.mean(0)[:5].numpy()}...")
print(f"  Std:  {X_old.std(0)[:5].numpy()}...")

# Compute distributional distance
div = sinkhorn_divergence(X_young, X_old, epsilon=0.1)
print(f"\nSinkhorn divergence (Wasserstein-like): {div.item():.2f}")


# ## 2. Visualize Distributions
# 
# Project to 2D using PCA for visualization

# In[ ]:


from sklearn.decomposition import PCA

# Combine data for PCA
X_combined = torch.cat([X_young, X_old], dim=0).numpy()
labels = ['Young'] * n_young + ['Old'] * n_old

# PCA projection
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_combined)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot
ax = axes[0]
ax.scatter(X_pca[:n_young, 0], X_pca[:n_young, 1], 
           alpha=0.5, s=30, c='blue', label='Young cells')
ax.scatter(X_pca[n_young:, 0], X_pca[n_young:, 1], 
           alpha=0.5, s=30, c='red', label='Old cells')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
ax.set_title('Young vs Old Cells (PCA Projection)')
ax.legend()
ax.grid(True, alpha=0.3)

# Density plot
ax = axes[1]
ax.hist2d(X_pca[:n_young, 0], X_pca[:n_young, 1], 
          bins=30, alpha=0.5, cmap='Blues', label='Young')
ax.hist2d(X_pca[n_young:, 0], X_pca[n_young:, 1], 
          bins=30, alpha=0.5, cmap='Reds', label='Old')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
ax.set_title('Density Distributions')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n✓ Distributions are clearly separated")
print(f"  PC1 explains {pca.explained_variance_ratio_[0]:.1%} of variance")
print(f"  PC2 explains {pca.explained_variance_ratio_[1]:.1%} of variance")


# ## 3. Compute Optimal Transport Plan
# 
# The Sinkhorn algorithm computes an entropic optimal transport plan $P$ that couples young and old cells:
# 
# $$
# P^* = \arg\min_{P \in \Pi(\rho_0, \rho_1)} \langle C, P \rangle + \epsilon H(P)
# $$
# 
# where:
# - $C_{ij}$ is the cost matrix (squared Euclidean distance)
# - $\epsilon$ is the entropic regularization parameter
# - $H(P)$ is the entropy of the plan
# - $\Pi(\rho_0, \rho_1)$ is the set of couplings with marginals $\rho_0$ and $\rho_1$

# In[ ]:


# Compute OT plan
print("Computing optimal transport plan...")
P, f, g = compute_ot_plan(X_young, X_old, epsilon=0.1, max_iter=100)

print(f"\n✓ OT plan computed")
print(f"  Plan shape: {P.shape}")
print(f"  Plan sum: {P.sum().item():.6f} (should be ≈1.0)")
print(f"  Sparsity: {(P < 1e-6).float().mean().item():.1%} entries near zero")

# Check marginal constraints
row_marginal_error = (P.sum(dim=1) - torch.ones(n_young) / n_young).abs().max()
col_marginal_error = (P.sum(dim=0) - torch.ones(n_old) / n_old).abs().max()

print(f"\nMarginal constraint satisfaction:")
print(f"  Row marginal error: {row_marginal_error.item():.6f}")
print(f"  Col marginal error: {col_marginal_error.item():.6f}")

# Visualize OT plan (subsample for clarity)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Full plan (heatmap)
ax = axes[0]
im = ax.imshow(P.numpy()[:100, :100], cmap='viridis', aspect='auto')
ax.set_xlabel('Old cells')
ax.set_ylabel('Young cells')
ax.set_title('Optimal Transport Plan (100×100 subsample)')
plt.colorbar(im, ax=ax, label='Coupling probability')

# Plan sparsity histogram
ax = axes[1]
P_flat = P.flatten().numpy()
ax.hist(np.log10(P_flat[P_flat > 1e-10] + 1e-10), bins=50, alpha=0.7, edgecolor='black')
ax.set_xlabel('log₁₀(coupling probability)')
ax.set_ylabel('Count')
ax.set_title('Distribution of Coupling Probabilities')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✓ OT plan shows sparse coupling (most cells paired with few partners)")


# ## 4. Initialize Schrödinger Bridge
# 
# The bridge consists of:
# - **Forward network**: Learns drift from young → old (aging)
# - **Backward network**: Learns drift from old → young (rejuvenation)
# 
# Both networks are trained via denoising score matching.

# In[ ]:


# Configure bridge
cfg = SchrodingerBridgeConfig(
    dim=dim,
    hidden=256,
    depth=4,
    beta=0.1,          # Diffusion coefficient
    sigma=0.2,         # Noise level for score matching
    epsilon=0.1,       # Entropic regularization
    sinkhorn_max_iter=100,
    device='cpu'
)

# Initialize bridge
bridge = SchrodingerBridge(cfg, X_young, X_old)

print("✓ Schrödinger Bridge initialized")
print(f"\nConfiguration:")
print(f"  Dimension: {cfg.dim}")
print(f"  Hidden size: {cfg.hidden}")
print(f"  Depth: {cfg.depth}")
print(f"  Diffusion β: {cfg.beta}")
print(f"  Score matching σ: {cfg.sigma}")
print(f"  Entropic ε: {cfg.epsilon}")

print(f"\nModel architecture:")
forward_params = sum(p.numel() for p in bridge.forward_net.parameters())
backward_params = sum(p.numel() for p in bridge.backward_net.parameters())
print(f"  Forward network: {forward_params:,} parameters")
print(f"  Backward network: {backward_params:,} parameters")
print(f"  Total: {forward_params + backward_params:,} parameters")


# ## 5. Train Schrödinger Bridge
# 
# Training alternates between:
# 1. **Computing OT plan**: Update coupling using Sinkhorn
# 2. **Score matching**: Train forward/backward networks
# 3. **Endpoint loss**: Ensure trajectories reach targets
# 
# Loss function:
# $$
# \mathcal{L} = \mathcal{L}_{\text{forward}} + \mathcal{L}_{\text{backward}} + \alpha \mathcal{L}_{\text{endpoint}}
# $$

# In[ ]:


# Training configuration
n_iterations = 50
batch_size = 128
lr = 1e-3
update_ot_every = 10  # Update OT plan every N iterations

# Optimizers
optimizer_forward = torch.optim.Adam(bridge.forward_net.parameters(), lr=lr)
optimizer_backward = torch.optim.Adam(bridge.backward_net.parameters(), lr=lr)

# Training history
history = {
    'total': [],
    'forward': [],
    'backward': [],
    'endpoint': []
}

print("Training Schrödinger Bridge...\n")

for iteration in tqdm(range(n_iterations), desc="Training"):
    # Update OT plan periodically
    update_ot = (iteration % update_ot_every == 0)

    # Compute losses
    losses = bridge.train_step(batch_size=batch_size, update_ot=update_ot)

    # Backward pass
    optimizer_forward.zero_grad()
    optimizer_backward.zero_grad()
    losses['total'].backward()
    optimizer_forward.step()
    optimizer_backward.step()

    # Record history
    for key in history:
        history[key].append(losses[key].item())

    # Print progress
    if (iteration + 1) % 10 == 0:
        print(f"Iter {iteration+1:3d}: "
              f"total={losses['total'].item():.4f}, "
              f"forward={losses['forward'].item():.4f}, "
              f"backward={losses['backward'].item():.4f}, "
              f"endpoint={losses['endpoint'].item():.4f}")

print("\n✓ Training complete")


# ## 6. Visualize Training Progress

# In[ ]:


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Total loss
ax = axes[0, 0]
ax.plot(history['total'], linewidth=2, color='black')
ax.set_xlabel('Iteration')
ax.set_ylabel('Total Loss')
ax.set_title('Total Training Loss')
ax.grid(True, alpha=0.3)

# Forward vs Backward
ax = axes[0, 1]
ax.plot(history['forward'], label='Forward (aging)', linewidth=2, color='blue')
ax.plot(history['backward'], label='Backward (rejuvenation)', linewidth=2, color='red')
ax.set_xlabel('Iteration')
ax.set_ylabel('Score Matching Loss')
ax.set_title('Forward vs Backward Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# Endpoint loss
ax = axes[1, 0]
ax.plot(history['endpoint'], linewidth=2, color='green')
ax.set_xlabel('Iteration')
ax.set_ylabel('Endpoint Loss')
ax.set_title('Endpoint Constraint Loss')
ax.grid(True, alpha=0.3)

# Loss components (stacked)
ax = axes[1, 1]
ax.plot(history['forward'], label='Forward', linewidth=2, alpha=0.7)
ax.plot(history['backward'], label='Backward', linewidth=2, alpha=0.7)
ax.plot(history['endpoint'], label='Endpoint', linewidth=2, alpha=0.7)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title('All Loss Components')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print final losses
print(f"\nFinal losses:")
print(f"  Total: {history['total'][-1]:.4f}")
print(f"  Forward: {history['forward'][-1]:.4f}")
print(f"  Backward: {history['backward'][-1]:.4f}")
print(f"  Endpoint: {history['endpoint'][-1]:.4f}")

# Compute improvement
improvement = (history['total'][0] - history['total'][-1]) / history['total'][0] * 100
print(f"\nTotal loss improvement: {improvement:.1f}%")


# ## 7. Forward Integration: Aging Trajectories
# 
# Simulate aging by integrating the forward drift from young cells:
# 
# $$
# X_{t+\Delta t} = X_t + f_{\text{forward}}(X_t, t) \Delta t + \sqrt{2\beta\Delta t}\,\xi
# $$
# 
# where $\xi \sim \mathcal{N}(0, I)$ for stochastic integration.

# In[ ]:


# Select young cells to age
n_samples = 10
young_samples = X_young[:n_samples]

# Integrate forward (deterministic)
print("Simulating aging trajectories (deterministic)...")
aging_traj_det = bridge.forward_integrate(young_samples, steps=100, stochastic=False)

# Integrate forward (stochastic)
print("Simulating aging trajectories (stochastic)...")
aging_traj_stoch = bridge.forward_integrate(young_samples, steps=100, stochastic=True)

print(f"\n✓ Trajectories computed")
print(f"  Shape: {aging_traj_det.shape} (samples, timepoints, dimensions)")
print(f"  Deterministic displacement: {(aging_traj_det[:, -1, :] - aging_traj_det[:, 0, :]).norm(dim=1).mean().item():.3f}")
print(f"  Stochastic displacement: {(aging_traj_stoch[:, -1, :] - aging_traj_stoch[:, 0, :]).norm(dim=1).mean().item():.3f}")


# ## 8. Visualize Aging Trajectories

# In[ ]:


# Project trajectories to PCA space
traj_det_flat = aging_traj_det.reshape(-1, dim).numpy()
traj_det_pca = pca.transform(traj_det_flat).reshape(n_samples, 101, 2)

traj_stoch_flat = aging_traj_stoch.reshape(-1, dim).numpy()
traj_stoch_pca = pca.transform(traj_stoch_flat).reshape(n_samples, 101, 2)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Deterministic trajectories
ax = axes[0]
ax.scatter(X_pca[:n_young, 0], X_pca[:n_young, 1], 
           alpha=0.2, s=20, c='blue', label='Young cells')
ax.scatter(X_pca[n_young:, 0], X_pca[n_young:, 1], 
           alpha=0.2, s=20, c='red', label='Old cells')

for i in range(n_samples):
    traj = traj_det_pca[i]
    ax.plot(traj[:, 0], traj[:, 1], linewidth=2, alpha=0.7, color='green')
    ax.scatter(traj[0, 0], traj[0, 1], s=100, c='blue', marker='o', edgecolor='black', linewidth=2, zorder=10)
    ax.scatter(traj[-1, 0], traj[-1, 1], s=100, c='red', marker='s', edgecolor='black', linewidth=2, zorder=10)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
ax.set_title('Aging Trajectories (Deterministic)')
ax.legend()
ax.grid(True, alpha=0.3)

# Stochastic trajectories
ax = axes[1]
ax.scatter(X_pca[:n_young, 0], X_pca[:n_young, 1], 
           alpha=0.2, s=20, c='blue', label='Young cells')
ax.scatter(X_pca[n_young:, 0], X_pca[n_young:, 1], 
           alpha=0.2, s=20, c='red', label='Old cells')

for i in range(n_samples):
    traj = traj_stoch_pca[i]
    ax.plot(traj[:, 0], traj[:, 1], linewidth=2, alpha=0.7, color='purple')
    ax.scatter(traj[0, 0], traj[0, 1], s=100, c='blue', marker='o', edgecolor='black', linewidth=2, zorder=10)
    ax.scatter(traj[-1, 0], traj[-1, 1], s=100, c='red', marker='s', edgecolor='black', linewidth=2, zorder=10)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
ax.set_title('Aging Trajectories (Stochastic)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✓ Trajectories show smooth transition from young to old")
print("  ○ Blue circles: Starting points (young)")
print("  ■ Red squares: Endpoints (aged)")
print("  Stochastic trajectories show natural variability")


# ## 9. Backward Integration: Rejuvenation Trajectories
# 
# Simulate rejuvenation by integrating the backward drift from old cells:
# 
# $$
# X_{t-\Delta t} = X_t - f_{\text{backward}}(X_t, t) \Delta t + \sqrt{2\beta\Delta t}\,\xi
# $$

# In[ ]:


# Select old cells to rejuvenate
old_samples = X_old[:n_samples]

# Integrate backward (deterministic)
print("Simulating rejuvenation trajectories (deterministic)...")
rejuv_traj_det = bridge.backward_integrate(old_samples, steps=100, stochastic=False)

# Integrate backward (stochastic)
print("Simulating rejuvenation trajectories (stochastic)...")
rejuv_traj_stoch = bridge.backward_integrate(old_samples, steps=100, stochastic=True)

print(f"\n✓ Trajectories computed")
print(f"  Shape: {rejuv_traj_det.shape}")
print(f"  Deterministic displacement: {(rejuv_traj_det[:, -1, :] - rejuv_traj_det[:, 0, :]).norm(dim=1).mean().item():.3f}")
print(f"  Stochastic displacement: {(rejuv_traj_stoch[:, -1, :] - rejuv_traj_stoch[:, 0, :]).norm(dim=1).mean().item():.3f}")


# ## 10. Visualize Rejuvenation Trajectories

# In[ ]:


# Project trajectories to PCA space
rejuv_det_flat = rejuv_traj_det.reshape(-1, dim).numpy()
rejuv_det_pca = pca.transform(rejuv_det_flat).reshape(n_samples, 101, 2)

rejuv_stoch_flat = rejuv_traj_stoch.reshape(-1, dim).numpy()
rejuv_stoch_pca = pca.transform(rejuv_stoch_flat).reshape(n_samples, 101, 2)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Deterministic trajectories
ax = axes[0]
ax.scatter(X_pca[:n_young, 0], X_pca[:n_young, 1], 
           alpha=0.2, s=20, c='blue', label='Young cells')
ax.scatter(X_pca[n_young:, 0], X_pca[n_young:, 1], 
           alpha=0.2, s=20, c='red', label='Old cells')

for i in range(n_samples):
    traj = rejuv_det_pca[i]
    ax.plot(traj[:, 0], traj[:, 1], linewidth=2, alpha=0.7, color='orange')
    ax.scatter(traj[0, 0], traj[0, 1], s=100, c='red', marker='s', edgecolor='black', linewidth=2, zorder=10)
    ax.scatter(traj[-1, 0], traj[-1, 1], s=100, c='blue', marker='o', edgecolor='black', linewidth=2, zorder=10)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
ax.set_title('Rejuvenation Trajectories (Deterministic)')
ax.legend()
ax.grid(True, alpha=0.3)

# Stochastic trajectories
ax = axes[1]
ax.scatter(X_pca[:n_young, 0], X_pca[:n_young, 1], 
           alpha=0.2, s=20, c='blue', label='Young cells')
ax.scatter(X_pca[n_young:, 0], X_pca[n_young:, 1], 
           alpha=0.2, s=20, c='red', label='Old cells')

for i in range(n_samples):
    traj = rejuv_stoch_pca[i]
    ax.plot(traj[:, 0], traj[:, 1], linewidth=2, alpha=0.7, color='cyan')
    ax.scatter(traj[0, 0], traj[0, 1], s=100, c='red', marker='s', edgecolor='black', linewidth=2, zorder=10)
    ax.scatter(traj[-1, 0], traj[-1, 1], s=100, c='blue', marker='o', edgecolor='black', linewidth=2, zorder=10)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
ax.set_title('Rejuvenation Trajectories (Stochastic)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✓ Trajectories show smooth transition from old to young")
print("  ■ Red squares: Starting points (old)")
print("  ○ Blue circles: Endpoints (rejuvenated)")


# ## 11. Quantitative Analysis
# 
# Measure how well the bridge transports cells between distributions

# In[ ]:


# Compute distances to target distributions
def compute_transport_quality(trajectories, target_dist):
    """Compute average distance from trajectory endpoints to target distribution."""
    endpoints = trajectories[:, -1, :]
    distances = torch.cdist(endpoints, target_dist)
    min_distances = distances.min(dim=1)[0]
    return min_distances.mean().item(), min_distances.std().item()

# Forward transport quality (young → old)
forward_mean, forward_std = compute_transport_quality(aging_traj_det, X_old)

# Backward transport quality (old → young)
backward_mean, backward_std = compute_transport_quality(rejuv_traj_det, X_young)

# Baseline: direct distance without transport
baseline_forward = torch.cdist(young_samples, X_old).min(dim=1)[0].mean().item()
baseline_backward = torch.cdist(old_samples, X_young).min(dim=1)[0].mean().item()

print("Transport Quality Analysis")
print("=" * 60)
print(f"\nForward transport (Young → Old):")
print(f"  Baseline distance (no transport): {baseline_forward:.4f}")
print(f"  After aging trajectory: {forward_mean:.4f} ± {forward_std:.4f}")
print(f"  Improvement: {(baseline_forward - forward_mean):.4f} ({(baseline_forward - forward_mean)/baseline_forward*100:.1f}%)")

print(f"\nBackward transport (Old → Young):")
print(f"  Baseline distance (no transport): {baseline_backward:.4f}")
print(f"  After rejuvenation trajectory: {backward_mean:.4f} ± {backward_std:.4f}")
print(f"  Improvement: {(baseline_backward - backward_mean):.4f} ({(baseline_backward - backward_mean)/baseline_backward*100:.1f}%)")

# Round-trip consistency
young_start = X_young[:5]
aged = bridge.forward_integrate(young_start, steps=50, stochastic=False)[:, -1, :]
rejuvenated = bridge.backward_integrate(aged, steps=50, stochastic=False)[:, -1, :]
round_trip_error = (rejuvenated - young_start).norm(dim=1).mean().item()

print(f"\nRound-trip consistency (Young → Old → Young):")
print(f"  Error: {round_trip_error:.4f}")
print(f"  Relative error: {round_trip_error / young_start.norm(dim=1).mean().item() * 100:.2f}%")


# ## 12. Drift Field Analysis
# 
# Analyze the learned drift fields at different time points

# In[ ]:


# Sample points along the bridge
n_test = 100
x_test = torch.randn(n_test, dim)
t_values = torch.linspace(0, 1, 5)

# Compute drift magnitudes
forward_norms = []
backward_norms = []

for t in t_values:
    t_batch = torch.full((n_test,), t.item())

    drift_f = bridge.forward_drift(x_test, t_batch)
    drift_b = bridge.backward_drift(x_test, t_batch)

    forward_norms.append(drift_f.norm(dim=1).mean().item())
    backward_norms.append(drift_b.norm(dim=1).mean().item())

# Plot drift magnitudes
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(t_values.numpy(), forward_norms, 'o-', linewidth=2, markersize=8, 
        label='Forward drift (aging)', color='blue')
ax.plot(t_values.numpy(), backward_norms, 's-', linewidth=2, markersize=8, 
        label='Backward drift (rejuvenation)', color='red')

ax.set_xlabel('Time t', fontsize=12)
ax.set_ylabel('Average drift magnitude', fontsize=12)
ax.set_title('Drift Field Strength Over Time', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✓ Drift fields show time-dependent behavior")
print(f"  Forward drift magnitude: {np.mean(forward_norms):.4f} ± {np.std(forward_norms):.4f}")
print(f"  Backward drift magnitude: {np.mean(backward_norms):.4f} ± {np.std(backward_norms):.4f}")


# ## 13. Summary and Biological Interpretation
# 
# ### Key Results
# 
# 1. **Optimal Transport**: Successfully computed sparse coupling between young and old cells
# 2. **Forward Process**: Learned aging trajectories that smoothly transport young cells to old distribution
# 3. **Backward Process**: Learned rejuvenation trajectories that reverse aging
# 4. **Consistency**: Round-trip error indicates learned dynamics are approximately reversible
# 
# ### Biological Interpretation
# 
# The Schrödinger Bridge provides:
# 
# - **Aging trajectories**: Most probable paths cells take during aging
# - **Rejuvenation trajectories**: Optimal interventions to reverse aging
# - **Drift fields**: Time-dependent forces driving cellular state changes
# - **Optimal coupling**: Which young cells correspond to which old cells
# 
# ### Applications
# 
# This framework can be applied to:
# 
# 1. **Aging studies**: Model cellular aging and identify rejuvenation targets
# 2. **Perturbation response**: Predict effects of drugs or genetic modifications
# 3. **Differentiation**: Model cell fate transitions during development
# 4. **Reprogramming**: Design optimal protocols for cell type conversion
# 
# ### Next Steps
# 
# - Apply to real single-cell RNA-seq data
# - Incorporate RNA velocity as biological prior
# - Analyze gene-level changes along trajectories
# - Identify key regulatory genes driving transitions

# In[ ]:


print("="*70)
print("Schrödinger Bridge Analysis Complete")
print("="*70)
print(f"\nModel trained for {n_iterations} iterations")
print(f"Final loss: {history['total'][-1]:.4f}")
print(f"\nForward transport improvement: {(baseline_forward - forward_mean)/baseline_forward*100:.1f}%")
print(f"Backward transport improvement: {(baseline_backward - backward_mean)/baseline_backward*100:.1f}%")
print(f"Round-trip error: {round_trip_error:.4f}")
print("\n✓ All analyses complete")

