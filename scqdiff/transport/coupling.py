"""
Utilities for working with OT couplings.
"""

import torch


def sample_from_coupling(X, Y, P, n_samples=None):
    """
    Sample from optimal transport coupling.
    
    Given transport plan P, sample pairs (x, y) according to P.
    
    Args:
        X: Source samples (N, D)
        Y: Target samples (M, D)
        P: Transport plan (N, M)
        n_samples: Number of samples (default: N)
    
    Returns:
        X_sampled: Sampled source points (n_samples, D)
        Y_sampled: Corresponding target points (n_samples, D)
    """
    N, M = P.shape
    if n_samples is None:
        n_samples = N
    
    # Flatten P and sample indices
    P_flat = P.flatten()
    indices = torch.multinomial(P_flat, n_samples, replacement=True)
    
    # Convert flat indices to (i, j) pairs
    i_indices = indices // M
    j_indices = indices % M
    
    # Sample corresponding points
    X_sampled = X[i_indices]
    Y_sampled = Y[j_indices]
    
    return X_sampled, Y_sampled


def coupling_loss(X, Y, P):
    """
    Compute loss that encourages following the OT coupling.
    
    This can be used to guide the learned drift to follow optimal transport.
    
    Args:
        X: Source samples (N, D)
        Y: Target samples (M, D)
        P: Transport plan (N, M)
    
    Returns:
        loss: Scalar coupling loss
    """
    # Expected target for each source point
    Y_expected = P @ Y  # (N, D)
    
    # Loss: encourage X to move toward Y_expected
    loss = torch.norm(Y_expected - X, dim=1).mean()
    
    return loss


def interpolate_along_coupling(X, Y, P, t):
    """
    Interpolate between X and Y along OT coupling at time t.
    
    Args:
        X: Source samples (N, D)
        Y: Target samples (M, D)
        P: Transport plan (N, M)
        t: Time in [0, 1] (scalar or (N,))
    
    Returns:
        X_t: Interpolated samples (N, D)
    """
    # Expected target for each source point
    Y_expected = P @ Y  # (N, D)
    
    # Linear interpolation
    if isinstance(t, float):
        t = torch.full((X.shape[0], 1), t, device=X.device)
    else:
        t = t.view(-1, 1)
    
    X_t = (1 - t) * X + t * Y_expected
    
    return X_t
