"""
Sinkhorn algorithm for entropic optimal transport.
"""

import torch
import torch.nn.functional as F


def sinkhorn_log(C, epsilon=0.1, max_iter=100, tol=1e-6):
    """
    Compute entropic optimal transport plan using log-domain Sinkhorn.
    
    Args:
        C: Cost matrix (N, M) - typically squared Euclidean distances
        epsilon: Entropic regularization parameter (smaller = closer to true OT)
        max_iter: Maximum number of Sinkhorn iterations
        tol: Convergence tolerance
    
    Returns:
        P: Transport plan (N, M) - doubly stochastic matrix
        f: Dual potential for source (N,)
        g: Dual potential for target (M,)
    """
    N, M = C.shape
    device = C.device
    
    # Uniform marginals
    a = torch.ones(N, device=device) / N
    b = torch.ones(M, device=device) / M
    
    # Initialize dual potentials
    f = torch.zeros(N, device=device)
    g = torch.zeros(M, device=device)
    
    # Log-domain Sinkhorn iterations
    for i in range(max_iter):
        f_prev = f.clone()
        
        # Update f
        # f = -epsilon * log(sum_j exp((f_i + g_j - C_ij) / epsilon))
        log_sum_exp_g = torch.logsumexp((g.unsqueeze(0) - C) / epsilon, dim=1)
        f = -epsilon * (log_sum_exp_g - torch.log(a))
        
        # Update g
        log_sum_exp_f = torch.logsumexp((f.unsqueeze(1) - C) / epsilon, dim=0)
        g = -epsilon * (log_sum_exp_f - torch.log(b))
        
        # Check convergence
        if torch.max(torch.abs(f - f_prev)) < tol:
            break
    
    # Compute transport plan
    P = torch.exp((f.unsqueeze(1) + g.unsqueeze(0) - C) / epsilon)
    
    return P, f, g


def compute_ot_plan(X, Y, epsilon=0.1, max_iter=100):
    """
    Compute optimal transport plan between two point clouds.
    
    Args:
        X: Source samples (N, D)
        Y: Target samples (M, D)
        epsilon: Entropic regularization
        max_iter: Maximum Sinkhorn iterations
    
    Returns:
        P: Transport plan (N, M)
        f: Dual potential for X
        g: Dual potential for Y
    """
    # Compute cost matrix (squared Euclidean distance)
    C = torch.cdist(X, Y, p=2) ** 2
    
    # Run Sinkhorn
    P, f, g = sinkhorn_log(C, epsilon=epsilon, max_iter=max_iter)
    
    return P, f, g


def sinkhorn_divergence(X, Y, epsilon=0.1):
    """
    Compute Sinkhorn divergence between two distributions.
    
    Sinkhorn divergence is a differentiable approximation to Wasserstein distance.
    
    Args:
        X: Source samples (N, D)
        Y: Target samples (M, D)
        epsilon: Entropic regularization
    
    Returns:
        divergence: Scalar Sinkhorn divergence
    """
    # OT(X, Y)
    C_xy = torch.cdist(X, Y, p=2) ** 2
    P_xy, _, _ = sinkhorn_log(C_xy, epsilon=epsilon)
    ot_xy = (P_xy * C_xy).sum()
    
    # OT(X, X)
    C_xx = torch.cdist(X, X, p=2) ** 2
    P_xx, _, _ = sinkhorn_log(C_xx, epsilon=epsilon)
    ot_xx = (P_xx * C_xx).sum()
    
    # OT(Y, Y)
    C_yy = torch.cdist(Y, Y, p=2) ** 2
    P_yy, _, _ = sinkhorn_log(C_yy, epsilon=epsilon)
    ot_yy = (P_yy * C_yy).sum()
    
    # Sinkhorn divergence
    divergence = ot_xy - 0.5 * (ot_xx + ot_yy)
    
    return divergence
