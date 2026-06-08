"""
Operator Metrics Module

Computes operator-derived state metrics from Jacobian eigenvalues.
These metrics define the dynamical properties of cellular states.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple


class OperatorMetrics:
    """
    Compute operator-derived state metrics from Jacobian eigenvalues.
    
    Given a drift field f(x,t), this class computes the Jacobian J(x,t) = ∂f/∂x
    and extracts four key metrics from its eigenvalue spectrum:
    
    1. λ_max⁺: Max unstable eigenvalue (detects bifurcations)
    2. λ_min⁻: Stability depth (commitment depth)
    3. P: Plasticity index (fraction of near-neutral modes)
    4. S: Stable subspace dimension (buffering capacity)
    
    Args:
        drift_model: Trained DriftField model
        epsilon: Threshold for near-neutral modes (default: 0.1)
        device: Device for computation (default: "cpu")
    """
    
    def __init__(
        self,
        drift_model,
        epsilon: float = 0.1,
        device: str = "cpu"
    ):
        self.drift_model = drift_model
        self.epsilon = epsilon
        self.device = device
        self.drift_model.to(device)
        self.drift_model.eval()
    
    def compute_jacobian(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Jacobian matrix ∂f/∂x at (x,t).
        
        Args:
            x: State tensor (batch_size, dim)
            t: Time tensor (batch_size,)
            
        Returns:
            Jacobian tensor (batch_size, dim, dim)
        """
        x = x.to(self.device).requires_grad_(True)
        t = t.to(self.device)
        
        batch_size, dim = x.shape
        jacobians = []
        
        for i in range(batch_size):
            xi = x[i:i+1]
            ti = t[i:i+1]
            
            # Compute drift at this point
            drift = self.drift_model(xi, ti)  # (1, dim)
            
            # Compute Jacobian row by row
            jac_rows = []
            for j in range(dim):
                # Compute gradient of j-th output w.r.t. input
                grad_outputs = torch.zeros_like(drift)
                grad_outputs[0, j] = 1.0
                
                grad = torch.autograd.grad(
                    outputs=drift,
                    inputs=xi,
                    grad_outputs=grad_outputs,
                    retain_graph=True,
                    create_graph=False
                )[0]  # (1, dim)
                
                jac_rows.append(grad)
            
            jac = torch.cat(jac_rows, dim=0)  # (dim, dim)
            jacobians.append(jac)
        
        return torch.stack(jacobians, dim=0)  # (batch_size, dim, dim)
    
    def compute_eigenvalues(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute eigenvalues of J(x,t).
        
        Args:
            x: State tensor (batch_size, dim)
            t: Time tensor (batch_size,)
            
        Returns:
            Eigenvalues (batch_size, dim) - complex-valued
        """
        jacobians = self.compute_jacobian(x, t)
        eigenvalues = torch.linalg.eigvals(jacobians)
        return eigenvalues
    
    def max_unstable_eigenvalue(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute λ_max⁺ = max(Re(λᵢ)).
        
        Detects bifurcation points and control sensitivity.
        Positive values indicate unstable directions.
        
        Args:
            x: State tensor (batch_size, dim)
            t: Time tensor (batch_size,)
            
        Returns:
            Max unstable eigenvalue (batch_size,)
        """
        eigenvalues = self.compute_eigenvalues(x, t)
        real_parts = eigenvalues.real
        return real_parts.max(dim=1)[0]
    
    def stability_depth(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute λ_min⁻ = min(Re(λᵢ)).
        
        Measures how strongly deviations are damped.
        More negative values indicate deeper commitment.
        
        Args:
            x: State tensor (batch_size, dim)
            t: Time tensor (batch_size,)
            
        Returns:
            Stability depth (batch_size,)
        """
        eigenvalues = self.compute_eigenvalues(x, t)
        real_parts = eigenvalues.real
        return real_parts.min(dim=1)[0]
    
    def plasticity_index(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute P = #{|Re(λᵢ)| < ε} / d.
        
        Fraction of near-neutral modes.
        High values indicate many accessible directions.
        
        Args:
            x: State tensor (batch_size, dim)
            t: Time tensor (batch_size,)
            
        Returns:
            Plasticity index (batch_size,)
        """
        eigenvalues = self.compute_eigenvalues(x, t)
        real_parts = eigenvalues.real
        dim = real_parts.shape[1]
        
        near_neutral = (real_parts.abs() < self.epsilon).sum(dim=1).float()
        return near_neutral / dim
    
    def stable_subspace_dim(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute S = #{Re(λᵢ) < 0}.
        
        Number of stable directions (buffering capacity).
        Higher values indicate more robust states.
        
        Args:
            x: State tensor (batch_size, dim)
            t: Time tensor (batch_size,)
            
        Returns:
            Stable subspace dimension (batch_size,)
        """
        eigenvalues = self.compute_eigenvalues(x, t)
        real_parts = eigenvalues.real
        return (real_parts < 0).sum(dim=1).float()
    
    def compute_all_metrics(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_size: int = 32
    ) -> Dict[str, np.ndarray]:
        """
        Compute all operator metrics for a set of cells.
        
        Processes data in batches to avoid memory issues.
        
        Args:
            x: State tensor (n_cells, dim)
            t: Time tensor (n_cells,)
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with keys:
                - 'lambda_max_plus': Max unstable eigenvalue
                - 'lambda_min_minus': Stability depth
                - 'plasticity': Plasticity index
                - 'stable_dim': Stable subspace dimension
                - 'eigenvalues': Full eigenvalue spectra (optional)
        """
        n_cells = x.shape[0]
        
        # Initialize result arrays
        lambda_max_plus = []
        lambda_min_minus = []
        plasticity = []
        stable_dim = []
        all_eigenvalues = []
        
        # Process in batches
        with torch.no_grad():
            for i in range(0, n_cells, batch_size):
                batch_x = x[i:i+batch_size]
                batch_t = t[i:i+batch_size]
                
                # Compute eigenvalues once per batch
                eigenvalues = self.compute_eigenvalues(batch_x, batch_t)
                all_eigenvalues.append(eigenvalues.cpu())
                
                # Compute metrics from eigenvalues
                real_parts = eigenvalues.real
                
                # Max unstable eigenvalue
                lambda_max_plus.append(real_parts.max(dim=1)[0].cpu())
                
                # Stability depth
                lambda_min_minus.append(real_parts.min(dim=1)[0].cpu())
                
                # Plasticity index
                dim = real_parts.shape[1]
                near_neutral = (real_parts.abs() < self.epsilon).sum(dim=1).float()
                plasticity.append((near_neutral / dim).cpu())
                
                # Stable subspace dimension
                stable_dim.append((real_parts < 0).sum(dim=1).float().cpu())
        
        # Concatenate results
        metrics = {
            'lambda_max_plus': torch.cat(lambda_max_plus).numpy(),
            'lambda_min_minus': torch.cat(lambda_min_minus).numpy(),
            'plasticity': torch.cat(plasticity).numpy(),
            'stable_dim': torch.cat(stable_dim).numpy(),
            'eigenvalues': torch.cat(all_eigenvalues).numpy()
        }
        
        return metrics
    
    def compute_metrics_at_pseudotime(
        self,
        x: torch.Tensor,
        pseudotime: np.ndarray,
        batch_size: int = 32
    ) -> Dict[str, np.ndarray]:
        """
        Compute operator metrics using pseudotime as the time coordinate.
        
        Args:
            x: State tensor (n_cells, dim)
            pseudotime: Pseudotime values (n_cells,)
            batch_size: Batch size for processing
            
        Returns:
            Dictionary of operator metrics
        """
        # Normalize pseudotime to [0, 1]
        ptime_min = pseudotime.min()
        ptime_max = pseudotime.max()
        t_normalized = (pseudotime - ptime_min) / (ptime_max - ptime_min + 1e-8)
        
        t = torch.tensor(t_normalized, dtype=torch.float32)
        
        return self.compute_all_metrics(x, t, batch_size=batch_size)
