"""
Operator Embedding Module

Projects high-dimensional Jacobian operators into low-dimensional space
for visualization and clustering.

This module enables the full pipeline:
    genes → latent → J_ic → operator_embedding → clustering
"""

import numpy as np
import torch
from sklearn.decomposition import PCA
from typing import Optional, Literal, Dict
import warnings


class OperatorEmbedding:
    """
    Compute low-dimensional embeddings of operator space.
    
    The operator embedding provides a low-dimensional representation of the
    high-dimensional Jacobian operators, enabling visualization and clustering
    in operator space.
    
    Methods:
        - 'pca': PCA on flattened Jacobians
        - 'spectrum': Eigenvalue spectrum features
        - 'metrics': Use operator metrics directly (λ_max⁺, λ_min⁻, P, S)
    
    Args:
        method: Embedding method ('pca', 'spectrum', or 'metrics')
        
    Example:
        >>> embedder = OperatorEmbedding(method='pca')
        >>> embedding = embedder.fit_transform(jacobians, n_components=10)
        >>> adata.obsm['X_operator_embedding'] = embedding
    """
    
    def __init__(self, method: Literal['pca', 'spectrum', 'metrics'] = 'pca'):
        self.method = method
        self.model = None
        self.feature_names = None
    
    def fit_transform(
        self,
        jacobians: Optional[np.ndarray] = None,
        metrics: Optional[Dict[str, np.ndarray]] = None,
        n_components: int = 10,
        **kwargs
    ) -> np.ndarray:
        """
        Fit embedding model and transform Jacobians or metrics.
        
        Args:
            jacobians: (n_cells, dim, dim) Jacobian matrices (required for 'pca' and 'spectrum')
            metrics: Dictionary with operator metrics (required for 'metrics' method)
            n_components: Number of embedding dimensions
            **kwargs: Additional arguments for specific methods
            
        Returns:
            embedding: (n_cells, n_components) operator coordinates
            
        Raises:
            ValueError: If required inputs are not provided for the selected method
        """
        if self.method == 'pca':
            if jacobians is None:
                raise ValueError("Jacobians required for PCA embedding")
            return self._pca_embedding(jacobians, n_components)
        
        elif self.method == 'spectrum':
            if jacobians is None:
                raise ValueError("Jacobians required for spectrum embedding")
            return self._spectrum_embedding(jacobians, n_components)
        
        elif self.method == 'metrics':
            if metrics is None:
                raise ValueError("Metrics required for metrics embedding")
            return self._metrics_embedding(metrics, n_components)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _pca_embedding(self, jacobians: np.ndarray, n_components: int) -> np.ndarray:
        """
        PCA on flattened Jacobians.
        
        This method treats each Jacobian as a high-dimensional vector by flattening
        the (dim × dim) matrix into a (dim²,) vector, then applies PCA.
        
        Args:
            jacobians: (n_cells, dim, dim) Jacobian matrices
            n_components: Number of PCA components
            
        Returns:
            embedding: (n_cells, n_components) PCA coordinates
        """
        n_cells, dim, _ = jacobians.shape
        
        # Flatten Jacobians
        jac_flat = jacobians.reshape(n_cells, dim * dim)
        
        # Apply PCA
        self.model = PCA(n_components=n_components)
        embedding = self.model.fit_transform(jac_flat)
        
        # Store feature names
        self.feature_names = [f'PC{i+1}' for i in range(n_components)]
        
        print(f"PCA embedding: {n_components} components explain "
              f"{self.model.explained_variance_ratio_.sum()*100:.1f}% variance")
        
        return embedding
    
    def _spectrum_embedding(self, jacobians: np.ndarray, n_components: int) -> np.ndarray:
        """
        Embedding based on eigenvalue spectra.
        
        This method computes the eigenvalues of each Jacobian and uses them as
        features. Both real and imaginary parts are included.
        
        Args:
            jacobians: (n_cells, dim, dim) Jacobian matrices
            n_components: Number of PCA components for dimensionality reduction
            
        Returns:
            embedding: (n_cells, n_components) spectrum-based coordinates
        """
        n_cells, dim, _ = jacobians.shape
        
        # Compute eigenvalues for each Jacobian
        eigenvalues = []
        for i in range(n_cells):
            eigvals = np.linalg.eigvals(jacobians[i])
            # Sort by real part (descending)
            eigvals_sorted = eigvals[np.argsort(-eigvals.real)]
            eigenvalues.append(eigvals_sorted)
        
        eigenvalues = np.array(eigenvalues)  # (n_cells, dim)
        
        # Use real and imaginary parts as features
        features = np.column_stack([eigenvalues.real, eigenvalues.imag])
        
        # PCA on eigenvalue features
        self.model = PCA(n_components=n_components)
        embedding = self.model.fit_transform(features)
        
        # Store feature names
        self.feature_names = [f'Spectrum_PC{i+1}' for i in range(n_components)]
        
        print(f"Spectrum embedding: {n_components} components explain "
              f"{self.model.explained_variance_ratio_.sum()*100:.1f}% variance")
        
        return embedding
    
    def _metrics_embedding(
        self,
        metrics: Dict[str, np.ndarray],
        n_components: int
    ) -> np.ndarray:
        """
        Embedding based on operator metrics.
        
        This method uses the operator metrics (λ_max⁺, λ_min⁻, P, S) directly
        as features, optionally with PCA for dimensionality reduction.
        
        Args:
            metrics: Dictionary with keys 'lambda_max_plus', 'lambda_min_minus',
                    'plasticity', 'stable_dim'
            n_components: Number of components (if < 4, applies PCA; otherwise returns all 4)
            
        Returns:
            embedding: (n_cells, n_components) metrics-based coordinates
        """
        # Extract metrics
        metric_keys = ['lambda_max_plus', 'lambda_min_minus', 'plasticity', 'stable_dim']
        features = []
        
        for key in metric_keys:
            if key not in metrics:
                raise ValueError(f"Metric '{key}' not found in metrics dictionary")
            features.append(metrics[key])
        
        features = np.column_stack(features)  # (n_cells, 4)
        
        # If n_components >= 4, return all metrics
        if n_components >= 4:
            self.feature_names = metric_keys
            print(f"Metrics embedding: Using all 4 operator metrics")
            return features
        
        # Otherwise, apply PCA
        self.model = PCA(n_components=n_components)
        embedding = self.model.fit_transform(features)
        
        self.feature_names = [f'Metrics_PC{i+1}' for i in range(n_components)]
        
        print(f"Metrics embedding: {n_components} components explain "
              f"{self.model.explained_variance_ratio_.sum()*100:.1f}% variance")
        
        return embedding
    
    def transform(self, jacobians: Optional[np.ndarray] = None,
                 metrics: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """
        Transform new data using fitted embedding model.
        
        Args:
            jacobians: New Jacobian matrices (for 'pca' or 'spectrum' methods)
            metrics: New operator metrics (for 'metrics' method)
            
        Returns:
            embedding: Transformed coordinates
            
        Raises:
            RuntimeError: If model has not been fitted yet
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before transform. Use fit_transform first.")
        
        if self.method == 'pca':
            if jacobians is None:
                raise ValueError("Jacobians required for PCA transform")
            n_cells, dim, _ = jacobians.shape
            jac_flat = jacobians.reshape(n_cells, dim * dim)
            return self.model.transform(jac_flat)
        
        elif self.method == 'spectrum':
            if jacobians is None:
                raise ValueError("Jacobians required for spectrum transform")
            n_cells, dim, _ = jacobians.shape
            eigenvalues = []
            for i in range(n_cells):
                eigvals = np.linalg.eigvals(jacobians[i])
                eigvals_sorted = eigvals[np.argsort(-eigvals.real)]
                eigenvalues.append(eigvals_sorted)
            eigenvalues = np.array(eigenvalues)
            features = np.column_stack([eigenvalues.real, eigenvalues.imag])
            return self.model.transform(features)
        
        elif self.method == 'metrics':
            if metrics is None:
                raise ValueError("Metrics required for metrics transform")
            metric_keys = ['lambda_max_plus', 'lambda_min_minus', 'plasticity', 'stable_dim']
            features = np.column_stack([metrics[key] for key in metric_keys])
            
            if self.model is not None:
                return self.model.transform(features)
            else:
                return features
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def get_feature_names(self) -> list:
        """
        Get feature names for the embedding dimensions.
        
        Returns:
            List of feature names
        """
        if self.feature_names is None:
            warnings.warn("Feature names not available. Run fit_transform first.")
            return []
        return self.feature_names
    
    def get_explained_variance_ratio(self) -> Optional[np.ndarray]:
        """
        Get explained variance ratio for PCA-based methods.
        
        Returns:
            Array of explained variance ratios, or None if not applicable
        """
        if self.model is not None and hasattr(self.model, 'explained_variance_ratio_'):
            return self.model.explained_variance_ratio_
        return None


def compute_operator_embedding(
    jacobians: Optional[np.ndarray] = None,
    metrics: Optional[Dict[str, np.ndarray]] = None,
    method: str = 'metrics',
    n_components: int = 10
) -> np.ndarray:
    """
    Convenience function to compute operator embedding.
    
    This is a simplified interface for quick embedding computation.
    
    Args:
        jacobians: (n_cells, dim, dim) Jacobian matrices
        metrics: Dictionary with operator metrics
        method: Embedding method ('pca', 'spectrum', or 'metrics')
        n_components: Number of embedding dimensions
        
    Returns:
        embedding: (n_cells, n_components) operator coordinates
        
    Example:
        >>> embedding = compute_operator_embedding(
        ...     metrics=atlas.metrics,
        ...     method='metrics',
        ...     n_components=4
        ... )
        >>> adata.obsm['X_operator'] = embedding
    """
    embedder = OperatorEmbedding(method=method)
    embedding = embedder.fit_transform(
        jacobians=jacobians,
        metrics=metrics,
        n_components=n_components
    )
    return embedding
