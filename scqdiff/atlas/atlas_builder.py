"""
Stable Operator Atlas Builder

Main interface for building and managing operator-based cell state atlases.
"""

import torch
import numpy as np
import anndata as ad
from typing import Dict, Optional, List, Tuple
import warnings

from .operator_metrics import OperatorMetrics
from .regime_classifier import OperatorRegimeClassifier


class StableOperatorAtlas:
    """
    Stable Operator Atlas: Defines cellular states by their local stability structure.
    
    This is the main interface for scOpAtlas functionality. It takes a trained
    scidiff drift model and computes operator regimes for all cells in an AnnData object.
    
    Key capabilities:
    - Compute operator metrics (λ_max⁺, λ_min⁻, P, S)
    - Classify cells into operator regimes (stable/plastic/unstable/deeply_stable)
    - Compare regimes across conditions
    - Validate non-redundancy with expression-based cell types
    - Store results in AnnData for downstream analysis
    
    Args:
        adata: AnnData object with single-cell data
        drift_model: Trained DriftField model from scidiff
        use_rep: Key in adata.obsm for state representation (default: "X_pca")
        pseudotime_key: Key in adata.obs for pseudotime (default: "pseudotime")
        device: Device for computation (default: "cpu")
    
    Example:
        >>> from scqdiff.models.drift import DriftField
        >>> from scqdiff.atlas import StableOperatorAtlas
        >>> 
        >>> # Load data and model
        >>> adata = ad.read_h5ad("data.h5ad")
        >>> drift_model = DriftField.load("my_model.pt")
        >>> 
        >>> # Build atlas
        >>> atlas = StableOperatorAtlas(adata, drift_model)
        >>> atlas.build()
        >>> 
        >>> # Access results
        >>> print(atlas.regimes)
        >>> print(atlas.metrics)
    """
    
    def __init__(
        self,
        adata: ad.AnnData,
        drift_model,
        use_rep: str = "X_pca",
        pseudotime_key: str = "pseudotime",
        device: str = "cpu"
    ):
        self.adata = adata
        self.drift_model = drift_model
        self.use_rep = use_rep
        self.pseudotime_key = pseudotime_key
        self.device = device
        
        # Initialize components
        self.metrics_computer = None
        self.classifier = None
        
        # Results
        self.metrics = None
        self.regimes = None
        self.regime_masks = None
        self.confidence = None
        
        # Validate inputs
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate that required data is present."""
        if self.use_rep not in self.adata.obsm:
            raise ValueError(
                f"Representation '{self.use_rep}' not found in adata.obsm. "
                f"Available: {list(self.adata.obsm.keys())}"
            )
        
        if self.pseudotime_key not in self.adata.obs:
            warnings.warn(
                f"Pseudotime key '{self.pseudotime_key}' not found in adata.obs. "
                f"Will use uniform time points."
            )
    
    def build(
        self,
        epsilon: float = 0.1,
        threshold_unstable: float = 0.1,
        threshold_plastic: float = 0.05,
        threshold_deeply_stable: float = -1.0,
        plasticity_threshold: float = 0.3,
        batch_size: int = 32,
        compute_confidence: bool = True
    ):
        """
        Build the Stable Operator Atlas.
        
        This is the main method that computes all operator metrics and classifies
        cells into operator regimes.
        
        Args:
            epsilon: Threshold for near-neutral modes (plasticity index)
            threshold_unstable: Threshold for unstable regime
            threshold_plastic: Threshold for near-zero eigenvalues
            threshold_deeply_stable: Threshold for deeply stable regime
            plasticity_threshold: Minimum plasticity index for plastic regime
            batch_size: Batch size for processing
            compute_confidence: Whether to compute confidence scores
        """
        print("Building Stable Operator Atlas...")
        
        # Initialize metrics computer
        self.metrics_computer = OperatorMetrics(
            drift_model=self.drift_model,
            epsilon=epsilon,
            device=self.device
        )
        
        # Initialize classifier
        self.classifier = OperatorRegimeClassifier(
            threshold_unstable=threshold_unstable,
            threshold_plastic=threshold_plastic,
            threshold_deeply_stable=threshold_deeply_stable,
            plasticity_threshold=plasticity_threshold
        )
        
        # Get state representation
        X = self.adata.obsm[self.use_rep]
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Get pseudotime
        if self.pseudotime_key in self.adata.obs:
            pseudotime = self.adata.obs[self.pseudotime_key].values
        else:
            # Use uniform time points
            pseudotime = np.linspace(0, 1, len(self.adata))
            warnings.warn("Using uniform time points as pseudotime not found.")
        
        # Compute operator metrics
        print("Computing operator metrics...")
        self.metrics = self.metrics_computer.compute_metrics_at_pseudotime(
            X_tensor, pseudotime, batch_size=batch_size
        )
        
        # Classify regimes
        print("Classifying operator regimes...")
        if compute_confidence:
            self.regimes, self.confidence = self.classifier.classify_with_confidence(
                self.metrics
            )
        else:
            self.regimes, self.regime_masks = self.classifier.classify(self.metrics)
            self.confidence = None
        
        # Store results in AnnData
        self._store_results()
        
        print("Atlas building complete!")
        self._print_summary()
    
    def _store_results(self):
        """Store operator metrics and regimes in AnnData object."""
        # Store metrics in obs
        self.adata.obs['operator_regime'] = self.regimes
        self.adata.obs['lambda_max_plus'] = self.metrics['lambda_max_plus']
        self.adata.obs['lambda_min_minus'] = self.metrics['lambda_min_minus']
        self.adata.obs['plasticity'] = self.metrics['plasticity']
        self.adata.obs['stable_dim'] = self.metrics['stable_dim']
        
        if self.confidence is not None:
            self.adata.obs['regime_confidence'] = self.confidence
        
        # Store full eigenvalue spectra in uns (for advanced analysis)
        self.adata.uns['operator_eigenvalues'] = self.metrics['eigenvalues']
    
    def _print_summary(self):
        """Print summary statistics of the atlas."""
        print("\n=== Operator Regime Summary ===")
        unique_regimes, counts = np.unique(self.regimes, return_counts=True)
        
        for regime, count in zip(unique_regimes, counts):
            fraction = count / len(self.regimes)
            print(f"{regime:15s}: {count:6d} cells ({fraction*100:5.1f}%)")
        
        print("\n=== Operator Metrics Summary ===")
        for metric_name, metric_values in self.metrics.items():
            if metric_name != 'eigenvalues':
                print(f"{metric_name:20s}: mean={metric_values.mean():7.3f}, std={metric_values.std():7.3f}")
    
    def validate_nonredundancy(
        self,
        celltype_key: str,
        condition_key: Optional[str] = None
    ) -> Dict:
        """
        Validate that operator regimes are not redundant with cell types.
        
        This is critical for demonstrating that operator regimes provide
        information beyond expression-based cell type annotations.
        
        Two key tests:
        1. Same cell type → different operator regimes across conditions
        2. Different cell types → shared operator regime
        
        Args:
            celltype_key: Key in adata.obs for cell type labels
            condition_key: Optional key in adata.obs for condition labels
            
        Returns:
            Dictionary with validation results
        """
        if celltype_key not in self.adata.obs:
            raise ValueError(f"Cell type key '{celltype_key}' not found in adata.obs")
        
        celltype = self.adata.obs[celltype_key].values
        
        validation = {
            'celltype_key': celltype_key,
            'condition_key': condition_key
        }
        
        # Test 1: Regime diversity within cell types
        print("\n=== Non-Redundancy Test 1: Regime Diversity Within Cell Types ===")
        unique_celltypes = np.unique(celltype)
        
        celltype_regime_diversity = {}
        for ct in unique_celltypes:
            ct_mask = celltype == ct
            ct_regimes = self.regimes[ct_mask]
            
            unique_regimes, counts = np.unique(ct_regimes, return_counts=True)
            regime_diversity = len(unique_regimes)
            
            celltype_regime_diversity[ct] = {
                'n_cells': ct_mask.sum(),
                'n_regimes': regime_diversity,
                'regime_distribution': dict(zip(unique_regimes, counts)),
                'entropy': self._compute_entropy(counts)
            }
            
            print(f"{ct:20s}: {regime_diversity} regimes across {ct_mask.sum()} cells")
        
        validation['celltype_regime_diversity'] = celltype_regime_diversity
        
        # Test 2: If conditions provided, compare within cell types
        if condition_key is not None and condition_key in self.adata.obs:
            print("\n=== Non-Redundancy Test 2: Same Cell Type, Different Conditions ===")
            conditions = self.adata.obs[condition_key].values
            
            comparison = self.classifier.compare_regimes_across_conditions(
                self.regimes, conditions, celltype
            )
            
            validation['condition_comparison'] = comparison
            
            # Print key findings
            if 'by_celltype' in comparison:
                for ct in unique_celltypes:
                    if ct in comparison['by_celltype']:
                        print(f"\n{ct}:")
                        for cond, stats in comparison['by_celltype'][ct].items():
                            fracs = stats['regime_fractions']
                            print(f"  {cond}: stable={fracs.get('stable', 0):.2f}, "
                                  f"plastic={fracs.get('plastic', 0):.2f}, "
                                  f"unstable={fracs.get('unstable', 0):.2f}")
        
        return validation
    
    def _compute_entropy(self, counts: np.ndarray) -> float:
        """Compute Shannon entropy of regime distribution."""
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    def compare_conditions(
        self,
        condition_key: str,
        celltype_key: Optional[str] = None
    ) -> Dict:
        """
        Compare operator regimes across experimental conditions.
        
        Args:
            condition_key: Key in adata.obs for condition labels
            celltype_key: Optional key for cell type labels
            
        Returns:
            Dictionary with comparison results
        """
        if condition_key not in self.adata.obs:
            raise ValueError(f"Condition key '{condition_key}' not found in adata.obs")
        
        conditions = self.adata.obs[condition_key].values
        celltype = self.adata.obs[celltype_key].values if celltype_key else None
        
        comparison = self.classifier.compare_regimes_across_conditions(
            self.regimes, conditions, celltype
        )
        
        return comparison
    
    def get_regime_statistics(self) -> Dict:
        """Get summary statistics for each operator regime."""
        return self.classifier.get_regime_statistics(self.regimes, self.metrics)
    
    def to_anndata(self) -> ad.AnnData:
        """
        Return AnnData object with atlas results.
        
        Returns:
            AnnData object with operator metrics and regimes stored
        """
        return self.adata
    
    def save(self, filename: str):
        """
        Save atlas results to H5AD file.
        
        Args:
            filename: Output filename (should end with .h5ad)
        """
        self.adata.write_h5ad(filename)
        print(f"Atlas saved to {filename}")
