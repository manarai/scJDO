"""
Operator Regime Classifier

Classifies cells into operator regimes based on eigenvalue-derived metrics.
"""

import numpy as np
from typing import Dict, Tuple, Optional


class OperatorRegimeClassifier:
    """
    Classify cells into operator regimes based on stability metrics.
    
    Operator regimes:
    - **Stable**: λ_max⁺ ≤ 0, large S (terminal states, homeostasis)
    - **Plastic**: λ_max⁺ ≈ 0, high P (progenitor states, decision points)
    - **Unstable**: λ_max⁺ > τ (transition states, bifurcations)
    - **Deeply Stable**: Very negative λ_min⁻ (resistant states, locked-in fates)
    
    Args:
        threshold_unstable: Threshold for unstable regime (default: 0.1)
        threshold_plastic: Threshold for near-zero eigenvalues (default: 0.05)
        threshold_deeply_stable: Threshold for deeply stable regime (default: -1.0)
        plasticity_threshold: Minimum plasticity index for plastic regime (default: 0.3)
    """
    
    def __init__(
        self,
        threshold_unstable: float = 0.1,
        threshold_plastic: float = 0.05,
        threshold_deeply_stable: float = -1.0,
        plasticity_threshold: float = 0.3
    ):
        self.tau = threshold_unstable
        self.eps = threshold_plastic
        self.deeply_stable_threshold = threshold_deeply_stable
        self.plasticity_threshold = plasticity_threshold
    
    def classify(
        self,
        metrics: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Classify cells into operator regimes.
        
        Classification logic:
        1. If λ_max⁺ > τ → Unstable
        2. Elif λ_min⁻ < deeply_stable_threshold → Deeply Stable
        3. Elif λ_max⁺ ≈ 0 and P > plasticity_threshold → Plastic
        4. Else → Stable
        
        Args:
            metrics: Dictionary with keys 'lambda_max_plus', 'lambda_min_minus',
                    'plasticity', 'stable_dim'
        
        Returns:
            regimes: Array of regime labels (n_cells,)
            regime_masks: Dictionary of boolean masks for each regime
        """
        n_cells = len(metrics['lambda_max_plus'])
        
        lambda_max = metrics['lambda_max_plus']
        lambda_min = metrics['lambda_min_minus']
        plasticity = metrics['plasticity']
        stable_dim = metrics['stable_dim']
        
        # Initialize regime array
        regimes = np.empty(n_cells, dtype=object)
        
        # Classify based on criteria
        unstable_mask = lambda_max > self.tau
        deeply_stable_mask = (lambda_min < self.deeply_stable_threshold) & (~unstable_mask)
        plastic_mask = (
            (np.abs(lambda_max) < self.eps) &
            (plasticity > self.plasticity_threshold) &
            (~unstable_mask) &
            (~deeply_stable_mask)
        )
        stable_mask = (
            (~unstable_mask) &
            (~deeply_stable_mask) &
            (~plastic_mask)
        )
        
        # Assign labels
        regimes[unstable_mask] = 'unstable'
        regimes[deeply_stable_mask] = 'deeply_stable'
        regimes[plastic_mask] = 'plastic'
        regimes[stable_mask] = 'stable'
        
        # Create regime masks dictionary
        regime_masks = {
            'unstable': unstable_mask,
            'deeply_stable': deeply_stable_mask,
            'plastic': plastic_mask,
            'stable': stable_mask
        }
        
        return regimes, regime_masks
    
    def classify_with_confidence(
        self,
        metrics: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify cells with confidence scores.
        
        Confidence is based on how far the metrics are from decision boundaries.
        
        Args:
            metrics: Dictionary of operator metrics
            
        Returns:
            regimes: Array of regime labels (n_cells,)
            confidence: Confidence scores (n_cells,) in [0, 1]
        """
        regimes, regime_masks = self.classify(metrics)
        
        lambda_max = metrics['lambda_max_plus']
        lambda_min = metrics['lambda_min_minus']
        plasticity = metrics['plasticity']
        
        n_cells = len(regimes)
        confidence = np.zeros(n_cells)
        
        # Compute confidence for each regime
        for i in range(n_cells):
            if regimes[i] == 'unstable':
                # Distance from unstable threshold
                confidence[i] = np.clip((lambda_max[i] - self.tau) / self.tau, 0, 1)
            
            elif regimes[i] == 'deeply_stable':
                # Distance from deeply stable threshold
                confidence[i] = np.clip(
                    (self.deeply_stable_threshold - lambda_min[i]) / abs(self.deeply_stable_threshold),
                    0, 1
                )
            
            elif regimes[i] == 'plastic':
                # Based on plasticity index
                confidence[i] = np.clip(
                    (plasticity[i] - self.plasticity_threshold) / (1 - self.plasticity_threshold),
                    0, 1
                )
            
            else:  # stable
                # Based on negative λ_max and moderate plasticity
                confidence[i] = np.clip(
                    (self.tau - lambda_max[i]) / self.tau,
                    0, 1
                )
        
        return regimes, confidence
    
    def get_regime_statistics(
        self,
        regimes: np.ndarray,
        metrics: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics for each regime.
        
        Args:
            regimes: Array of regime labels
            metrics: Dictionary of operator metrics
            
        Returns:
            Dictionary mapping regime names to statistics
        """
        unique_regimes = np.unique(regimes)
        stats = {}
        
        for regime in unique_regimes:
            mask = regimes == regime
            
            stats[regime] = {
                'count': mask.sum(),
                'fraction': mask.mean(),
                'lambda_max_mean': metrics['lambda_max_plus'][mask].mean(),
                'lambda_max_std': metrics['lambda_max_plus'][mask].std(),
                'lambda_min_mean': metrics['lambda_min_minus'][mask].mean(),
                'lambda_min_std': metrics['lambda_min_minus'][mask].std(),
                'plasticity_mean': metrics['plasticity'][mask].mean(),
                'plasticity_std': metrics['plasticity'][mask].std(),
                'stable_dim_mean': metrics['stable_dim'][mask].mean(),
                'stable_dim_std': metrics['stable_dim'][mask].std(),
            }
        
        return stats
    
    def compare_regimes_across_conditions(
        self,
        regimes: np.ndarray,
        conditions: np.ndarray,
        celltype: Optional[np.ndarray] = None
    ) -> Dict[str, Dict]:
        """
        Compare operator regime distributions across conditions.
        
        This is critical for showing non-redundancy with expression-based cell types.
        
        Args:
            regimes: Array of regime labels (n_cells,)
            conditions: Array of condition labels (n_cells,)
            celltype: Optional array of cell type labels (n_cells,)
            
        Returns:
            Dictionary with comparison statistics
        """
        unique_conditions = np.unique(conditions)
        comparison = {}
        
        # Overall regime distribution per condition
        for cond in unique_conditions:
            cond_mask = conditions == cond
            cond_regimes = regimes[cond_mask]
            
            regime_counts = {}
            for regime in ['stable', 'plastic', 'unstable', 'deeply_stable']:
                regime_counts[regime] = (cond_regimes == regime).sum()
            
            comparison[cond] = {
                'total_cells': cond_mask.sum(),
                'regime_counts': regime_counts,
                'regime_fractions': {
                    k: v / cond_mask.sum() for k, v in regime_counts.items()
                }
            }
        
        # If cell types provided, compare within cell types
        if celltype is not None:
            comparison['by_celltype'] = {}
            unique_celltypes = np.unique(celltype)
            
            for ct in unique_celltypes:
                ct_mask = celltype == ct
                comparison['by_celltype'][ct] = {}
                
                for cond in unique_conditions:
                    cond_ct_mask = ct_mask & (conditions == cond)
                    if cond_ct_mask.sum() == 0:
                        continue
                    
                    cond_ct_regimes = regimes[cond_ct_mask]
                    
                    regime_counts = {}
                    for regime in ['stable', 'plastic', 'unstable', 'deeply_stable']:
                        regime_counts[regime] = (cond_ct_regimes == regime).sum()
                    
                    comparison['by_celltype'][ct][cond] = {
                        'total_cells': cond_ct_mask.sum(),
                        'regime_counts': regime_counts,
                        'regime_fractions': {
                            k: v / cond_ct_mask.sum() if cond_ct_mask.sum() > 0 else 0
                            for k, v in regime_counts.items()
                        }
                    }
        
        return comparison
