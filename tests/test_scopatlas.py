"""
Unit tests for scOpAtlas (Stable Operator Atlas)
"""

import unittest
import torch
import numpy as np
import anndata as ad
from unittest.mock import Mock, MagicMock

import sys
sys.path.insert(0, '/home/ubuntu/scidiff')

from scqdiff.atlas.operator_metrics import OperatorMetrics
from scqdiff.atlas.regime_classifier import OperatorRegimeClassifier
from scqdiff.atlas.atlas_builder import StableOperatorAtlas


class TestOperatorMetrics(unittest.TestCase):
    """Test OperatorMetrics class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a simple mock drift model
        self.drift_model = Mock()
        self.drift_model.eval = Mock()
        self.drift_model.to = Mock(return_value=self.drift_model)
        
        # Create simple linear drift: f(x) = Ax where A has known eigenvalues
        # A = [[-1, 0], [0, 0.5]]  -> eigenvalues: -1, 0.5
        self.A = torch.tensor([[-1.0, 0.0], [0.0, 0.5]])
        
        def mock_forward(x, t):
            # Linear drift: f(x) = Ax
            return torch.matmul(x, self.A.T)
        
        self.drift_model.__call__ = mock_forward
        
        self.metrics_computer = OperatorMetrics(
            drift_model=self.drift_model,
            epsilon=0.1,
            device="cpu"
        )
    
    def test_compute_jacobian(self):
        """Test Jacobian computation"""
        # For linear drift f(x) = Ax, Jacobian should be A
        x = torch.tensor([[1.0, 2.0]], requires_grad=True)
        t = torch.tensor([0.5])
        
        # Note: For linear model, Jacobian = A^T (transpose)
        # This is because we compute ∂f/∂x where f = x @ A^T
        jacobian = self.metrics_computer.compute_jacobian(x, t)
        
        self.assertEqual(jacobian.shape, (1, 2, 2))
        # Check that Jacobian is close to A (within numerical tolerance)
        # Due to autograd, we get A^T
        expected = self.A.T.numpy()
        np.testing.assert_allclose(
            jacobian[0].numpy(), expected, rtol=1e-5, atol=1e-5
        )
    
    def test_compute_eigenvalues(self):
        """Test eigenvalue computation"""
        x = torch.tensor([[1.0, 2.0]])
        t = torch.tensor([0.5])
        
        eigenvalues = self.metrics_computer.compute_eigenvalues(x, t)
        
        self.assertEqual(eigenvalues.shape, (1, 2))
        
        # Eigenvalues should be approximately [-1, 0.5]
        eig_real = eigenvalues.real.numpy()[0]
        eig_real_sorted = np.sort(eig_real)
        expected_sorted = np.sort([-1.0, 0.5])
        
        np.testing.assert_allclose(
            eig_real_sorted, expected_sorted, rtol=1e-4, atol=1e-4
        )
    
    def test_max_unstable_eigenvalue(self):
        """Test max unstable eigenvalue computation"""
        x = torch.tensor([[1.0, 2.0]])
        t = torch.tensor([0.5])
        
        lambda_max = self.metrics_computer.max_unstable_eigenvalue(x, t)
        
        # Should be 0.5 (the positive eigenvalue)
        self.assertAlmostEqual(lambda_max.item(), 0.5, places=3)
    
    def test_stability_depth(self):
        """Test stability depth computation"""
        x = torch.tensor([[1.0, 2.0]])
        t = torch.tensor([0.5])
        
        lambda_min = self.metrics_computer.stability_depth(x, t)
        
        # Should be -1.0 (the negative eigenvalue)
        self.assertAlmostEqual(lambda_min.item(), -1.0, places=3)
    
    def test_plasticity_index(self):
        """Test plasticity index computation"""
        x = torch.tensor([[1.0, 2.0]])
        t = torch.tensor([0.5])
        
        plasticity = self.metrics_computer.plasticity_index(x, t)
        
        # With epsilon=0.1, neither eigenvalue (-1, 0.5) is near-neutral
        # So plasticity should be 0
        self.assertAlmostEqual(plasticity.item(), 0.0, places=3)
    
    def test_stable_subspace_dim(self):
        """Test stable subspace dimension computation"""
        x = torch.tensor([[1.0, 2.0]])
        t = torch.tensor([0.5])
        
        stable_dim = self.metrics_computer.stable_subspace_dim(x, t)
        
        # Only one negative eigenvalue (-1)
        self.assertEqual(stable_dim.item(), 1.0)


class TestOperatorRegimeClassifier(unittest.TestCase):
    """Test OperatorRegimeClassifier class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.classifier = OperatorRegimeClassifier(
            threshold_unstable=0.1,
            threshold_plastic=0.05,
            threshold_deeply_stable=-1.0,
            plasticity_threshold=0.3
        )
    
    def test_classify_stable(self):
        """Test classification of stable regime"""
        metrics = {
            'lambda_max_plus': np.array([-0.5]),
            'lambda_min_minus': np.array([-0.5]),
            'plasticity': np.array([0.2]),
            'stable_dim': np.array([5.0])
        }
        
        regimes, masks = self.classifier.classify(metrics)
        
        self.assertEqual(regimes[0], 'stable')
    
    def test_classify_plastic(self):
        """Test classification of plastic regime"""
        metrics = {
            'lambda_max_plus': np.array([0.01]),  # Near zero
            'lambda_min_minus': np.array([-0.3]),
            'plasticity': np.array([0.5]),  # High plasticity
            'stable_dim': np.array([3.0])
        }
        
        regimes, masks = self.classifier.classify(metrics)
        
        self.assertEqual(regimes[0], 'plastic')
    
    def test_classify_unstable(self):
        """Test classification of unstable regime"""
        metrics = {
            'lambda_max_plus': np.array([0.5]),  # Positive
            'lambda_min_minus': np.array([-0.2]),
            'plasticity': np.array([0.3]),
            'stable_dim': np.array([2.0])
        }
        
        regimes, masks = self.classifier.classify(metrics)
        
        self.assertEqual(regimes[0], 'unstable')
    
    def test_classify_deeply_stable(self):
        """Test classification of deeply stable regime"""
        metrics = {
            'lambda_max_plus': np.array([-0.2]),
            'lambda_min_minus': np.array([-2.0]),  # Very negative
            'plasticity': np.array([0.1]),
            'stable_dim': np.array([8.0])
        }
        
        regimes, masks = self.classifier.classify(metrics)
        
        self.assertEqual(regimes[0], 'deeply_stable')
    
    def test_get_regime_statistics(self):
        """Test regime statistics computation"""
        regimes = np.array(['stable', 'plastic', 'stable', 'unstable'])
        metrics = {
            'lambda_max_plus': np.array([-0.5, 0.0, -0.3, 0.5]),
            'lambda_min_minus': np.array([-1.0, -0.5, -0.8, -0.2]),
            'plasticity': np.array([0.2, 0.5, 0.3, 0.4]),
            'stable_dim': np.array([5.0, 3.0, 4.0, 2.0])
        }
        
        stats = self.classifier.get_regime_statistics(regimes, metrics)
        
        # Check stable regime statistics
        self.assertEqual(stats['stable']['count'], 2)
        self.assertAlmostEqual(stats['stable']['lambda_max_mean'], -0.4, places=5)
        
        # Check plastic regime statistics
        self.assertEqual(stats['plastic']['count'], 1)
        self.assertAlmostEqual(stats['plastic']['plasticity_mean'], 0.5, places=5)
        
        # Check unstable regime statistics
        self.assertEqual(stats['unstable']['count'], 1)
        self.assertAlmostEqual(stats['unstable']['lambda_max_mean'], 0.5, places=5)


class TestStableOperatorAtlas(unittest.TestCase):
    """Test StableOperatorAtlas class"""
    
    def setUp(self):
        """Set up test fixtures with synthetic data"""
        # Create synthetic AnnData
        n_cells = 100
        n_genes = 50
        n_pcs = 10
        
        # Random expression data
        X = np.random.randn(n_cells, n_genes)
        
        # Create AnnData
        self.adata = ad.AnnData(X)
        
        # Add PCA representation
        self.adata.obsm['X_pca'] = np.random.randn(n_cells, n_pcs)
        
        # Add pseudotime
        self.adata.obs['pseudotime'] = np.linspace(0, 1, n_cells)
        
        # Add cell types
        self.adata.obs['cell_type'] = ['TypeA'] * 50 + ['TypeB'] * 50
        
        # Add conditions
        self.adata.obs['condition'] = ['Control'] * 25 + ['Treatment'] * 25 + \
                                       ['Control'] * 25 + ['Treatment'] * 25
        
        # Create mock drift model
        self.drift_model = Mock()
        self.drift_model.eval = Mock()
        self.drift_model.to = Mock(return_value=self.drift_model)
        
        # Simple linear drift
        A = torch.randn(n_pcs, n_pcs) * 0.1
        A = (A + A.T) / 2  # Make symmetric
        
        def mock_forward(x, t):
            return torch.matmul(x, A.T)
        
        self.drift_model.__call__ = mock_forward
    
    def test_initialization(self):
        """Test atlas initialization"""
        atlas = StableOperatorAtlas(
            adata=self.adata,
            drift_model=self.drift_model,
            use_rep="X_pca",
            pseudotime_key="pseudotime"
        )
        
        self.assertIsNotNone(atlas)
        self.assertEqual(atlas.use_rep, "X_pca")
        self.assertEqual(atlas.pseudotime_key, "pseudotime")
    
    def test_validation_missing_rep(self):
        """Test validation with missing representation"""
        with self.assertRaises(ValueError):
            atlas = StableOperatorAtlas(
                adata=self.adata,
                drift_model=self.drift_model,
                use_rep="X_missing",  # This doesn't exist
                pseudotime_key="pseudotime"
            )
    
    def test_build_atlas(self):
        """Test atlas building"""
        atlas = StableOperatorAtlas(
            adata=self.adata,
            drift_model=self.drift_model,
            use_rep="X_pca",
            pseudotime_key="pseudotime"
        )
        
        # Build atlas (this will take some time)
        atlas.build(batch_size=10)
        
        # Check that results are stored
        self.assertIsNotNone(atlas.metrics)
        self.assertIsNotNone(atlas.regimes)
        
        # Check that metrics are in adata.obs
        self.assertIn('operator_regime', atlas.adata.obs.columns)
        self.assertIn('lambda_max_plus', atlas.adata.obs.columns)
        self.assertIn('lambda_min_minus', atlas.adata.obs.columns)
        self.assertIn('plasticity', atlas.adata.obs.columns)
        self.assertIn('stable_dim', atlas.adata.obs.columns)
        
        # Check that all cells have been classified
        self.assertEqual(len(atlas.regimes), len(self.adata))


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestOperatorMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestOperatorRegimeClassifier))
    suite.addTests(loader.loadTestsFromTestCase(TestStableOperatorAtlas))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_tests()
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed.")
