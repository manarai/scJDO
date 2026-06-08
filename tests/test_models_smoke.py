import pytest
import torch

@pytest.mark.parametrize("model_name", ["DriftField", "SchrodingerBridge"])
def test_model_instantiation(model_name, tiny_gene_dim):
    if model_name == "DriftField":
        try:
            from scjdo.models.drift import DriftField, DriftConfig
        except Exception as e:
            pytest.skip(f"Cannot import scjdo.models.drift: {e}")

        cfg = DriftConfig(dim=tiny_gene_dim, hidden=32, depth=2)
        model = DriftField(cfg)
        x = torch.randn(4, tiny_gene_dim)
        t = torch.rand(4)
        out = model(x, t)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (4, tiny_gene_dim)

    elif model_name == "SchrodingerBridge":
        try:
            from scjdo.models.schrodinger_bridge import SchrodingerBridge, SchrodingerBridgeConfig
        except Exception as e:
            pytest.skip(f"Cannot import scjdo.models.schrodinger_bridge: {e}")

        X_0 = torch.randn(16, tiny_gene_dim)
        X_1 = torch.randn(16, tiny_gene_dim) + 1.0
        cfg = SchrodingerBridgeConfig(dim=tiny_gene_dim, hidden=32, depth=2, max_iterations=2, n_score_steps=2)
        model = SchrodingerBridge(cfg, X_0, X_1)
        assert model is not None
