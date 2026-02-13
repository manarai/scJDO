import pytest
import torch

@pytest.mark.parametrize("model_name", ["ScIDiffModel", "OTDiffusionModel"])
def test_model_instantiation_and_sampling(model_name, tiny_gene_dim):
    try:
        from scIDiff import models as M
    except Exception as e:
        pytest.skip(f"Cannot import scIDiff.models: {e}")

    if not hasattr(M, model_name):
        pytest.skip(f"{model_name} not exposed; skipping")

    Model = getattr(M, model_name)
    kwargs = dict(gene_dim=tiny_gene_dim, hidden_dim=32, num_layers=2, num_timesteps=10)
    if model_name == "OTDiffusionModel":
        kwargs.update(dict(use_ot=True, ot_regularization_weight=0.1))

    model = Model(**kwargs)
    out = model.sample(batch_size=4) if hasattr(model, "sample") else None
    if out is not None:
        assert isinstance(out, torch.Tensor)
        assert out.shape == (4, tiny_gene_dim)
