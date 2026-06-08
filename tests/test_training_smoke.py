import pytest

def test_spectral_trainer_two_steps(tiny_loader, tiny_gene_dim):
    try:
        from scjdo.models.fourier_score_network import MultiBandScoreNet
        from scjdo.training.fourier_trainer import SpectralDiffusionTrainer
        import torch
    except Exception as e:
        pytest.skip(f"Trainer or model import failed: {e}")

    model = MultiBandScoreNet(gene_dim=tiny_gene_dim, hidden_dim=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = SpectralDiffusionTrainer(model=model, optimizer=opt, device="cpu")
    history = trainer.fit(tiny_loader, epochs=1)
    assert len(history) > 0
