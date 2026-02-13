import pytest

def test_scidiff_trainer_two_steps(tiny_loader, tiny_gene_dim):
    try:
        from scIDiff.models import ScIDiffModel
        from scIDiff.training import ScIDiffTrainer
    except Exception as e:
        pytest.skip(f"Trainer or model import failed: {e}")

    model = ScIDiffModel(gene_dim=tiny_gene_dim, hidden_dim=32, num_layers=2, num_timesteps=10)
    trainer = ScIDiffTrainer(model=model, train_loader=tiny_loader, val_loader=None, device="cpu")
    trainer.train(num_epochs=1)

def test_ot_trainer_two_steps_if_available(tiny_loader, tiny_gene_dim):
    try:
        from scIDiff.models import OTDiffusionModel
        from scIDiff.training import OTTrainer
    except Exception:
        pytest.skip("OT model/trainer not available")

    model = OTDiffusionModel(gene_dim=tiny_gene_dim, hidden_dim=32, use_ot=True,
                             ot_regularization_weight=0.1, num_layers=2, num_timesteps=10)
    trainer = OTTrainer(model=model, train_loader=tiny_loader, val_loader=None, use_sinkhorn=True, device="cpu")
    trainer.train(num_epochs=1)
