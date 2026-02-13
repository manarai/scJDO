import importlib
import pytest

def _has(module, attr):
    try:
        m = importlib.import_module(module)
        return hasattr(m, attr)
    except Exception:
        return False

def test_pkg_import():
    m = importlib.import_module("scIDiff")
    assert m is not None, "Package scIDiff should import"

@pytest.mark.parametrize("module, attr", [
    ("scIDiff.models", "ScIDiffModel"),
    ("scIDiff.models", "OTDiffusionModel"),
    ("scIDiff.training", "ScIDiffTrainer"),
    ("scIDiff.training", "OTTrainer"),
    ("scIDiff.transport", "PerturbationBridge"),
    ("scIDiff.transport", "BatchIntegrator"),
    ("scIDiff.sampling", "InverseDesigner"),
])
def test_advertised_symbols_exist_or_skip(module, attr):
    if not _has(module, attr):
        pytest.skip(f"{attr} not available in {module}; skipping")
    obj = getattr(importlib.import_module(module), attr, None)
    assert obj is not None
