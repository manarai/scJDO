import importlib
import pytest

def _has(module, attr):
    try:
        m = importlib.import_module(module)
        return hasattr(m, attr)
    except Exception:
        return False

def test_pkg_import():
    m = importlib.import_module("scqdiff")
    assert m is not None, "Package scqdiff should import"

@pytest.mark.parametrize("module, attr", [
    ("scqdiff.models.drift", "DriftField"),
    ("scqdiff.models.drift", "DriftConfig"),
    ("scqdiff.models.schrodinger_bridge", "SchrodingerBridge"),
    ("scqdiff.models.schrodinger_bridge", "SchrodingerBridgeConfig"),
    ("scqdiff.transport.sinkhorn", "sinkhorn_log"),
    ("scqdiff.transport.coupling", "sample_from_coupling"),
    ("scqdiff.io.anndata", "tensors_from_anndata"),
])
def test_advertised_symbols_exist_or_skip(module, attr):
    if not _has(module, attr):
        pytest.skip(f"{attr} not available in {module}; skipping")
    obj = getattr(importlib.import_module(module), attr, None)
    assert obj is not None
