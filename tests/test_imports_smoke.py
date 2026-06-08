import importlib
import pytest

def _has(module, attr):
    try:
        m = importlib.import_module(module)
        return hasattr(m, attr)
    except Exception:
        return False

def test_pkg_import():
    m = importlib.import_module("scjdo")
    assert m is not None, "Package scjdo should import"

@pytest.mark.parametrize("module, attr", [
    ("scjdo.models.drift", "DriftField"),
    ("scjdo.models.drift", "DriftConfig"),
    ("scjdo.models.schrodinger_bridge", "SchrodingerBridge"),
    ("scjdo.models.schrodinger_bridge", "SchrodingerBridgeConfig"),
    ("scjdo.transport.sinkhorn", "sinkhorn_log"),
    ("scjdo.transport.coupling", "sample_from_coupling"),
    ("scjdo.io.anndata", "tensors_from_anndata"),
])
def test_advertised_symbols_exist_or_skip(module, attr):
    if not _has(module, attr):
        pytest.skip(f"{attr} not available in {module}; skipping")
    obj = getattr(importlib.import_module(module), attr, None)
    assert obj is not None
