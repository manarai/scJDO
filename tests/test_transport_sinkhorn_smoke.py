import pytest
import torch

def test_sinkhorn_balances_marginals():
    try:
        from scIDiff.transport import sinkhorn as S
    except Exception as e:
        pytest.skip(f"transport.sinkhorn not importable: {e}")

    a = torch.full((10,), 0.1)
    b = torch.full((10,), 0.1)
    C = torch.cdist(torch.arange(10, dtype=torch.float32).unsqueeze(1),
                    torch.arange(10, dtype=torch.float32).unsqueeze(1), p=2)

    if not hasattr(S, "sinkhorn"):
        pytest.skip("No sinkhorn() available")

    T = S.sinkhorn(a, b, C, reg=0.1)  # Adjust if API differs
    assert T.shape == (10, 10)
    rs = T.sum(dim=1)
    cs = T.sum(dim=0)
    assert torch.allclose(rs, a, atol=1e-2)
    assert torch.allclose(cs, b, atol=1e-2)
