import pytest
import torch

def test_sinkhorn_balances_marginals():
    try:
        from scqdiff.transport import sinkhorn as S
    except Exception as e:
        pytest.skip(f"transport.sinkhorn not importable: {e}")

    N, M = 10, 10
    X = torch.arange(N, dtype=torch.float32).unsqueeze(1)
    Y = torch.arange(M, dtype=torch.float32).unsqueeze(1)
    C = torch.cdist(X, Y, p=2)

    if not hasattr(S, "sinkhorn_log"):
        pytest.skip("No sinkhorn_log() available")

    P, f, g = S.sinkhorn_log(C, epsilon=0.1)
    assert P.shape == (N, M)
    rs = P.sum(dim=1)
    cs = P.sum(dim=0)
    assert torch.allclose(rs, torch.full((N,), 1.0 / N), atol=1e-2)
    assert torch.allclose(cs, torch.full((M,), 1.0 / M), atol=1e-2)
