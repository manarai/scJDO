import random
import numpy as np
import pytest
import torch

@pytest.fixture(autouse=True)
def _reproducibility_seed():
    seed = 13
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@pytest.fixture
def tiny_gene_dim():
    return 32

@pytest.fixture
def tiny_batch():
    return 8

@pytest.fixture
def tiny_loader(tiny_gene_dim, tiny_batch):
    class _TinyLoader:
        def __iter__(self):
            for _ in range(2):
                yield torch.randn(tiny_batch, tiny_gene_dim)
        def __len__(self):
            return 2
    return _TinyLoader()
