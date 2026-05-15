"""Training losses for the drift field."""
import torch
import torch.nn.functional as F


def denoising_score_matching(
    model, x: torch.Tensor, t: torch.Tensor, sigma=0.1
) -> torch.Tensor:
    """
    Denoising score matching loss.

    Parameters
    ----------
    sigma : float or (B,) tensor
        Noise level. Pass a per-sample tensor for local adaptive noise.
    """
    if isinstance(sigma, (float, int)):
        sigma_t = torch.full((x.shape[0],), float(sigma), device=x.device)
    else:
        sigma_t = sigma.to(x.device)

    sigma_t = sigma_t.view(-1, *([1] * (x.dim() - 1)))
    noise   = torch.randn_like(x)
    x_noisy = x + sigma_t * noise
    target  = -noise / sigma_t
    s       = model.score(x_noisy, t)
    return F.mse_loss(s, target)


def control_energy(u: torch.Tensor) -> torch.Tensor:
    """L2 penalty on drift magnitude — prevents exploding vectors."""
    return (u ** 2).mean()


def fp_residual_loss(model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Fokker-Planck divergence regularizer (expensive — use sparingly)."""
    x = x.requires_grad_(True)
    u = model(x, t)
    divs = []
    for k in range(u.shape[1]):
        g = torch.autograd.grad(u[:, k].sum(), x, create_graph=True)[0][:, k]
        divs.append(g)
    return torch.stack(divs, dim=1).sum(dim=1).pow(2).mean()


def local_sigma(X_pca: torch.Tensor, k: int = 10,
                lo: float = 0.05, hi: float = 0.5) -> torch.Tensor:
    """
    Per-cell adaptive DSM noise: sigma_i = distance to k-th nearest neighbour.
    Denser regions get smaller sigma (fine-grained score), sparse regions larger.
    """
    D  = torch.cdist(X_pca, X_pca)          # (N, N)
    kd = D.topk(k + 1, largest=False).values[:, -1]  # k-th neighbour distance
    return kd.clamp(lo, hi)
