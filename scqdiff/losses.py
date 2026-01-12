
import torch, torch.nn.functional as F
def denoising_score_matching(model, x, t, sigma):
    noise=torch.randn_like(x)*sigma; x_noisy=x+noise; target=-noise/(sigma**2); s=model.score(x_noisy,t); return F.mse_loss(s,target)
def control_energy(u_pred): return (u_pred**2).mean()
def fp_residual_loss(model, x, t, beta, nabla_logrho_est=None, dxdt_est=None):
    x=x.requires_grad_(True); u=model(x,t); div_terms=[]
    for k in range(u.shape[1]): g=torch.autograd.grad(u[:,k].sum(), x, create_graph=True)[0][:,k]; div_terms.append(g)
    div_u=torch.stack(div_terms, dim=1).sum(dim=1); return (div_u**2).mean()
def laplacian_smooth_drift(u_pred, L):
    if L is None or L.numel()==0: return u_pred.new_zeros(())
    return torch.einsum('bi,ij,bj->', u_pred, L, u_pred)/u_pred.shape[0]
