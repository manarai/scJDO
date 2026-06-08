
import torch
def fate_probs_from_anndata(adata, keys=("to_fates","to_terminal_states","to_absorption_probabilities","lineages" )):
    import numpy as np
    for k in keys:
        if hasattr(adata,'obsm') and k in adata.obsm:
            M=adata.obsm[k]; 
            if hasattr(M,'toarray'): M=M.toarray()
            names=list(getattr(adata,'uns',{}).get(k+'_names', []))
            return torch.tensor(M, dtype=torch.float32), names
    return None, []
def fate_weighted_timebins(T: torch.Tensor, fate_probs: torch.Tensor, nbins: int = 10, fate_idx: int = 0):
    w=fate_probs[:,fate_idx] if fate_probs is not None else torch.ones_like(T)
    edges=torch.linspace(0.,1.,nbins+1, device=T.device); idx_bins=[]; w_bins=[]
    for i in range(nbins):
        m=(T>=edges[i]) & (T<(edges[i+1]) if i<nbins-1 else T<=edges[i+1]); idx=torch.nonzero(m).squeeze(1)
        idx_bins.append(idx); w_bins.append(w[idx] if idx.numel()>0 else torch.tensor([], device=T.device))
    return edges.cpu(), idx_bins, w_bins
def fate_conditioned_mean_jacobians(model, X: torch.Tensor, T: torch.Tensor, P: torch.Tensor, nbins: int = 10, fate_idx: int = 0):
    edges, idx_bins, w_bins = fate_weighted_timebins(T, P, nbins=nbins, fate_idx=fate_idx); J_list=[]
    for idx,w in zip(idx_bins, w_bins):
        if idx.numel()==0: J_list.append(None); continue
        xb, tb = X[idx], T[idx]; Jb=model.jacobian(xb, tb)
        if w.numel()==0: Jm=Jb.mean(dim=0)
        else: ww=(w/(w.sum()+1e-8)).view(-1,1,1); Jm=(ww*Jb).sum(dim=0)
        J_list.append(Jm.detach())
    for i in range(len(J_list)):
        if J_list[i] is None:
            left = next((j for j in range(i-1,-1,-1) if J_list[j] is not None), None)
            right= next((j for j in range(i+1,len(J_list)) if J_list[j] is not None), None)
            if left is not None and right is not None: J_list[i]=0.5*(J_list[left]+J_list[right])
            elif left is not None: J_list[i]=J_list[left]
            elif right is not None: J_list[i]=J_list[right]
            else: d=X.shape[1]; J_list[i]=torch.zeros(d,d)
    return torch.stack(J_list, dim=0), edges
