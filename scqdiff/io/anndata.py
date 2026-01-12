
from typing import Optional, Tuple, Dict
import numpy as np, torch
import anndata as ad
from scipy.sparse import csgraph

_PTIME = ["rank_pseudotime","latent_time","dpt_pseudotime","pseudotime"]
_VEL = ["velocity","velocity_u","velocity_s"]
def _pick_key(d, cands): 
    for k in cands:
        if k in d: return k
    return None
def tensors_from_anndata(adata, use_raw: bool=False, n_hvg: Optional[int]=None, x_key: Optional[str]=None,
                         vel_layer: Optional[str]=None, pseudotime_key: Optional[str]=None, device: str="cpu") -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    A=adata.copy()
    if x_key is not None: X=A.layers[x_key]
    elif use_raw and A.raw is not None: X=A.raw.X
    else: X=A.X
    if not isinstance(X, np.ndarray): X=X.toarray()
    X=X.astype(np.float32)
    if n_hvg is not None and 'highly_variable' in A.var:
        hv=A.var['highly_variable'].values
        if hv.sum()>0: X=X[:,hv]
    if vel_layer is None and hasattr(A,'layers'): vel_layer=_pick_key(A.layers, _VEL)
    V=None
    if vel_layer is not None and vel_layer in A.layers:
        V=A.layers[vel_layer]; 
        if not isinstance(V, np.ndarray): V=V.toarray()
        V=V.astype(np.float32)
        if X.shape[1]!=V.shape[1] and 'highly_variable' in A.var:
            hv=A.var['highly_variable'].values
            if hv.sum()>0 and hv.sum()==X.shape[1]: V=V[:,hv]
    if pseudotime_key is None: pseudotime_key=_pick_key(A.obs, _PTIME)
    if pseudotime_key is None: raise KeyError("No pseudotime column found; supply --ptime-key.")
    T=A.obs[pseudotime_key].values.astype(np.float32); tmin,tmax=np.nanmin(T),np.nanmax(T); T=(T-tmin)/(tmax-tmin) if tmax>tmin else np.zeros_like(T, dtype=np.float32)
    return torch.from_numpy(X).to(device), (torch.from_numpy(V).to(device) if V is not None else None), torch.from_numpy(T).to(device)
def laplacian_from_connectivities(adata, normalized: bool=True):
    if 'connectivities' in getattr(adata,'obsp',{}):
        A=adata.obsp['connectivities']; L=csgraph.laplacian(A, normed=normalized); return torch.from_numpy(L.toarray()).float()
    return None
