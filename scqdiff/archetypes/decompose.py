
import torch
def jacobian_modes(J_tensor, rank=5):
    if J_tensor.dim()==4: J=J_tensor.mean(dim=1)
    else: J=J_tensor
    T,d1,d2=J.shape; M=J.reshape(T, d1*d2); U,S,Vh=torch.linalg.svd(M, full_matrices=False)
    U_r=U[:,:rank]; V_r=Vh[:rank,:].T; patterns=V_r.reshape(d1,d2,rank).permute(2,0,1)
    return patterns, U_r, S[:rank]
