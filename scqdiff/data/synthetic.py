
import torch, math, numpy as np
def make_spiral(n=2000, noise=0.05, turns=2.0, dim=2, seed=0):
    rng=np.random.default_rng(seed); t=rng.uniform(0,1,n); theta=turns*2*math.pi*t; r=0.1+0.8*t
    x=np.stack([r*np.cos(theta), r*np.sin(theta)], axis=1); x += noise*rng.normal(size=x.shape)
    if dim>2: x=np.concatenate([x, np.zeros((n, dim-2))], axis=1)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(t, dtype=torch.float32)
def rna_velocity_field(x, t, strength=0.5):
    v=torch.zeros_like(x); 
    if x.shape[1]>=2: angle=torch.atan2(x[:,1], x[:,0]); v[:,0]=-torch.sin(angle); v[:,1]=torch.cos(angle); v*=strength
    return v
