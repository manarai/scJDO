
import numpy as np, matplotlib.pyplot as plt
def plot_archetype_heatmaps(patterns_path, max_archetypes=3, figsize=(6,5), save_prefix=None):
    P=np.load(patterns_path); r=min(P.shape[0], max_archetypes)
    for i in range(r):
        plt.figure(figsize=figsize); plt.imshow(P[i], aspect='auto'); plt.title(f'Archetype {i}'); plt.colorbar(); plt.tight_layout()
        if save_prefix: plt.savefig(f"{save_prefix}.archetype_{i}.png", dpi=150); plt.show()
def plot_temporal_activations(U_t_path, edges_meta=None, figsize=(7,4), save_path=None):
    U=np.load(U_t_path); import numpy as np
    plt.figure(figsize=figsize); x=np.arange(U.shape[0]); plt.plot(x, U); 
    if edges_meta and len(edges_meta)==U.shape[0]+1:
        for xi in range(1,len(edges_meta)-1): plt.axvline(xi, linestyle='--', linewidth=1)
    plt.xlabel('time bins'); plt.ylabel('activation'); plt.title('Temporal archetype activations'); plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150); plt.show()
def plot_singular_values(S_path, figsize=(5,4), save_path=None):
    S=np.load(S_path); import numpy as np
    plt.figure(figsize=figsize); plt.plot(np.arange(len(S)), S, marker='o'); plt.title('Singular values'); plt.xlabel('component'); plt.ylabel('value'); plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150); plt.show()
