
import argparse, torch, json, numpy as np, anndata as ad
from scqdiff.io.anndata import tensors_from_anndata, laplacian_from_connectivities
from scqdiff.models.drift import DriftField, DriftConfig
from scqdiff.losses import denoising_score_matching, control_energy, fp_residual_loss, laplacian_smooth_drift
from scqdiff.utils.graph import build_knn_graph, graph_laplacian
from scqdiff.archetypes.decompose import jacobian_modes
from scqdiff.archetypes.cellrank import fate_probs_from_anndata, fate_conditioned_mean_jacobians
from tqdm import trange

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--h5ad', required=True); ap.add_argument('--vel-layer', default=None); ap.add_argument('--ptime-key', default=None)
    ap.add_argument('--use-raw', action='store_true'); ap.add_argument('--n-hvg', type=int, default=None)
    ap.add_argument('--epochs', type=int, default=200); ap.add_argument('--beta', type=float, default=0.1); ap.add_argument('--sigma', type=float, default=0.2)
    ap.add_argument('--laplacian-lambda', type=float, default=0.0); ap.add_argument('--laplacian-reg', type=float, default=1e-3); ap.add_argument('--k', type=int, default=15)
    ap.add_argument('--fate-index', type=int, default=None); ap.add_argument('--nbins', type=int, default=10); ap.add_argument('--rank', type=int, default=3)
    ap.add_argument('--out-prefix', type=str, default='scqdiff_from_anndata')
    args=ap.parse_args()

    adata=ad.read_h5ad(args.h5ad)
    X,V,T=tensors_from_anndata(adata, use_raw=args.use_raw, n_hvg=args.n_hvg, vel_layer=args.vel_layer, pseudotime_key=args.ptime_key)
    L=laplacian_from_connectivities(adata)
    if L is None:
        from scqdiff.utils.graph import build_knn_graph, graph_laplacian
        A=build_knn_graph(X.numpy(), k=args.k, mode='distance'); L=graph_laplacian(A, normalized=True)
    cfg=DriftConfig(dim=X.shape[1], beta=args.beta, laplacian_lambda=args.laplacian_lambda); model=DriftField(cfg, laplacian=L)
    opt=torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in trange(args.epochs):
        idx=np.random.choice(X.shape[0], size=min(2048, X.shape[0]), replace=True); xb,tb=X[idx],T[idx]
        loss=denoising_score_matching(model, xb, tb, sigma=args.sigma); u=model(xb,tb)
        if V is not None: loss += 0.5*((u-V[idx])**2).mean()
        loss += 1e-3*control_energy(u) + 1e-3*fp_residual_loss(model, xb, tb, beta=args.beta) + args.laplacian_reg*laplacian_smooth_drift(u, model.L)
        opt.zero_grad(); loss.backward(); opt.step()
    torch.save({'model':model.state_dict(),'cfg':cfg}, f"{args.out_prefix}.pt"); print(f"Saved {args.out_prefix}.pt")
    if args.fate_index is not None:
        from scqdiff.archetypes.cellrank import fate_probs_from_anndata, fate_conditioned_mean_jacobians
        P,names=fate_probs_from_anndata(adata)
        if P is not None:
            J_bins, edges = fate_conditioned_mean_jacobians(model, X, T, P, nbins=args.nbins, fate_idx=args.fate_index)
            patterns, U_t, S = jacobian_modes(J_bins, rank=args.rank)
            import numpy as np, json
            np.save(f"{args.out_prefix}.J_bins.npy", J_bins.cpu().numpy()); np.save(f"{args.out_prefix}.patterns.npy", patterns.cpu().numpy())
            np.save(f"{args.out_prefix}.U_t.npy", U_t.cpu().numpy()); np.save(f"{args.out_prefix}.S.npy", S.cpu().numpy())
            meta={'edges':edges.tolist(),'fate_index':int(args.fate_index),'fate_name':(names[args.fate_index] if names and args.fate_index<len(names) else None)}
            with open(f"{args.out_prefix}.meta.json", 'w', encoding='utf-8') as f: json.dump(meta, f, indent=2)
            print(f"Saved fate-conditioned archetypes to {args.out_prefix}.*")
        else:
            print("No CellRank fate probabilities found; skipping fate-conditioned archetypes.")
if __name__=='__main__': main()
