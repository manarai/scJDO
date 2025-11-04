import pandas as pd, numpy as np
from sklearn.neighbors import NearestNeighbors

def build_cluster_adj(means, pvals, alpha=0.05, lr_weights=None):
    # means/pvals: DataFrames indexed by ['time','sender','receiver','ligand','receptor']
    key = ['time','sender','receiver','ligand','receptor']
    df = means.reset_index().merge(pvals.reset_index(), on=key, suffixes=('_mean','_p'))
    if lr_weights is not None:
        df = df.merge(lr_weights, on=['ligand','receptor'], how='left').assign(w_lr=lambda x: x.w_lr.fillna(1.0))
    else:
        df['w_lr'] = 1.0
    df['edge'] = df['value_mean'] * (df['value_p'] <= alpha).astype(float) * df['w_lr']
    A = df.groupby(['time','sender','receiver'], as_index=False)['edge'].sum().rename(columns={'edge':'A'})
    return A  # cluster-level adjacency over time

def knn_mask(X, k=20):
    nbrs = NearestNeighbors(n_neighbors=min(k+1, len(X))).fit(X)
    ind = nbrs.kneighbors(return_distance=False)
    M = np.zeros((len(X), len(X)), bool)
    for i, row in enumerate(ind):
        M[i, row[row != i]] = True
    return M

def lift_to_cells(A_t_df, cell_meta, knn=None):
    # A_t_df: ['sender','receiver','A'] for one time bin
    # cell_meta: DataFrame with index=cell_id and columns ['cluster']
    clusters = cell_meta['cluster']
    idx_by_c = {c: cell_meta.index[clusters == c].to_numpy() for c in clusters.unique()}
    n = len(cell_meta)
    W = np.zeros((n, n), float)
    for _, r in A_t_df.iterrows():
        S = idx_by_c.get(r.sender, [])
        R = idx_by_c.get(r.receiver, [])
        if len(S) and len(R):
            W[np.ix_(S, R)] = r.A
    if knn is not None:
        W *= knn
    row_sum = W.sum(axis=1, keepdims=True) + 1e-12
    return W / row_sum

