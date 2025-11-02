
import torch, numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph
def build_knn_graph(X, k=10, mode='distance'):
    A=kneighbors_graph(X, n_neighbors=k, mode=mode, include_self=False); A=0.5*(A+A.T); return A
def graph_laplacian(A, normalized=True):
    L=csgraph.laplacian(A, normed=normalized); return torch.from_numpy(L.toarray()).float()
