import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score
)

def train_mst_clustering(X, n_clusters=5):
    dist_matrix = squareform(pdist(X, metric="euclidean"))
    mst = minimum_spanning_tree(dist_matrix).toarray()
    
    if n_clusters > 1:
        edges = []
        rows, cols = np.nonzero(mst)
        for r, c in zip(rows, cols):
            edges.append((mst[r, c], r, c))
        edges.sort(key=lambda x: x[0], reverse=True)
        
        for i in range(min(n_clusters - 1, len(edges))):
            weight, r, c = edges[i]
            mst[r, c] = 0.0
            mst[c, r] = 0.0
            
    n_components, labels = connected_components(mst, directed=False)
    return labels

def train_gmm(X, n_components=5, covariance_type="full"):
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=42
    )
    model.fit(X)
    labels = model.predict(X)
    return model, labels

def train_dbscan(X, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = model.fit_predict(X)
    return model, labels

def compute_k_distance(X, k=5):
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(X)
    distances, _ = neighbors.kneighbors(X)
    k_distances = np.sort(distances[:, k - 1])
    return k_distances

def compute_clustering_metrics(X, y_true, labels):
    mask = labels != -1
    if np.sum(mask) == 0 or len(np.unique(labels[mask])) <= 1:
        return {
            "ari": 0.0,
            "nmi": 0.0,
            "silhouette": -1.0,
            "n_clusters": int(len(np.unique(labels[mask]))),
            "n_noise": int(np.sum(~mask))
        }
        
    ari = adjusted_rand_score(y_true[mask], labels[mask])
    nmi = normalized_mutual_info_score(y_true[mask], labels[mask])
    
    if len(np.unique(labels[mask])) > 1:
        sample_size = min(5000, np.sum(mask))
        sil = silhouette_score(X[mask], labels[mask], sample_size=sample_size, random_state=42)
    else:
        sil = -1.0
        
    return {
        "ari": ari,
        "nmi": nmi,
        "silhouette": sil,
        "n_clusters": int(len(np.unique(labels[mask]))),
        "n_noise": int(np.sum(~mask))
    }
