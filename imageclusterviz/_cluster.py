import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


__all__ = ["cluster_kmeans", "cluster_gmm", "auto_k"]


def cluster_kmeans(vecs: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Return integer labels from a k-means fit predict."""
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    return km.fit_predict(vecs)


def cluster_gmm(vecs: np.ndarray, k: int, *, seed: int = 0) -> np.ndarray:
    gmm = GaussianMixture(n_components=k, random_state=seed)
    return gmm.fit_predict(vecs)


def auto_k(*args, **kwargs):
    """Not implemented yet."""
    raise NotImplementedError("auto_k() is planned for a future release")
