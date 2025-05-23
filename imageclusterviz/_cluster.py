from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from typing import Sequence, Dict, List
from collections import defaultdict


__all__ = ["cluster_kmeans", "cluster_gmm", "auto_k", "cluster_dict"]


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


def cluster_dict(
    paths: Sequence[str | Path], labels: Sequence[int]
) -> Dict[int, List[Path]]:
    """
    Pair each image path with it's cluster label and return a dictionary containing
    {label1: [Path1, Path2, ...], label2: [Path3, Path4, ...]}

    Parameters
    ----------
    paths : list of image paths, usually the same returned by embed_dir()
    labels : cluster labels produced from the corresponding vectors

    Raises
    ------
    ValueError if lengths mismatch.

    """

    if len(paths) != len(labels):
        raise ValueError("Label length does not match the number of images")

    buckets: Dict[int, List[Path]] = defaultdict(list)
    for path, label in zip(paths, labels):
        buckets[int(label)].append(str(Path(path).resolve()))

    return buckets
