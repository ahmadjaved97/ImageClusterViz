from importlib.metadata import version as _v
from .legacy_script import run_pipeline, create_feature_dict, get_clustered_data
from ._embed import embed_dir, embed_image, load_model
from ._cluster import cluster_kmeans, cluster_gmm, auto_k, cluster_dict
from ._vis import make_grid

__all__ = [
    "run_pipeline",
    "create_feature_dict",
    "get_clustered_data",
    "embed_image",
    "embed_dir",
    "load_model",
    "cluster_kmeans",
    "cluster_gmm",
    "auto_k",
    "cluster_dict",
    "make_grid",
]

__version__ = _v(__package__ or "imageclusterviz")
