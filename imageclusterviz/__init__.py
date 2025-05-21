from importlib.metadata import version as _v
from .legacy_script import run_pipeline, create_feature_dict, get_clustered_data

__all__ = ["run_pipeline", "create_feature_dict", "get_clustered_data"]
__version__ = _v(__package__ or "imageclusterviz")
