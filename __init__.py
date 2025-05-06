# __init__.py
"""
Package init – handy star‑import in notebooks:

    from my_bp_pkg import *
"""
from .data import load_data, DEFAULT_FEATURES
from .model import MODEL_REGISTRY
from .evaluation import ensemble_search, threshold_sweep

__all__ = [
    "load_data",
    "DEFAULT_FEATURES",
    "MODEL_REGISTRY",
    "ensemble_search",
    "threshold_sweep",
]
