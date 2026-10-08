"""
AssurPrime Package - Insurance Premium Prediction
Crédit Agricole Assurances Hackathon (ENS Challenge Data)
"""

__version__ = "1.0.0"
__author__ = "AssurPrime Team"

# Expose core pipeline functions
try:
    from .data_loader import load_data_in_chunks, optimize_memory
    from .features import build_features, preprocess_pipeline
    from .models import train_lgbm_optuna, train_ensemble_cv
    from .predict import generate_submission
except ImportError:
    # Fallback to allow package import during modular updates
    pass

__all__ = [
    "load_data_in_chunks",
    "optimize_memory",
    "build_features",
    "preprocess_pipeline",
    "train_lgbm_optuna",
    "train_ensemble_cv",
    "generate_submission",
]
