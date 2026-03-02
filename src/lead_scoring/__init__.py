"""Lead scoring package."""

from .config import TARGET_COLUMN, FEATURE_COLUMNS
from .inference import load_artifacts, predict_one

__all__ = ["TARGET_COLUMN", "FEATURE_COLUMNS", "load_artifacts", "predict_one"]
