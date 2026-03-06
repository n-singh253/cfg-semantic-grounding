"""Common utilities for baseline training and defense."""

from src.baseline.common.data_preparation import prepare_training_data
from src.baseline.common.split_strategies import get_split_strategy

__all__ = ["prepare_training_data", "get_split_strategy"]
