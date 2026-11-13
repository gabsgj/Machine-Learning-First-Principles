"""
ML Core Utilities Package
=========================
Foundational implementations of data preprocessing pipelines, evaluation metrics,
and visualization utilities built from mathematical first principles.
"""

from .preprocessing import (
    StandardScalerScratch,
    MinMaxScalerScratch,
    SimpleImputerScratch,
    LabelEncoderScratch,
    OneHotEncoderScratch,
    train_test_split_scratch,
)
from .metrics import (
    confusion_matrix_scratch,
    accuracy_score_scratch,
    precision_score_scratch,
    recall_score_scratch,
    f1_score_scratch,
    mean_squared_error_scratch,
    root_mean_squared_error_scratch,
    mean_absolute_error_scratch,
    r2_score_scratch,
    classification_report_scratch,
)

__all__ = [
    "StandardScalerScratch",
    "MinMaxScalerScratch",
    "SimpleImputerScratch",
    "LabelEncoderScratch",
    "OneHotEncoderScratch",
    "train_test_split_scratch",
    "confusion_matrix_scratch",
    "accuracy_score_scratch",
    "precision_score_scratch",
    "recall_score_scratch",
    "f1_score_scratch",
    "mean_squared_error_scratch",
    "root_mean_squared_error_scratch",
    "mean_absolute_error_scratch",
    "r2_score_scratch",
    "classification_report_scratch",
]
