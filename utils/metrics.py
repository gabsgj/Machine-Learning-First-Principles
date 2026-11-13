"""
Machine Learning Evaluation Metrics from Scratch
=================================================
Vectorized NumPy implementations of classification, regression, and clustering metrics:
- Binary & Multi-Class Confusion Matrix
- Accuracy, Precision, Recall, F1-Score (Micro, Macro, Weighted)
- Comprehensive Classification Report
- Mean Squared Error (MSE), Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE), Coefficient of Determination (R²)
"""

import numpy as np
from typing import Optional, List, Dict, Union


def confusion_matrix_scratch(
    y_true: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List],
    labels: Optional[Union[np.ndarray, List]] = None,
) -> np.ndarray:
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    
    By definition a confusion matrix C is such that C[i, j] is equal to the number of
    observations known to be in group i and predicted to be in group j.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([yt, yp]))
    else:
        labels = np.asarray(labels)

    n_labels = len(labels)
    label_to_idx = {val: i for i, val in enumerate(labels)}
    cm = np.zeros((n_labels, n_labels), dtype=int)

    for true_val, pred_val in zip(yt, yp):
        if true_val in label_to_idx and pred_val in label_to_idx:
            cm[label_to_idx[true_val], label_to_idx[pred_val]] += 1

    return cm


def accuracy_score_scratch(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy classification score: (TP + TN) / Total"""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    return float(np.mean(yt == yp))


def precision_score_scratch(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: int = 1,
    average: str = "binary",
) -> float:
    """
    Compute the precision: TP / (TP + FP).
    Ability of classifier not to label as positive a sample that is negative.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    if average == "binary":
        tp = np.sum((yt == pos_label) & (yp == pos_label))
        fp = np.sum((yt != pos_label) & (yp == pos_label))
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    classes = np.unique(np.concatenate([yt, yp]))
    precisions = []
    supports = []

    for c in classes:
        tp = np.sum((yt == c) & (yp == c))
        fp = np.sum((yt != c) & (yp == c))
        p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        precisions.append(p)
        supports.append(np.sum(yt == c))

    if average == "macro":
        return float(np.mean(precisions))
    elif average == "weighted":
        total_support = np.sum(supports)
        return float(np.sum(np.array(precisions) * np.array(supports)) / total_support) if total_support > 0 else 0.0
    else:
        raise ValueError(f"Unsupported average mode: {average}")


def recall_score_scratch(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: int = 1,
    average: str = "binary",
) -> float:
    """
    Compute the recall (sensitivity): TP / (TP + FN).
    Ability of classifier to find all positive samples.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    if average == "binary":
        tp = np.sum((yt == pos_label) & (yp == pos_label))
        fn = np.sum((yt == pos_label) & (yp != pos_label))
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    classes = np.unique(np.concatenate([yt, yp]))
    recalls = []
    supports = []

    for c in classes:
        tp = np.sum((yt == c) & (yp == c))
        fn = np.sum((yt == c) & (yp != c))
        r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        recalls.append(r)
        supports.append(np.sum(yt == c))

    if average == "macro":
        return float(np.mean(recalls))
    elif average == "weighted":
        total_support = np.sum(supports)
        return float(np.sum(np.array(recalls) * np.array(supports)) / total_support) if total_support > 0 else 0.0
    else:
        raise ValueError(f"Unsupported average mode: {average}")


def f1_score_scratch(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: int = 1,
    average: str = "binary",
) -> float:
    """
    Compute F1-score: harmonic mean of precision and recall.
    Formula: 2 * (precision * recall) / (precision + recall)
    """
    if average == "binary":
        p = precision_score_scratch(y_true, y_pred, pos_label=pos_label, average="binary")
        r = recall_score_scratch(y_true, y_pred, pos_label=pos_label, average="binary")
        return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    classes = np.unique(np.concatenate([yt, yp]))
    f1s = []
    supports = []

    for c in classes:
        p = precision_score_scratch(yt, yp, pos_label=c, average="binary")
        r = recall_score_scratch(yt, yp, pos_label=c, average="binary")
        f = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        f1s.append(f)
        supports.append(np.sum(yt == c))

    if average == "macro":
        return float(np.mean(f1s))
    elif average == "weighted":
        total_support = np.sum(supports)
        return float(np.sum(np.array(f1s) * np.array(supports)) / total_support) if total_support > 0 else 0.0
    else:
        raise ValueError(f"Unsupported average mode: {average}")


def classification_report_scratch(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
) -> str:
    """
    Build a formatted text report showing main classification metrics.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    classes = np.unique(np.concatenate([yt, yp]))
    
    header = f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n"
    divider = "-" * 61 + "\n"
    lines = [header, divider]

    supports = []
    precisions = []
    recalls = []
    f1s = []

    for idx, c in enumerate(classes):
        name = target_names[idx] if (target_names and idx < len(target_names)) else str(c)
        p = precision_score_scratch(yt, yp, pos_label=c, average="binary")
        r = recall_score_scratch(yt, yp, pos_label=c, average="binary")
        f = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        s = int(np.sum(yt == c))

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        supports.append(s)

        lines.append(f"{name:<15} {p:<12.4f} {r:<12.4f} {f:<12.4f} {s:<10}\n")

    lines.append(divider)
    acc = accuracy_score_scratch(yt, yp)
    lines.append(f"{'accuracy':<15} {'':<12} {'':<12} {acc:<12.4f} {sum(supports):<10}\n")

    # Macro avg
    lines.append(
        f"{'macro avg':<15} {np.mean(precisions):<12.4f} {np.mean(recalls):<12.4f} {np.mean(f1s):<12.4f} {sum(supports):<10}\n"
    )

    # Weighted avg
    total_s = sum(supports)
    w_p = sum(np.array(precisions) * np.array(supports)) / total_s
    w_r = sum(np.array(recalls) * np.array(supports)) / total_s
    w_f = sum(np.array(f1s) * np.array(supports)) / total_s
    lines.append(f"{'weighted avg':<15} {w_p:<12.4f} {w_r:<12.4f} {w_f:<12.4f} {total_s:<10}\n")

    return "".join(lines)


def mean_squared_error_scratch(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error (MSE): 1/n * sum((y - y_hat)^2)"""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean((yt - yp) ** 2))


def root_mean_squared_error_scratch(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error (RMSE): sqrt(MSE)"""
    return float(np.sqrt(mean_squared_error_scratch(y_true, y_pred)))


def mean_absolute_error_scratch(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error (MAE): 1/n * sum(|y - y_hat|)"""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yt - yp)))


def r2_score_scratch(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R² (coefficient of determination) regression score function.
    Formula: 1 - sum((y - y_pred)^2) / sum((y - y_mean)^2)
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - (ss_res / ss_tot))
