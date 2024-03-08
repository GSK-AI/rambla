from typing import Dict

import numpy as np

from rambla.utils.misc import squeeze_dict_dim


def compute_recall_from_confmat(confusion_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    """Computes `recall` from a confusion matrix.

    AKA True Positive Rate (TPR).

    For binary case: = TP / (TP + FN)

    Parameters
    ----------
        confusion_matrix (np.ndarray): A binary or multi-class confusion matrix.

    Returns
    -------
        Dict[str, np.ndarray]: `recall` computed three different ways:
            micro: Sum statistics over all classes.
            macro: Calculate statistics for each class and average them.
            per_class: Calculates statistic for each class and applies no reduction.
    """
    n_targets_per_class = np.sum(confusion_matrix, axis=1)
    tp_per_class = np.diag(confusion_matrix)
    per_class_recall = tp_per_class / n_targets_per_class

    zero_by_zero_division = np.logical_and(tp_per_class == 0, n_targets_per_class == 0)

    per_class_recall[zero_by_zero_division] = 1

    return {
        "per_class": per_class_recall,
        "macro": np.mean(per_class_recall),
        "micro": np.sum(tp_per_class) / np.sum(n_targets_per_class),
    }


def compute_precision_from_confmat(
    confusion_matrix: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Computes `precision` from a confusion matrix

    For binary case: = TP / (TP + FP)

    Parameters
    ----------
        confusion_matrix (np.ndarray): A binary or multi-class confusion matrix.

    Returns
    -------
        Dict[str, np.ndarray]: `precision` computed three different ways:
            micro: Sum statistics over all classes.
            macro: Calculate statistics for each class and average them.
            per_class: Calculates statistic for each class and applies no reduction.
    """
    n_preds_per_class = np.sum(confusion_matrix, axis=0)
    tp_per_class = np.diag(confusion_matrix)
    per_class_precision = tp_per_class / n_preds_per_class

    zero_by_zero_division = np.logical_and(tp_per_class == 0, n_preds_per_class == 0)

    per_class_precision[zero_by_zero_division] = 1

    return {
        "per_class": per_class_precision,
        "macro": np.mean(per_class_precision),
        "micro": np.sum(tp_per_class) / np.sum(n_preds_per_class),
    }


def compute_f1_from_confmat(confusion_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    """Computes `f1` from a confusion matrix

    For binary case: 2 * (precision * recall) / (precision + recall)

    Parameters
    ----------
        confusion_matrix (np.ndarray): A binary or multi-class confusion matrix.

    Returns
    -------
        Dict[str, np.ndarray]: `f1` computed three different ways:
            micro: Sum statistics over all classes.
            macro: Calculate statistics for each class and average them.
            per_class: Calculates statistic for each class and applies no reduction.
    """
    recall = compute_recall_from_confmat(confusion_matrix)
    precision = compute_precision_from_confmat(confusion_matrix)
    return compute_f1_from_precision_and_recall(recall, precision)


def compute_f1_from_precision_and_recall(
    recall: Dict[str, np.ndarray],
    precision: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Computes `F1` from recall and precision tensors.

    Parameters
    ----------
        recall (Dict[str, np.ndarray]):
            Contains tensors for: micro, macro and per-class recall.
        precision (Dict[str, np.ndarray]):
            Contains tensors for: micro, macro and per-class precision.

    Returns
    -------
        Dict[str, np.ndarray]: `F1` computed three different ways:
            micro: Sum statistics over all classes.
            macro: Calculate statistics for each class and average them.
            per_class: Calculates statistic for each class and applies no reduction.
    """
    results = {}
    for key in ["micro", "per_class"]:
        results[key] = (
            2 * (recall[key] * precision[key]) / (recall[key] + precision[key])
        )

    results["macro"] = np.mean(results["per_class"])
    return results


def compute_metrics_from_confmat(
    confusion_matrix: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Helper function for computing metrics from confusion matrix.

    Parameters
    ----------
        confusion_matrix (np.ndarray):
            Binary or multi-class confusion matrix.

        include_mcc (bool, optional):
            Whether to include MCC in the output metrics. Defaults to False.

    Returns
    -------
        Dict[str, np.ndarray]:
            Dictionary containing metrics computed from the confusion matrix.
    """
    return {
        "recall": compute_recall_from_confmat(confusion_matrix),
        "precision": compute_precision_from_confmat(confusion_matrix),
        "f1": compute_f1_from_confmat(confusion_matrix),
    }


def filter_confmat(
    confusion_matrix: np.ndarray,
    label_encoder: dict[str, int],
    to_exclude: list[str],
    strict: bool = False,
) -> tuple[np.ndarray, dict[str, int]]:
    """Returns a new confmat with just the entries not included in `to_exclude`."""
    if strict:
        indices_to_exclude = [label_encoder[key] for key in to_exclude]
    else:
        indices_to_exclude = [label_encoder.get(key, -1) for key in to_exclude]

    indices_to_include = [
        idx for idx in range(confusion_matrix.shape[0]) if idx not in indices_to_exclude
    ]
    filtered_confusion_matrix = confusion_matrix[indices_to_include, :][
        :, indices_to_include
    ]

    # Creating a label encoder for the new reduced confusion matrix
    filtered_label_encoder_keys = [
        k for k, v in label_encoder.items() if v in indices_to_include
    ]
    sorted_keys = sorted(filtered_label_encoder_keys, key=lambda x: label_encoder[x])
    new_label_encoder = dict(zip(sorted_keys, range(len(sorted_keys))))
    return filtered_confusion_matrix, new_label_encoder


def get_metrics_helper(
    confmat: np.ndarray,
    label_encoder: dict[str, int],
    to_exclude: list[str],
) -> tuple[dict, dict[str, int]]:
    """Filters the confmat and computes the metrics."""
    new_confmat, new_label_encoder = filter_confmat(confmat, label_encoder, to_exclude)
    new_metrics = compute_metrics_from_confmat(new_confmat)
    flattened_metrics = squeeze_dict_dim(new_metrics)
    return flattened_metrics, new_label_encoder


def compute_class_counts_from_confmat(
    confmat: np.ndarray,
    label_encoder: dict[str, int],
) -> dict[str, int | float]:
    """Computes totals and proportions of each class for predictions and targets."""
    preds = confmat.sum(axis=0)
    targets = confmat.sum(axis=1)
    n_total = confmat.sum()

    out = {}
    for key, value in label_encoder.items():
        out[f"n_pred_{key}"] = preds[value]
        out[f"prop_pred_{key}"] = preds[value] / n_total

        out[f"n_target_{key}"] = targets[value]
        out[f"prop_target_{key}"] = targets[value] / n_total

    for k, v in out.items():
        if isinstance(v, np.ndarray):
            out[k] = v.item()
    return out
