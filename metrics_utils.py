import numpy as np
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

MIN_FEATURES = 12
MIN_INSTANCES = 500


def encode_target(y, classes):
    """Map raw target labels to integer ids using the class list saved at training time."""
    mapping = {str(c): i for i, c in enumerate(classes)}
    ids = y.astype(str).map(mapping)
    unknown = sorted(set(y.astype(str)[ids.isna()]))
    return ids, unknown


def compute_metrics(y_true, y_pred, y_proba, n_classes):
    """y_* are integer class ids 0..n_classes-1. Binary -> binary averaging,
    multi-class -> weighted averaging and one-vs-rest AUC."""
    y_true = np.asarray(y_true)
    avg = "binary" if n_classes == 2 else "weighted"
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted",
                                labels=list(range(n_classes)))
    except ValueError:  # e.g. a class missing from the uploaded test file
        auc = float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
