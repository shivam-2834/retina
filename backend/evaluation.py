"""Evaluation utilities for independent labelled validation/test data.

No clinical performance numbers are hard-coded. Metrics must be computed from
held-out labelled data from the intended population.
"""
from collections import Counter

SEVERITY_LEVEL = {"No DR": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Proliferative DR": 4}


def referable_metrics(y_true, y_pred):
    true = [SEVERITY_LEVEL[str(x)] >= 2 for x in y_true]
    pred = [SEVERITY_LEVEL.get(str(x), -1) >= 2 for x in y_pred]
    tp = sum(a and b for a, b in zip(true, pred))
    tn = sum((not a) and (not b) for a, b in zip(true, pred))
    fp = sum((not a) and b for a, b in zip(true, pred))
    fn = sum(a and (not b) for a, b in zip(true, pred))
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {
        "n": len(true),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": (tp + tn) / len(true) if true else None,
    }


def class_counts(labels):
    return dict(Counter(str(x) for x in labels))
