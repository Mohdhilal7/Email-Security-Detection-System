import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def evaluate_predictions(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }


def save_metrics(metrics, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(path, index=False)


def save_json(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
