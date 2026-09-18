import matplotlib

matplotlib.use("Agg")  # sin display; solo generamos figuras para loguear en MLflow
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, y_pred, y_predproba) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_predproba)),
    }


def roc_curve_figure(y_true, y_predproba, model_name: str, computed_metrics: dict):
    fig, ax = plt.subplots()
    fpr, tpr, _ = roc_curve(y_true, y_predproba)
    ax.plot(fpr, tpr, label=f"{model_name} AUC={computed_metrics['auc']:.2f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curva ROC-AUC")
    ax.legend(loc="lower right")
    return fig


def pr_curve_figure(y_true, y_predproba, model_name: str):
    fig, ax = plt.subplots()
    precision, recall, _ = precision_recall_curve(y_true, y_predproba)
    ax.plot(recall, precision, label=model_name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall")
    ax.legend(loc="lower left")
    return fig
