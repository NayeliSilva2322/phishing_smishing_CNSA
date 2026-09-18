import numpy as np

from src.metrics import compute_metrics, pr_curve_figure, roc_curve_figure


def test_compute_metrics_perfect_predictions():
    y_true = [0, 1, 0, 1, 1]
    y_pred = [0, 1, 0, 1, 1]
    y_predproba = [0.05, 0.95, 0.1, 0.9, 0.99]

    result = compute_metrics(y_true, y_pred, y_predproba)

    assert result["accuracy"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["auc"] == 1.0


def test_compute_metrics_returns_expected_keys():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 0, 0, 1]
    y_predproba = [0.1, 0.4, 0.2, 0.8]

    result = compute_metrics(y_true, y_pred, y_predproba)

    assert set(result.keys()) == {"accuracy", "precision", "recall", "auc"}
    assert all(isinstance(v, float) for v in result.values())


def test_roc_curve_figure_returns_matplotlib_figure():
    y_true = [0, 1, 0, 1]
    y_predproba = [0.2, 0.8, 0.3, 0.7]
    m = compute_metrics(y_true, [0, 1, 0, 1], y_predproba)

    fig = roc_curve_figure(y_true, y_predproba, "test-model", m)

    assert fig is not None
    assert len(fig.axes) == 1


def test_pr_curve_figure_returns_matplotlib_figure():
    y_true = [0, 1, 0, 1]
    y_predproba = [0.2, 0.8, 0.3, 0.7]

    fig = pr_curve_figure(y_true, y_predproba, "test-model")

    assert fig is not None
    assert len(fig.axes) == 1
