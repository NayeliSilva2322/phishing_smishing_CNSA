from unittest.mock import MagicMock, patch

from src.mlflow_tracking import log_run


@patch("src.mlflow_tracking.mlflow")
def test_log_run_logs_params_metrics_and_model(mock_mlflow):
    mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()
    mock_sklearn = MagicMock()
    mock_mlflow.sklearn = mock_sklearn

    dummy_model = object()
    dummy_figure = MagicMock()

    log_run(
        run_name="xgboost-text",
        params={"n_estimators": 100},
        metrics={"accuracy": 0.9},
        figures={"roc_curve.png": dummy_figure},
        model=dummy_model,
        flavor="sklearn",
        registered_model_name="phishing-xgboost-text",
    )

    mock_mlflow.log_params.assert_called_once_with({"n_estimators": 100})
    mock_mlflow.log_metrics.assert_called_once_with({"accuracy": 0.9})
    mock_mlflow.log_figure.assert_called_once_with(dummy_figure, "roc_curve.png")
    mock_sklearn.log_model.assert_called_once_with(
        dummy_model, artifact_path="model", registered_model_name="phishing-xgboost-text"
    )
