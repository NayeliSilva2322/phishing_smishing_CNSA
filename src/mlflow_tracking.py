import mlflow

from . import config


def setup_mlflow():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)


def log_run(
    run_name: str,
    params: dict,
    metrics: dict,
    figures: dict,
    model,
    flavor: str,
    registered_model_name: str,
    input_example=None,
):
    """Loguea una corrida de evaluacion (test) y registra el modelo en el Model Registry.

    flavor: uno de "sklearn", "xgboost", "tensorflow", "transformers".
    figures: dict {nombre_de_archivo.png: matplotlib.figure.Figure}
    """
    log_model_fn = getattr(mlflow, flavor).log_model

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        for filename, fig in figures.items():
            mlflow.log_figure(fig, filename)

        log_kwargs = {
            "artifact_path": "model",
            "registered_model_name": registered_model_name,
        }
        if input_example is not None:
            log_kwargs["input_example"] = input_example

        log_model_fn(model, **log_kwargs)
