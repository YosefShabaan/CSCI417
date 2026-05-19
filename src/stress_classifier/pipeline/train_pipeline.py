from __future__ import annotations

from pathlib import Path
import joblib
import mlflow

from stress_classifier.components.data_processing import (
    load_data,
    split_features_target,
    train_test_split_data,
)
from stress_classifier.components.model_evaluation import evaluate_model
from stress_classifier.components.model_trainer import build_pipeline, train_model
from stress_classifier.utils.common import create_dir, read_yaml


CONFIG_PATH = "configs/config.yaml"


def run_training(config_path: str = CONFIG_PATH) -> dict:
    config = read_yaml(config_path)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    data_cfg = config["data"]
    model_cfg = config["model"]
    artifacts_cfg = config["artifacts"]
    random_state = config["project"]["random_state"]

    df = load_data(data_cfg["input_path"])
    X, y = split_features_target(df, data_cfg["target_column"])

    X_train, X_test, y_train, y_test = train_test_split_data(
        X,
        y,
        test_size=data_cfg["test_size"],
        random_state=random_state,
    )

    model_pipeline = build_pipeline(X_train, model_cfg["params"])

    with mlflow.start_run():
        trained_model = train_model(model_pipeline, X_train, y_train)
        metrics = evaluate_model(trained_model, X_test, y_test)

        mlflow.log_param("model_type", model_cfg["type"])
        mlflow.log_params(model_cfg["params"])
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(trained_model, artifact_path="model")

        model_dir = Path(artifacts_cfg["model_dir"])
        create_dir(model_dir)
        model_path = model_dir / artifacts_cfg["model_filename"]
        joblib.dump(trained_model, model_path)

    return metrics


if __name__ == "__main__":
    results = run_training()
    print("Training complete. Metrics:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
