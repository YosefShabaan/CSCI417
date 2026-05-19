from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

from stress_classifier.utils.common import read_yaml


CONFIG_PATH = "configs/config.yaml"


def get_model_path(config_path: str = CONFIG_PATH) -> Path:
    config = read_yaml(config_path)
    artifacts_cfg = config["artifacts"]
    return Path(artifacts_cfg["model_dir"]) / artifacts_cfg["model_filename"]


def load_trained_model(config_path: str = CONFIG_PATH):
    model_path = get_model_path(config_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run training first with `python main.py`."
        )
    return joblib.load(model_path)


def predict_single(sample: dict, config_path: str = CONFIG_PATH) -> str:
    model = load_trained_model(config_path)
    input_df = pd.DataFrame([sample])
    prediction = model.predict(input_df)[0]
    return str(prediction)
