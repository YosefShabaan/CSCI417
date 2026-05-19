from __future__ import annotations

import streamlit as st
import pandas as pd

from stress_classifier.pipeline.predict_pipeline import predict_single
from stress_classifier.utils.common import read_yaml


CONFIG_PATH = "configs/config.yaml"


@st.cache_data
def load_form_choices(config_path: str = CONFIG_PATH):
    cfg = read_yaml(config_path)
    df = pd.read_csv(cfg["data"]["input_path"])
    target_col = cfg["data"]["target_column"]
    feature_cols = [c for c in df.columns if c != target_col]

    choices = {}
    for col in feature_cols:
        unique_values = sorted(df[col].dropna().astype(str).unique().tolist())
        choices[col] = unique_values

    return feature_cols, choices


def main():
    st.set_page_config(page_title="Stress Level Classifier", page_icon="🧠", layout="wide")
    st.title("🧠 Student Stress Level Classifier")
    st.caption("Easy deployment UI: fill the survey-style fields and get predicted stress label.")

    feature_cols, choices = load_form_choices()

    with st.form("predict_form"):
        st.subheader("Input Student Details")
        values = {}

        cols = st.columns(2)
        for idx, feature in enumerate(feature_cols):
            with cols[idx % 2]:
                values[feature] = st.selectbox(feature, choices[feature], key=feature)

        submitted = st.form_submit_button("Predict Stress Level")

    if submitted:
        prediction = predict_single(values)
        st.success(f"Predicted Stress Label: **{prediction}**")


if __name__ == "__main__":
    main()
