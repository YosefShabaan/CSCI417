from stress_classifier.pipeline.train_pipeline import run_training


if __name__ == "__main__":
    metrics = run_training()
    print(metrics)
