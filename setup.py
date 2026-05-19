from setuptools import find_packages, setup

setup(
    name="stress-classifier",
    version="0.1.0",
    author="CSCI417",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pandas",
        "scikit-learn",
        "mlflow",
        "pyyaml",
        "joblib",
        "numpy",
    ],
)
