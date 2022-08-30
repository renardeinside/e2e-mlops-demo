"""
This file configures the Python package with entrypoints used for future runs on Databricks.

Please follow the `entry_points` documentation for more details on how to configure the entrypoint:
* https://setuptools.pypa.io/en/latest/userguide/entry_point.html
"""

from setuptools import find_packages, setup
from e2e_mlops_demo import __version__

BASE_REQUIREMENTS = [
    # ml and training
    "hyperopt",
    "imbalanced-learn==0.9.1",
    "scikit-learn==1.1.2",
    "mlflow==1.27.0",
    "threadpoolctl==2.2.0",
    "xgboost==1.5.2",
    "PyYAML",  # reading configs
    "openml",  # loading source data
]

SERVING_REQUIREMENTS = [
    # serving and logical components
    "pydantic",
    "uvicorn[standard]",
    "fastapi",
    # monitoring
    "kafka-python",
]

DEV_REQUIREMENTS = [
    "setuptools",
    "wheel",
    "pyspark==3.2.2",
    "pyyaml",
    "pytest",
    "pytest-cov",
    "coverage[toml]",
    "pytest-docker",
    "dbx>=0.7,<0.8",
    "delta-spark",
    "pandas",
]

ALL_REQUIREMENTS = DEV_REQUIREMENTS + SERVING_REQUIREMENTS

setup(
    name="e2e_mlops_demo",
    packages=find_packages(exclude=["tests", "tests.*"]),
    setup_requires=["wheel"],
    install_requires=BASE_REQUIREMENTS,
    extras_require={"dev": DEV_REQUIREMENTS, "serving": SERVING_REQUIREMENTS, "all": ALL_REQUIREMENTS},
    entry_points={
        "console_scripts": [
            "loader = e2e_mlops_demo.tasks.dataset_loader_task:entrypoint",
            "builder = e2e_mlops_demo.tasks.model_builder_task:entrypoint",
            "serve = e2e_mlops_demo.serving.app:entrypoint",
        ]
    },
    version=__version__,
    description="",
    author="",
)
