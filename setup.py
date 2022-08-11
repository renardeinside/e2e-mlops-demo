"""
This file configures the Python package with entrypoints used for future runs on Databricks.

Please follow the `entry_points` documentation for more details on how to configure the entrypoint:
* https://setuptools.pypa.io/en/latest/userguide/entry_point.html
"""

from setuptools import find_packages, setup
from e2e_mlops_demo import __version__

PACKAGE_REQUIREMENTS = [
    "PyYAML",
    "openml",
    "hyperopt",
    "pydantic",
    "imbalanced-learn",
    "xgboost",
    "uvicorn[standard]",
    "fastapi",
]

DEV_REQUIREMENTS = [
    "setuptools",
    "wheel",
    "pyspark",
    "pyyaml",
    "pytest",
    "pytest-cov",
    "dbx",
    "mlflow",
    "delta-spark",
    "scikit-learn",
    "pandas",
]

setup(
    name="e2e_mlops_demo",
    packages=find_packages(exclude=["tests", "tests.*"]),
    setup_requires=["wheel"],
    install_requires=PACKAGE_REQUIREMENTS,
    extras_require={"dev": DEV_REQUIREMENTS},
    entry_points={
        "console_scripts": [
            "loader = e2e_mlops_demo.tasks.dataset_loader_task:entrypoint",
            "builder = e2e_mlops_demo.tasks.model_builder_task:entrypoint",
        ]
    },
    version=__version__,
    description="",
    author="",
)
