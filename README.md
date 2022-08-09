# E2E MLOps demo with Databricks

This is the source code for this blogpost series.


## Local environment setup

For local operations (e.g. deveopment / running tests), you'll need Python 3.9.X, `pip` and `conda` for package management.

1. Instantiate a local Python environment via a tool of your choice. This example is based on `conda`, but you can use any environment management tool:
```bash
conda create -n e2e_mlops_demo python=3.9
conda activate e2e_mlops_demo
```

2. If you don't have JDK installed on your local machine, install it (in this example we use `conda`-based installation):
```bash
conda install -c conda-forge openjdk=11.0.15
```

3. Install project in a dev mode (this will also install dev requirements):
```bash
pip install -e ".[dev]"
```

## Running unit tests

For unit testing, please use `pytest`:
```
pytest tests/unit --cov
```

Please check the directory `tests/unit` for more details on how to use unit tests.
In the `tests/unit/conftest.py` you'll also find useful testing primitives, such as local Spark instance with Delta support, local MLflow and DBUtils fixture.


## CI/CD setup

- To trigger the CI pipeline, simply push your code to the repository.
- To trigger the release pipeline, get the current version from the `e2e_mlops_demo/__init__.py` file and tag the current code version:
```
git tag -a v<new-version> -m "Release tag for version <new-version>"
git push origin --tags
```
