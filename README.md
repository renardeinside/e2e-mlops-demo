# E2E MLOps demo with Databricks

This is the source code for the E2E MLOps on Databricks blogpost series.

- [Part #1. Model training](https://polarpersonal.medium.com/end-to-end-mlops-with-azure-databricks-azure-aks-and-azure-eventhubs-part-1-model-training-f191139a36db)
- [In-progress] Part #2 - Model serving
- [TBD] Part #3 - Model monitoring

## Local environment setup

For local operations (e.g. development / running tests), you'll need Python 3.9.X, `pip` and `conda` for package
management.

1. Instantiate a local Python environment via a tool of your choice. This example is based on `conda`, but you can use
   any environment management tool:

```bash
conda create -n e2e_mlops_demo python=3.9
conda activate e2e_mlops_demo
```

2. If you don't have JDK installed on your local machine, install it (in this example we use `conda`-based
   installation):

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
In the `tests/unit/conftest.py` you'll also find useful testing primitives, such as local Spark instance with Delta
support, local MLflow and DBUtils fixture.

## Model training scalability testing results

General considerations:

- Tests performed on Azure Databricks Jobs cluster with Databricks ML Runtime `11.1.x-cpu-ml-scala2.12`.
- Cluster nodes were pre-warmed to avoid VM startup time effects.
- Average worker node utilization was around **85-90%** during the active training phase.
- Since `num_estimators` was also one of the parameters in XGBoost, it might happen that some tasks were heavier than
  others. However, most of the training results chose `num_estimators` in range from 80 to 100.

Configuration details:

| Parameter                       | Value and comments                                                                                                  |
|---------------------------------|---------------------------------------------------------------------------------------------------------------------|
| Dataset size                    | 29 double columns in X, train rows: `454881`, test rows: `56962`                                                    |
| Driver and worker type          | `Standard_F8s_v2` (8 cores, no GPU, [details](https://docs.microsoft.com/en-us/azure/virtual-machines/fsv2-series)) |
| Additional Spark Configurations | `spark.task.cpus=8` so XGBoost could fully utilise all cores (`n_jobs=-1`)                                          |
| Total model evaluations         | 100                                                                                                                 |

Results:

| Number of workers | Time taken (human-readable) | Time taken (seconds) |
|-------------------|-----------------------------|----------------------|
| 14                | 9m 54s                      | 585                  |
| 10                | 11m 54s                     | 714                  |
| 6                 | 16m 55s                     | 1015                 |
| 2                 | 43m 49s                     | 2629                 |
| 1                 | 1h 25m 33s                  | 5133                 |


## CI/CD setup

- To trigger the CI pipeline, simply push your code to the repository.
- To trigger the release pipeline, get the current version from the `e2e_mlops_demo/__init__.py` file and tag the
  current code version:

```
git tag -a v<new-version> -m "Release tag for version <new-version>"
git push origin --tags
```
