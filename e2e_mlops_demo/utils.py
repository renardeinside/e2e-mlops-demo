from typing import Any

from e2e_mlops_demo.models import DatabricksApiInfo, MlflowInfo


class EnvironmentInfoProvider:  # pragma: no cover
    @staticmethod
    def get_databricks_api_info(dbutils: Any) -> DatabricksApiInfo:
        host = (
            dbutils.notebook.entry_point.getDbutils()
            .notebook()
            .getContext()
            .apiUrl()
            .getOrElse(None)
        )
        token = (
            dbutils.notebook.entry_point.getDbutils()
            .notebook()
            .getContext()
            .apiToken()
            .getOrElse(None)
        )
        return DatabricksApiInfo(host=host, token=token)

    @staticmethod
    def get_mlflow_info() -> MlflowInfo:
        return MlflowInfo(tracking_uri="databricks", registry_uri="databricks")
