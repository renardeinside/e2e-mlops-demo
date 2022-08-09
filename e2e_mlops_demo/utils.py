from typing import Optional

from e2e_mlops_demo.models import MlflowInfo


class EnvironmentInfoProvider:  # pragma: no cover
    @staticmethod
    def get_mlflow_info() -> Optional[MlflowInfo]:
        """This method is only used in local unit tests"""
        return None
