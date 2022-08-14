from http import HTTPStatus

from fastapi.testclient import TestClient

from e2e_mlops_demo.models import PredictionInfo
from e2e_mlops_demo.reporting import InMemoryReporter
from e2e_mlops_demo.serving.app import get_app


def test_serving(model_fixture: str, dataset_fixture):
    test_reporter = InMemoryReporter()
    app = get_app(model_fixture, test_reporter)
    client = TestClient(app)

    response = client.post(
        "/v1/invocations",
        json=dataset_fixture.head(1).T.squeeze().to_dict(),
    )
    assert response.status_code == HTTPStatus.OK
    _info = PredictionInfo(**response.json())
    assert isinstance(_info.value, int)
    assert _info.model_version == 1
    assert len(test_reporter.storage) > 1
