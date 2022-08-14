from uuid import uuid4

from e2e_mlops_demo.models import PredictionInfo
from e2e_mlops_demo.reporting import KafkaReporter


def test_reporter(kafka_fixture, spark):
    topic_name = f"test-topic-{uuid4()}"
    kafka_reporter = KafkaReporter(configs={"bootstrap_servers": kafka_fixture, "acks": "all"}, topic=topic_name)
    kafka_reporter.report("prediction", PredictionInfo(value=1, model_version=1))
