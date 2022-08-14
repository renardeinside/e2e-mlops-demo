import json
from uuid import uuid4

from kafka import KafkaConsumer

from e2e_mlops_demo.models import PredictionInfo
from e2e_mlops_demo.reporting import KafkaReporter


def test_reporter(kafka_fixture):
    topic_name = f"test-topic-{uuid4()}"
    kafka_reporter = KafkaReporter(configs={"bootstrap_servers": kafka_fixture, "acks": "all"}, topic=topic_name)
    kafka_reporter.report("prediction", PredictionInfo(value=1, model_version=1))
    consumer = KafkaConsumer(
        topic_name,
        bootstrap_servers=[kafka_fixture],
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=False,
    )
    records = consumer.poll(10 * 1000)
    assert len(records) == 1
