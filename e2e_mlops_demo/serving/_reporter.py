import logging
import os

from e2e_mlops_demo.reporting import KafkaReporter

logger = logging.getLogger("serving-app")


def get_kafka_reporter() -> KafkaReporter:
    configs = {
        "bootstrap_servers": os.environ["KAFKA_BOOTSTRAP_SERVERS"],
        "client_id": "kafka-reporter",
    }

    if "EH_CONNECTION_STRING" in os.environ:
        sasl_configs = {
            "security_protocol": "SASL_SSL",
            "sasl_mechanism": "PLAIN",
            "sasl_plain_username": "$ConnectionString",
            "sasl_plain_password": os.environ["EH_CONNECTION_STRING"],
        }
        configs.update(sasl_configs)
    else:
        logging.info("EventHub auth information wasn't provided in the environment config")
    topic = os.environ.get("REPORTER_TOPIC", "reporter")
    reporter = KafkaReporter(configs=configs, topic=topic)
    return reporter
