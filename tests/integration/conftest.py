import logging

import pytest
from kafka import KafkaClient


def check_kafka(bootstrap_servers: str):
    try:
        _client = KafkaClient(bootstrap_servers=bootstrap_servers)
        return True
    except Exception as _:
        return False


@pytest.fixture(scope="session")
def kafka_fixture(docker_ip, docker_services) -> str:
    # mute kafka logger a little bit
    kafka_logger = logging.getLogger("kafka")
    kafka_logger.setLevel(logging.ERROR)
    # `port_for` takes a container port and returns the corresponding host port
    port = docker_services.port_for("kafka", 9092)
    bootstrap_servers = f"localhost:{port}"
    docker_services.wait_until_responsive(timeout=120.0, pause=2, check=lambda: check_kafka(bootstrap_servers))
    return bootstrap_servers
