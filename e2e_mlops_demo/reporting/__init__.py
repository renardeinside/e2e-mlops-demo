import datetime as dt
from abc import ABC, abstractmethod
from logging import Logger
from typing import Dict, Any

from kafka import KafkaProducer
from pydantic import BaseModel


class PayloadWithEventType(BaseModel):
    event_type: str
    payload: BaseModel


class Reporter(ABC):
    @abstractmethod
    def report(self, event_type: str, payload: BaseModel):
        """"""

    @staticmethod
    def _serialize_payload(event_type: str, payload: BaseModel) -> bytes:
        _wrapped = PayloadWithEventType(event_type=event_type, payload=payload)
        return _wrapped.json().encode("utf-8")


class KafkaReporter(Reporter):
    def __init__(self, configs: Dict[str, Any], topic: str):
        self._producer = KafkaProducer(**configs)
        self._topic = topic

    def report(self, event_type: str, payload: BaseModel):
        self._producer.send(self._topic, self._serialize_payload(event_type, payload))


class InMemoryReporter(Reporter):
    def __init__(self):
        self.storage = []

    def report(self, event_type: str, payload: BaseModel):
        self.storage.append(
            {
                "timestamp": dt.datetime.now(),
                "payload": self._serialize_payload(event_type, payload),
            }
        )


class LoggingReporter(Reporter):
    def __init__(self, logger: Logger):
        self._logger = logger

    def report(self, event_type: str, payload: BaseModel):
        self._logger.info(f"{self._serialize_payload(event_type, payload)}")
