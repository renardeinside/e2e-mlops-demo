import logging
from logging.config import dictConfig

import uvicorn

LOGGING_FORMAT = "[%(asctime)s][%(levelname)s][%(name)s][%(funcName)s][%(message)s]"

LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "default": {
            "()": uvicorn.logging.DefaultFormatter,
            "format": LOGGING_FORMAT,
        },
        "access": {
            "()": uvicorn.logging.AccessFormatter,
            "format": LOGGING_FORMAT,
        },
        "basic": {
            "format": LOGGING_FORMAT,
        },
    },
    "handlers": {
        "console": {
            "formatter": "basic",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "level": logging.INFO,
        }
    },
    "loggers": {
        "serving_app": {
            "handlers": ["console"],
            "level": logging.INFO,
        },
        "uvicorn": {"handlers": ["console"], "level": logging.INFO},
    },
}


def configure_logger() -> logging.Logger:
    dictConfig(LOGGING_CONFIG)
    return logging.getLogger("serving_app")
