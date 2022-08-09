export SPARK_CONF_DIR=$(PWD)/conf/local
PYTHON_ENTRYPOINT=$(shell which python)
export PYSPARK_PYTHON=$(PYTHON_ENTRYPOINT)
export PYSPARK_DRIVER_PYTHON=$(PYTHON_ENTRYPOINT)

test:
	pytest tests/unit $(ARGS)