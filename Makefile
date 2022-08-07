export SPARK_CONF_DIR=$(PWD)/conf/local

test:
	pytest tests/unit $(ARGS)