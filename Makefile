# Self-contained build/test helpers for the MIB solution image.
#   make build        # build the offline Docker image
#   make docker-test  # build + run under the real offline scoring contract on INPUT
#   make run          # run the pipeline locally (no Docker) on INPUT
# Override INPUT / OUT / IMAGE on the command line, e.g. `make docker-test INPUT=/path/pdfs`.

HERE   := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PY     ?= python3
IMAGE  ?= mib-submission
INPUT  ?= $(HERE)/sample_pdfs
OUT    ?= $(HERE)/out
PREDS  := $(OUT)/predictions.jsonl

.PHONY: help build run docker-test clean

help:
	@echo "targets: build | run | docker-test | clean   (INPUT=$(INPUT))"

build:
	docker build -t $(IMAGE) $(HERE)

run:
	MIB_DEBUG=1 $(PY) $(HERE)/solution.py $(INPUT) $(PREDS)

docker-test: build
	mkdir -p $(OUT)
	docker run --rm \
	  --network none --cpus 4 --memory 8g --pids-limit 512 \
	  --read-only --tmpfs /tmp:rw,nosuid,nodev,size=2g \
	  --mount type=bind,src=$(INPUT),dst=/input,readonly \
	  --mount type=bind,src=$(OUT),dst=/output \
	  $(IMAGE) /input /output/predictions.jsonl

clean:
	rm -rf $(OUT)
