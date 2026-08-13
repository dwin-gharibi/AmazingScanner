.PHONY: help setup data train eval figures test app label docker docker-run k8s clean lint

PY ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || command -v python3 || echo python)
PIP ?= $(PY) -m pip
RUNS ?= runs

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:
	@# Colab and some Docker images ship a python whose `ensurepip` is removed,
	@# so venv creation fails. That is not a reason to stop: installing into the
	@# active environment is the correct thing to do there anyway.
	@python3 -m venv .venv 2>/dev/null && echo "==> virtualenv at .venv" || \
		echo "==> no virtualenv (this python cannot build one) — installing into the current environment"
	@P=$$([ -x .venv/bin/python ] && echo .venv/bin/python || command -v python3); \
	echo "==> using $$P"; \
	$$P -m pip install --upgrade pip && \
	$$P -m pip install --index-url https://download.pytorch.org/whl/cpu torch && \
	$$P -m pip install -r requirements.txt && \
	$$P -m pip install -e .
	@echo "Install the OCR engine too:  sudo apt-get install tesseract-ocr"
	@echo "Then:  make data   (downloads ~1.8 GB of corpora on first run)"

fetch:
	$(PY) -m docscanner.data.fetch

data:
	$(PY) -m docscanner.data.prepare --all

train:
	$(PY) -m docscanner.all_in_one --out $(RUNS)

train-enhance:
	$(PY) -m docscanner.engine.train_enhance --name enhance_main --minutes 50

train-corners:
	$(PY) -m docscanner.engine.train_corners --approach heatmap --minutes 45
	$(PY) -m docscanner.engine.train_corners --approach regression --minutes 30

eval:
	$(PY) -m docscanner.eval.evaluate --all

figures:
	$(PY) -m docscanner.eval.figures --all

export-models:
	$(PY) -m docscanner.engine.export_models

check-labels:
	$(PY) -m docscanner.data.check_labels

test:
	$(PY) -m pytest tests/ -q

app:
	$(PY) -m docscanner.app.gradio_app

label:
	$(PY) -m docscanner.app.labeler --out data/real/own

scan:
	$(PY) -m docscanner.pipeline.run scan $(IMG) -o outputs/pipeline

docker:
	docker build -t docscanner:latest .

docker-run:
	docker run --rm -p 7860:7860 -v $$(pwd)/models:/app/models:ro docscanner:latest

k8s:
	kubectl apply -f deploy/k8s/

clean:
	rm -rf outputs .pytest_cache **/__pycache__
