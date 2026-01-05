PY ?= python3
NMAX ?= 200
CONFIG ?= configs/submit.json
NAME ?= baseline
SUBMISSIONS_DIR ?= submissions
GEN_ARGS ?=

.PHONY: help install-dev install-train test lint format build-fastcollide submit score

help:
	@echo "Targets:"
	@echo "  install-dev       pip install -e '.[dev]'"
	@echo "  install-train     pip install -e '.[train]'"
	@echo "  test              run unittests"
	@echo "  lint              ruff check ."
	@echo "  format            ruff format ."
	@echo "  build-fastcollide build C++ extension in-place"
	@echo "  submit            generate + strict-score + archive under submissions/"
	@echo "  score             score a submission (SUB=path)"

install-dev:
	$(PY) -m pip install -e ".[dev]"

install-train:
	$(PY) -m pip install -e ".[train]"

test:
	$(PY) -m unittest discover -s tests

lint:
	ruff check .

format:
	ruff format .

build-fastcollide:
	$(PY) scripts/build/build_fastcollide.py

submit:
	$(PY) scripts/submission/make_submit.py --config $(CONFIG) --nmax $(NMAX) --name $(NAME) --submissions-dir $(SUBMISSIONS_DIR) -- $(GEN_ARGS)

score:
	$(PY) scripts/evaluation/score_submission.py $(SUB) --pretty
