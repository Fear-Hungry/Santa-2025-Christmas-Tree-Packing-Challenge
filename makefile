PY ?= python3
NMAX ?= 200
CONFIG ?= configs/submit.json
ENSEMBLE_CONFIG ?= configs/ensemble.json
NAME ?= baseline
SUBMISSIONS_DIR ?= submissions
GEN_ARGS ?=
OUT ?= submission.csv
SEED ?= 1
OVERLAP_MODE ?= kaggle
SEEDS ?= 1..3
JOBS ?= 1
TRAIN_META_ARGS ?=
TRAIN_HEATMAP_ARGS ?=
TRAIN_L2O_ARGS ?=
TRAIN_L2O_BC_ARGS ?=
SA_DATASET_ARGS ?=
SWEEP_ARGS ?=
SUB ?= submission.csv
DET_IN ?= submission.csv
DET_OUT ?= submission_detouched.csv
DET_SCALE ?= 1.01

.PHONY: help install-dev install-train test check lint lint-fix format format-check build-fastcollide submit generate-submission score \
	sweep-ensemble detouch-submission train-meta train-heatmap train-l2o train-l2o-bc collect-sa-dataset bench-fastcollide \
	pre-commit

help:
	@echo "Targets:"
	@echo "  install-dev       pip install -e '.[dev]'"
	@echo "  install-train     pip install -e '.[train]'"
	@echo "  test              run unittests"
	@echo "  check             run lint + tests"
	@echo "  lint              ruff check ."
	@echo "  lint-fix          ruff check . --fix"
	@echo "  format            ruff format ."
	@echo "  format-check      verify formatting"
	@echo "  build-fastcollide build C++ extension in-place"
	@echo "  generate-submission generate a submission.csv"
	@echo "  submit            generate + strict-score + archive under submissions/"
	@echo "  score             score a submission (SUB=path)"
	@echo "  sweep-ensemble    sweep recipes/seeds and ensemble per n"
	@echo "  detouch-submission post-process a submission CSV"
	@echo "  train-meta        train meta-init (JAX)"
	@echo "  train-heatmap     train heatmap meta (JAX)"
	@echo "  train-l2o         train L2O policy (JAX)"
	@echo "  train-l2o-bc      train L2O via behavior cloning (JAX)"
	@echo "  collect-sa-dataset collect SA dataset for BC"
	@echo "  bench-fastcollide benchmark fastcollide extension"
	@echo "  pre-commit        run pre-commit on all files"

install-dev:
	$(PY) -m pip install -e ".[dev]"

install-train:
	$(PY) -m pip install -e ".[train]"

test:
	$(PY) -m pytest -q

check: lint test

lint:
	$(PY) -m ruff check .

lint-fix:
	$(PY) -m ruff check . --fix

format:
	$(PY) -m ruff format . --exclude notebooks

format-check:
	$(PY) -m ruff format . --check --exclude notebooks

build-fastcollide:
	$(PY) scripts/build/build_fastcollide.py

generate-submission:
	$(PY) -m santa_packing.cli.generate_submission --config $(CONFIG) --nmax $(NMAX) --seed $(SEED) --overlap-mode $(OVERLAP_MODE) --out $(OUT) $(GEN_ARGS)

submit:
	$(PY) -m santa_packing.cli.make_submit --config $(CONFIG) --nmax $(NMAX) --name $(NAME) --submissions-dir $(SUBMISSIONS_DIR) -- $(GEN_ARGS)

score:
	$(PY) -m santa_packing.cli.score_submission $(SUB) --pretty

sweep-ensemble:
	$(PY) -m santa_packing.cli.sweep_ensemble --config $(ENSEMBLE_CONFIG) --nmax $(NMAX) --seeds $(SEEDS) --jobs $(JOBS) --out $(OUT) $(SWEEP_ARGS)

detouch-submission:
	$(PY) -m santa_packing.cli.detouch_submission $(DET_IN) --out $(DET_OUT) --scale $(DET_SCALE) --nmax $(NMAX)

train-meta:
	$(PY) -m santa_packing.cli.train_meta_init $(TRAIN_META_ARGS)

train-heatmap:
	$(PY) -m santa_packing.cli.train_heatmap_meta $(TRAIN_HEATMAP_ARGS)

train-l2o:
	$(PY) -m santa_packing.cli.train_l2o $(TRAIN_L2O_ARGS)

train-l2o-bc:
	$(PY) -m santa_packing.cli.train_l2o_bc $(TRAIN_L2O_BC_ARGS)

collect-sa-dataset:
	$(PY) -m santa_packing.cli.collect_sa_dataset $(SA_DATASET_ARGS)

bench-fastcollide:
	$(PY) -m santa_packing.cli.bench_fastcollide

pre-commit:
	pre-commit run -a
