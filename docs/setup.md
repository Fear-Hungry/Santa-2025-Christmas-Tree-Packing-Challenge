# Setup e execução

Este repo usa **Python (NumPy)** e tem partes aceleradas em **JAX** (opcional, mas recomendado para SA/L2O). Há também uma extensão C++ opcional (`fastcollide`) para acelerar o scorer local.

## Setup (recomendado)

```bash
bash scripts/setup_venv.sh
```

Requer **Python 3.12+** (este repo fixa `3.12.3` em `.python-version`). Se preferir manual:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

# Recomendado (JAX habilita SA/L2O):
python -m pip install -U -e ".[train,notebooks]"

# Opcional (dev tooling):
python -m pip install -U -e ".[dev]"

# Opcional (acelera polygons_intersect no scorer local):
python scripts/build/build_fastcollide.py
```

Se você tiver GPU/CUDA, instale o pacote JAX adequado ao seu ambiente.

## Notebooks (VS Code/Jupyter)

Depois do setup, selecione o interpretador/kernel da `.venv` (assim o `ipykernel` instalado via `.[notebooks]` será usado).

Notebook recomendado (gerar + score + log):
* `notebooks/01_generate_and_score.ipynb`

## Estrutura do código

* `santa_packing/`: pacote principal (geometria, colisão, SA/L2O, scorer, etc.).
  * `santa_packing/main.py`: runner de SA batch (JAX; gera `best_packing.png`).
  * `santa_packing/cli/`: CLIs (`generate_submission`, `score_submission`, `make_submit`, `sweep_ensemble`, treino, etc.).
* `scripts/`: wrappers e rotinas auxiliares (build, training, submission, etc.).
* `configs/`: configs JSON para evitar flags longas.
* `runs/`: logs/artefatos de treino/experimentos.
* `tests/`: testes unitários.

## Configs (JSON/YAML)

As CLIs principais carregam configs por padrão (quando presentes):

* `generate_submission` / `make_submit`: `configs/submit.json`
* `sweep_ensemble`: `configs/ensemble.json`

Para sobrescrever, passe flags normalmente; para trocar/ignorar config: `--config ...` ou `--no-config`.
Mais detalhes em `configs/README.md`.

## Comandos rápidos

Atalhos (Makefile):

```bash
make test
make lint
make format
```

Gerar um submission (baseline):

```bash
python -m santa_packing.cli.generate_submission --out submission.csv --nmax 200
```

Scorar um submission:

```bash
python -m santa_packing.cli.score_submission submission.csv --pretty
```

Dica: para estimativa rápida use `--no-overlap`; para validação antes de submeter use `--overlap-mode kaggle` (mais conservador, evita ERROR por “touching”/tolerância).

```bash
python -m santa_packing.cli.score_submission submission.csv --overlap-mode kaggle
```

Melhorar um submission existente (subset-smoothing + melhora do `n=200` via insert+SA + reparo/validação):

```bash
python -m santa_packing.cli.improve_submission submission.csv --out submission.csv --smooth-window 60 --improve-n200
```

Gerar + validar + arquivar (recomendado para “submeter” com rastreabilidade):

```bash
python -m santa_packing.cli.make_submit --name baseline
# -> submissions/<timestamp>_<sha>.../{submission.csv,score.json,meta.json,*.log}
```

Para detalhes de SA, veja `docs/sa.md`. Para ensemble/sweep, veja `docs/ensemble.md`. Para treino (L2O/meta-init/heatmap), veja `docs/l2o.md`.
