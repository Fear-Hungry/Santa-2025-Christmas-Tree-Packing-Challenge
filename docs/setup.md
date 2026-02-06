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

# Opcional (solvers exatos via MILP / subset-smoothing):
python -m pip install -U -e ".[milp]"

# Opcional (dev tooling):
python -m pip install -U -e ".[dev]"

# Opcional (acelera polygons_intersect no scorer local):
python scripts/build/build_fastcollide.py
```

Se você tiver GPU/CUDA, instale o pacote JAX adequado ao seu ambiente.

## Notebooks (VS Code/Jupyter)

Depois do setup, selecione o interpretador/kernel da `.venv` (assim o `ipykernel` instalado via `.[notebooks]` será usado).

Notebooks recomendados:
* `notebooks/01_generate_and_score.ipynb` (gerar + score + log)
* `notebooks/02_optimization_hunt.ipynb` (hunt/otimização para melhores resultados)

### Otimizar a partir de um CSV existente (ex: score 70.74)

Exemplo: o `submission.csv` versionado na raiz está em `70.734908808395` com validação `--overlap-mode strict` (nmax=200).

No `notebooks/02_optimization_hunt.ipynb`:
* aponte `BASE_SUBMISSION` para o seu arquivo (`Path("/caminho/para/submission.csv")`)
* ajuste `TARGET_SCORE = 69.0` (ou `None` para não parar cedo)
* rode em `MODE="full"` e aumente seeds/iters conforme quiser

O notebook valida em `OVERLAP_MODE="kaggle"` e faz repair automático quando necessário, então o `submission.csv` final fica sem overlap.

Obs: o avaliador do Kaggle usa Shapely com `scale_factor=1e18` e pode acusar **micro-overlaps** em soluções “no limite”. Para checar/ajustar com a mesma regra do Kaggle, use:

```bash
.venv/bin/python -m santa_packing._tools.kaggle_autofix_submission submission.csv --out submission_kaggle_metric.csv
```

### Refinar por instância (por `n`)

Quando quiser otimizar *cada* puzzle separadamente (ex: `n=1..200`), um atalho é:

```bash
python -m santa_packing._tools.refine_submission \
  --base submission.csv --out runs/refine/refined.csv \
  --ns 1..200 --jobs 16 --repeats 2 --overlap-mode strict
```

O refinador mantém apenas melhorias estritas em `s_n` (após quantização) e garante que o CSV final esteja sem overlap no modo escolhido.

### Otimizar `n` médios (multi-seed, C++)

Para focar em uma faixa (ex: `n=50..150`) usando o `compact_contact` (C++) com multi-seed:

```bash
python -m santa_packing._tools.optimize_by_n_compact_contact \
  --base submission.csv --out runs/opt_by_n_cc/medium.csv \
  --ns 50..150 --seeds 40000..40031 --jobs 16 \
  --passes 300 --attempts-per-pass 150 --quantize-decimals 17 \
  --overlap-mode strict --report runs/opt_by_n_cc/medium_report.csv
```

O `optimize_by_n_compact_contact` também suporta **multi-restart** (perturbação do estado inicial + recompact) para diversificar buscas:

```bash
python -m santa_packing._tools.optimize_by_n_compact_contact \
  --base submission.csv --out runs/opt_by_n_cc/medium_restart.csv \
  --ns 50..150 --seeds 40000..40007 --restarts 4 --jobs 16 \
  --restart-scale-min 1.02 --restart-scale-max 1.08 \
  --restart-jitter-xy 0.05 --restart-jitter-deg 10 \
  --cc-timeout-s 10 \
  --overlap-mode strict --report runs/opt_by_n_cc/medium_restart_report.csv
```

Obs: `--restart-allow-overlap` aumenta diversidade, mas pode deixar alguns candidatos com overlaps severos e tornar o repair/finalize bem mais lento. Em geral, comece sem essa flag.

### Busca “iterativa” (>=1000 seeds) com `hunt_compact_contact` (C++)

Quando quiser fazer uma busca rápida de muitas seeds (e ir acumulando melhorias), use uma configuração **low-pass** e rode em batches (ex.: 4×256 seeds = 1024):

```bash
best_csv=$(python -m santa_packing._tools.hunt_compact_contact \
  --base submission.csv \
  --out-dir runs/hunt_cc_lowpass \
  --seeds 63000..63255 --jobs 16 \
  --target-range 1,200 \
  --passes 50 --attempts-per-pass 30 --patience 30 \
  --shake-pos 0.06 --shake-rot-deg 18 --shake-prob 0.4 \
  --quantize-decimals 17)

cp "$best_csv" submission.csv
python -m santa_packing.cli.score_submission submission.csv --nmax 200 --overlap-mode strict --pretty
```

Se você ativar `--post-opt`, dá para tentar “salvar” melhorias que vierem com overlap usando repair por puzzle:

```bash
best_csv=$(python -m santa_packing._tools.hunt_compact_contact \
  --base submission.csv \
  --out-dir runs/hunt_cc_postopt \
  --seeds 1000..1031 --jobs 16 \
  --post-opt --post-iters 4000 --post-restarts 4 \
  --post-overlap-mode strict \
  --post-repair-mode finalize --post-repair-max-puzzles 25)
```

## Estrutura do código

* `santa_packing/`: pacote principal (geometria, colisão, SA/L2O, scorer, etc.).
  * `santa_packing/workflow.py`: workflow de alto nível (generate → improve → validar/scorar → archive) e CLI único (`python -m santa_packing`).
  * `santa_packing/cli/`: CLIs menores/internas (`generate_submission`, `improve_submission`, `autofix_submission`, `score_submission`).
  * `santa_packing/_tools/`: ferramentas pesadas de experimento (hunt, sweep/ensemble, treino, bench, etc.).
  * `santa_packing/main.py`: runner de SA batch (JAX; gera `best_packing.png`).
* `bin/`: binaries C++ (solvers e pós-otimização).
* `scripts/`: wrappers e rotinas auxiliares (build, training, submission, etc.).
* `configs/`: configs JSON para evitar flags longas.
* `runs/`: logs/artefatos de treino/experimentos.
* `tests/`: testes unitários.

## Configs (JSON/YAML)

Os comandos principais carregam configs por padrão (quando presentes):

* `python -m santa_packing`: `configs/submit_strong.json` (fallback `configs/submit.json`)
* `generate_submission`: `configs/submit.json`
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
python -m santa_packing
```

O comando acima já gera, melhora, valida/scora e exporta um `submission.csv` pronto para enviar ao Kaggle (além de arquivar artefatos em `submissions/`).

Opcional: re-scorar um CSV manualmente (útil para checar outros arquivos):

```bash
python -m santa_packing.cli.score_submission submission.csv --overlap-mode kaggle --pretty
```

Para detalhes de SA, veja `docs/sa.md`. Para ensemble/sweep, veja `docs/ensemble.md`. Para treino (L2O/meta-init/heatmap), veja `docs/l2o.md`.
