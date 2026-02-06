[![CI](https://github.com/Fear-Hungry/Santa-2025-Christmas-Tree-Packing-Challenge/actions/workflows/ci.yml/badge.svg)](https://github.com/Fear-Hungry/Santa-2025-Christmas-Tree-Packing-Challenge/actions/workflows/ci.yml)

# Santa 2025 — Christmas Tree Packing Challenge (baseline)

Este repositório é um **baseline/laboratório** para o desafio de *packing* 2D do Santa 2025: gerar um `submission.csv` com `(x, y, deg)` para `n=1..200` sem overlap e com bom score.

## Comece aqui

```bash
bash scripts/setup_venv.sh
python -m santa_packing
# -> escreve ./submission.csv e arquiva o run em ./submissions/<timestamp>...
```

Via Python (script/notebook): `from santa_packing.workflow import solve`

## Melhorar o score (pipeline atual)

O pipeline padrão (`python -m santa_packing`) já roda pós-processamento (subset-smoothing + polish do `n=200`)
e valida em `--overlap-mode kaggle` (mesma semântica do Kaggle: “touching” é permitido).

Obs: a métrica oficial é de **minimização** (menor é melhor).

Melhor score local atual (n=200, validação *strict*): `70.734908808395` (arquivo `submission.csv`). Para conferir:

```bash
python -m santa_packing.cli.score_submission submission.csv --nmax 200 --overlap-mode strict --pretty
```

Para buscar melhorias (multi-seed + ensemble por `n` + smoothing + post-opt), um comando típico é:

```bash
# Requer os binários C++ em `bin/` (compact_contact + post_opt).
# Roda vários seeds, faz ensemble por n, aplica smoothing e post-opt.
best_csv=$(python -m santa_packing._tools.hunt_compact_contact \
  --base submission.csv \
  --out-dir /tmp/hunt_cc \
  --seeds 4000..4127 --jobs 16 \
  --smooth-window 199 --post-opt)

cp "$best_csv" submission.csv
python -m santa_packing.cli.score_submission submission.csv --nmax 200 --overlap-mode strict --pretty
```

Para uma busca mais “iterativa” (muitas seeds rápidas), usamos uma configuração **low-pass** (mais barata por seed) e rodamos em batches até acumular ~1000 seeds:

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

Obs: o `submission.csv` atual foi obtido com *blend seguro por puzzle* (aceita apenas instâncias que melhoram e continuam sem overlap em `strict`),
incluindo rodadas de `hunt_compact_contact` + `post_opt` (C++) que trouxeram pequenas melhorias em vários puzzles.

Se quiser ser mais agressivo com `post_opt` (que pode gerar overlaps), use o repair por puzzle no `hunt_compact_contact`:

```bash
best_csv=$(python -m santa_packing._tools.hunt_compact_contact \
  --base submission.csv \
  --out-dir runs/hunt_cc_postopt \
  --seeds 1000..1031 --jobs 16 \
  --post-opt --post-iters 4000 --post-restarts 4 \
  --post-overlap-mode strict \
  --post-repair-mode finalize --post-repair-max-puzzles 25)
```

## Subset-smoothing “exato” (MILP)

Para tentar um subset-smoothing *exato* (escolher o melhor subconjunto de `n` poses vindo de algum `m>n` sem alterar as poses), use o MILP com OR-Tools:

```bash
python -m pip install -U -e ".[milp]"

.venv/bin/python -m santa_packing._tools.milp_subset_smooth \
  submission.csv --out runs/milp/milp.csv \
  --window 199 --topk-m 8 --time-limit-s 0.2 --ns 1..200 --verbose

python -m santa_packing.cli.score_submission runs/milp/milp.csv --nmax 200 --overlap-mode strict --pretty
```

## Submissão no Kaggle (CLI)

```bash
.venv/bin/kaggle competitions submit -c santa-2025 -f submission.csv -m "kaggle-valid (improve_submission)"
```

## Kaggle: “ERROR” (overlap/touch)

O Kaggle permite encostar (touching), então `--overlap-mode kaggle` é equivalente a `strict` (touching permitido).  
Se você quiser uma validação mais conservadora (às custas de densidade/score), use `--overlap-mode conservative`.

Na prática, o avaliador oficial usa **Shapely** com `scale_factor=1e18`, então soluções “no limite” podem passar no checker local e ainda assim falhar no Kaggle por **micro-overlaps**. Se isso acontecer, valide/ajuste usando a mesma regra do Kaggle (Shapely):

```bash
.venv/bin/python -m santa_packing._tools.kaggle_autofix_submission submission.csv \
  --out submission_kaggle_metric.csv
```

Para “auto-fixar” um CSV existente, use:

```bash
python -m santa_packing.cli.autofix_submission submission.csv \
  --out submission_kaggle.csv --overlap-mode kaggle
python -m santa_packing.cli.score_submission submission_kaggle.csv --nmax 200 --overlap-mode kaggle --pretty
```

## Configuração (reprodutibilidade)

O workflow (`python -m santa_packing`) carrega config por padrão (quando existir):

* `configs/submit_strong.json` (default do workflow; preset “pesado”)
* `configs/submit.json` (preset mais leve; use via `--config`)
* `configs/ensemble.json` (default para `sweep_ensemble`)

Para sobrescrever, passe flags normalmente; para trocar/ignorar config: use `--config ...` ou `--no-config`.

## Documentação (por tema)

Detalhes extensos (roadmap, guia de L2O, instruções de ambiente) foram movidos para `docs/`:

* `docs/overview.md` — panorama da competição + estratégia
* `docs/roadmap.md` — roadmap de otimização (alto ROI)
* `docs/setup.md` — setup, estrutura do código e comandos rápidos
* `docs/sa.md` — SA (proposals, vizinhança, objetivos) e exemplos
* `docs/l2o.md` — L2O, meta-init e heatmap meta
* `docs/ensemble.md` — sweep/ensemble/repair

## Desenvolvimento

```bash
make test
make lint
make format
```
