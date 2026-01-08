[![CI](https://github.com/Fear-Hungry/Santa-2025-Christmas-Tree-Packing-Challenge/actions/workflows/ci.yml/badge.svg)](https://github.com/Fear-Hungry/Santa-2025-Christmas-Tree-Packing-Challenge/actions/workflows/ci.yml)

# Santa 2025 — Christmas Tree Packing Challenge (baseline)

Este repositório é um **baseline/laboratório** para o desafio de *packing* 2D do Santa 2025: gerar um `submission.csv` com `(x, y, deg)` para `n=1..200` sem overlap e com bom score.

## Comece aqui

```bash
bash scripts/setup_venv.sh
python -m santa_packing.cli.generate_submission --out submission.csv --nmax 200
python -m santa_packing.cli.score_submission submission.csv --pretty
```

## Melhorar o score (pipeline atual)

O gerador acima é **baseline**. Para resultados bem melhores, use o pós-processamento:

```bash
python -m santa_packing.cli.improve_submission submission.csv --out submission.csv \
  --smooth-window 60 --improve-n200 --overlap-mode strict
python -m santa_packing.cli.score_submission submission.csv --nmax 200 --overlap-mode strict --pretty
```

Reprodução do melhor run local atual (atingiu `~72.816` com validação *strict*):

```bash
# Requer os binários C++ em `bin/` (compact_contact + post_opt).
# Roda vários seeds, faz ensemble por n, aplica smoothing e post-opt.
best_csv=$(python -m santa_packing.cli.hunt_compact_contact \
  --base submission.csv \
  --out-dir /tmp/hunt_cc \
  --seeds 4000..4127 --jobs 16 \
  --smooth-window 199 --post-opt)

cp "$best_csv" submission.csv
python -m santa_packing.cli.score_submission submission.csv --nmax 200 --overlap-mode strict --pretty
```

## Submissão no Kaggle (CLI)

```bash
.venv/bin/kaggle competitions submit -c santa-2025 -f submission.csv -m "strict ~73 (improve_submission)"
```

## Kaggle: “ERROR” (overlap/touch)

Algumas soluções passam no `--overlap-mode strict` local mas dão **ERROR** no Kaggle (tipicamente por encostar/overlap).

Para “auto-fixar” um CSV existente, use:

```bash
python -m santa_packing.cli.autofix_submission submission.csv \
  --out submission_kaggle.csv --overlap-mode conservative
python -m santa_packing.cli.score_submission submission_kaggle.csv --nmax 200 --overlap-mode conservative --pretty
```

## Configuração (reprodutibilidade)

Os defaults das CLIs ficam centralizados em `configs/`:

* `configs/submit.json` (default para `generate_submission` e `make_submit`)
* `configs/submit_strong.json` (preset “pesado”: `--mother-prefix` + SA com vizinhança; use via `--config`)
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
