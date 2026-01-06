[![CI](https://github.com/Fear-Hungry/Santa-2025-Christmas-Tree-Packing-Challenge/actions/workflows/ci.yml/badge.svg)](https://github.com/Fear-Hungry/Santa-2025-Christmas-Tree-Packing-Challenge/actions/workflows/ci.yml)

# Santa 2025 — Christmas Tree Packing Challenge (baseline)

Este repositório é um **baseline/laboratório** para o desafio de *packing* 2D do Santa 2025: gerar um `submission.csv` com `(x, y, deg)` para `n=1..200` sem overlap e com bom score.

## Comece aqui

```bash
bash scripts/setup_venv.sh
python -m santa_packing.cli.generate_submission --out submission.csv --nmax 200
python -m santa_packing.cli.score_submission submission.csv --pretty
```

## Configuração (reprodutibilidade)

Os defaults das CLIs ficam centralizados em `configs/`:

* `configs/submit.json` (default para `generate_submission` e `make_submit`)
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
