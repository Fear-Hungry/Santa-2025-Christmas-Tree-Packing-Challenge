# scripts/submission

Arquivos relacionados a *submission* e portfólios de receitas.

## Recomendações

Para evitar “mágica” escondida, prefira invocar CLIs como módulos:

* `python -m santa_packing.cli.generate_submission`
* `python -m santa_packing.cli.make_submit`
* `python -m santa_packing.cli.sweep_ensemble`

Ou, após instalar, use os console scripts definidos em `pyproject.toml` (ex.: `santa-score-submission`).

## O que fica aqui

* `portfolios/*.json`: coleções de receitas para `sweep_ensemble`.
* `portfolio_merge.py`: workflow **legado** (baseado em binários C++ antigos). Para o pipeline atual, use `santa_packing/cli/sweep_ensemble.py`.

