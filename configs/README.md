# Configs

Arquivos em `configs/` existem para reduzir a quantidade de flags nas CLIs e facilitar reprodutibilidade.

## Convenções

* Use **JSON** (suportado por padrão). YAML é opcional (requer `pyyaml` instalado).
* Os comandos carregam configs por padrão (quando existir um default), e **flags na linha de comando sempre sobrescrevem** a config.
* Formatos suportados:
  * `{"args": ["--flag", "value", "--bool-flag"]}` (argv explícito; útil para flags repetidas)
  * `{"<section>": {"flag": 123, "nested": {"flag": 1}}}` (mapeamento; chaves aninhadas viram `--nested-flag`)

## Arquivos padrão

* `configs/submit.json` — defaults para `make_submit` e `generate_submission`
* `configs/ensemble.json` — defaults para `sweep_ensemble`

