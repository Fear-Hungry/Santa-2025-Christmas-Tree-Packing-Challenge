[![CI](https://github.com/Fear-Hungry/Santa-2025-Christmas-Tree-Packing-Challenge/actions/workflows/ci.yml/badge.svg)](https://github.com/Fear-Hungry/Santa-2025-Christmas-Tree-Packing-Challenge/actions/workflows/ci.yml)

## MODO ESTRATEGIA DE COMPETICAO — Santa 2025 (Christmas Tree Packing)

### [Resumo da competicao]

* **Tipo de problema:** otimizacao geometrica 2D (packing). Voce nao “treina modelo”; voce **constroi uma solucao geometrica** (posicoes + rotacoes) que minimize a funcao de score. ([AI Competition Hub][1])
* **Instancias:** voce precisa dar uma configuracao para **n = 1 ate 200 arvores**. ([AI Competition Hub][1])
* **O que vai no submission:** para cada `id` (ex.: `002_1`) voce entrega **posicao `(x,y)` e rotacao `deg`**. E tem uma pegadinha: os valores precisam ser **string com prefixo `s`** (ex.: `s0.0`). ([AI Competition Hub][1])
* **Restricoes importantes:**

  * **Nao pode haver sobreposicao** (overlap) — se houver, sua submissao pode dar erro. ([AI Competition Hub][1])
  * `-100 <= x,y <= 100`. ([AI Competition Hub][1])
* **Metrica (minimizacao):**
  [
  \text{score} = \sum_{n=1}^{200}\frac{s_n^2}{n}
  ]
  onde `s_n` e o lado do **quadrado** que “encaixota” a configuracao daquele `n`. ([AI Competition Hub][1])
  Em notebooks publicos, isso costuma ser implementado via **bounding box axis-aligned** (pega `min/max` de todos os vertices, e `s_n = max(width, height)`). ([Kaggle][2])

---

## Status atual do projeto (laboratorio)

* **Importante:** este repo e um **baseline/laboratorio**, nao uma “solucao final”. Ele ja tem blocos de estrategia (lattice, SA, vizinhanca, LNS/GA/hill-climb, ensembling), mas o ganho real vem de **tunar parametros + multi-start + mother-prefix + selecao por n**.
* **Score local:** rode `python -m santa_packing.cli.score_submission submission.csv --pretty` (use `--no-overlap` so para estimativa rapida; para submeter, valide com overlap).
* **Pipeline atual:** geracao via `santa_packing/cli/generate_submission.py` e registro de experimentos em `runs/`.
  * **Multi-start/ensemble:** `santa_packing/cli/sweep_ensemble.py` (selecao por instancia/`n`).

## O que “da score alto” aqui (na pratica)

Voce sobe score com 3 coisas (ordem de importancia):

1. **Geometria correta + checagem de colisao robusta**
   Se sua checagem falha (falso negativo), voce perde tempo e toma erro no submission. Se for conservadora demais, voce reduz densidade e piora score.

2. **Boa configuracao base (packing “estrutural”)**
   Tipicamente: algum **tiling/lattice** (triangular/hex) + padrao de rotacoes (ex.: alternar rotacoes por linha/coluna) e depois ajuste fino.

3. **Metaheuristica para refino** (quase sempre necessario para “score alto”)
   O espaco de busca e continuo e grande (≈ `3n` variaveis por instancia: `x,y,deg`). Abordagens comuns: **Simulated Annealing (SA)**, hill-climb, mutacoes estilo GA, etc.

---

## Checklist inicial (pra voce nao travar)

1. **Reproduzir a metrica localmente** (igual a do Kaggle) e ter um `score(submission)` rapido. Use a propria descricao/formula do score para validar. ([AI Competition Hub][1])
2. **Implementar a forma do “tree” como poligono** e transformacao rigida (rotacao + translacao). Em notebooks publicos, aparece como poligono com **NV=15 vertices** (bom sinal: da pra otimizar rapido). ([Kaggle][4])
3. **Implementar colisao eficiente**:

   * filtro grosso: **circulo envolvente** (raio = max distancia do vertice ao centro)
   * filtro fino: **intersecao de poligonos** (SAT se for convexo; se for concavo, triangule ou decomponha).
4. **Fazer um baseline deterministico** que sempre produz solucao valida (mesmo que ruim). So depois colocar SA/GA.

---

## Roadmap de otimizacao (alto ROI)

### 1) Baseline que “nao erra”

* Coloque centros em **grade** ou **lattice triangular** (densidade melhor que grade).
* Use uma rotacao fixa (ex.: `deg=0`) inicialmente.
* Gere solucao para todos `n` pegando os **n primeiros pontos** de uma lista ordenada por “proximidade do centro” (isso ajuda os `n` pequenos a terem caixa menor).
  Esse padrao aparece em abordagens publicas (“ordenar pontos por distancia ao centro para compactar”). ([Kaggle][5])

### 2) Insight que geralmente vale muito: “solucao-mae + recorte”

Em vez de resolver 200 problemas independentes, faca:

* Otimize bem um packing para **N=200**.
* Defina uma **ordem de inclusao** das arvores (camadas do centro para fora).
* Para cada `n`, use o prefixo das `n` primeiras arvores.

Isso costuma dar um salto grande porque voce gasta compute pesado em **uma** configuracao e “reaproveita” para todas.

### 3) SA (Simulated Annealing) / Hill-climb em cima da base

Movimentos tipicos:

* escolher arvore `i`
* propor `dx, dy, dθ` pequenos
* aceitar se:

  * nao colide
  * melhora `s_n` (ou melhora um custo suavizado)
  * ou aceita por temperatura (SA)

**Neste repo, o SA ja suporta "vizinhanca" (movimentos estruturados):**
* **swap**: permuta duas arvores (util quando `objective=prefix`/mother-prefix).
* **teleport**: move uma arvore da borda para um "pocket" perto de uma ancora no interior.
* **compact/push**: passos tipo greedy que empurram arvores em direcao ao centro (reduz o bbox mais rapido).
* Atalho: no gerador, use `--sa-proposal mixed --sa-neighborhood` (e, para mother-prefix, combine com `--sa-objective prefix` + `--sa-swap-prob ...`).

**Custo recomendado para SA:**
Se voce otimizar diretamente `s_n` (max de extents) pode ser “nao suave”. Uma alternativa e usar um custo com penalidade:

* `cost = max_x - min_x` e `max_y - min_y` e penalizar o maior; ou
* `cost = max(width, height) + λ * overlap_penalty` (overlap_penalty grande).

### 4) “Ensembling” aqui e selecao por instancia (nao media)

Como o score e soma por `n`, da pra fazer um ensemble extremamente forte:

* Rode 5–20 variacoes do solver (seeds, parametros, padroes de lattice, etc.)
* Para cada `n`, **pegue a configuracao com menor `s_n`** dentre as candidatas
* Monte um submission final misturando “o melhor de cada”

Isso aparece explicitamente como estrategia em notebooks publicos (ex.: “Ensembling_Santa_2025”). ([Kaggle][6])

### 5) Micro-otimizacoes que decidem leaderboard

* **Spatial hashing / grid de vizinhanca**: checar colisao so com arvores proximas (corta O(n^2)).
* **Numba/C++**: collision check e o gargalo; NV pequeno (ex.: 15) favorece otimizacao agressiva. ([Kaggle][4])
* **Precisao numerica / margens**: voce quer encostar sem “quase-colidir” (evitar erro na avaliacao). A exigencia do prefixo `s` e o foco em nao-overlap deixam claro que precisao importa. ([AI Competition Hub][1])

---

## Snippet minimo (formato de submission)

Formato exigido (com `s` prefixado) ([AI Competition Hub][1]):

```python
import pandas as pd

def fmt(x: float) -> str:
    # Ajuste casas conforme sua necessidade (mais casas = mais controle)
    return f"s{x:.9f}"

rows = []
for n in range(1, 201):
    for i in range(n):
        x, y, deg = 0.0, 0.0, 0.0  # <- preencher com sua solucao
        rows.append({
            "id": f"{n:03d}_{i}",
            "x": fmt(x),
            "y": fmt(y),
            "deg": fmt(deg),
        })

sub = pd.DataFrame(rows)
sub.to_csv("submission.csv", index=False)
```

---

## Se voce quer top 5%: o plano “curto” e eficaz

1. Pegar um baseline publico (tiling/greedy) e **rodar local score** (garantir que voce consegue submeter sem erro).
2. Rodar **SA com vizinhanca** + multi-start so para `N=200` (use `--sa-neighborhood` / `--refine-neighborhood`).
3. Definir uma boa **ordem de inclusao** (centro → borda) e gerar todos `n` por recorte.
4. Rodar varias seeds e fazer **selecao por `n`** (ensemble por instancia). ([Kaggle][6])

---

## Para eu te levar do seu baseline para “score alto”

Cole aqui:

* seu score atual (LB publico e/ou local),
* qual baseline esta usando (grid/hex/greedy/SA/GA),
* tempo de execucao que voce esta aceitando (ex.: “quero algo que rode em 10 min” vs “posso rodar horas offline”),
* se voce ja tem checagem de colisao (SAT/triangulacao/shapely/numba).

---

## Implementacao base neste repositorio (Python/JAX)

Este repo usa **Python (NumPy)** e tem um solver acelerado em **JAX** (opcional), substituindo a base C++ anterior.

### Setup

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

# Alternativas:
# - Minimo (so NumPy; roda lattice/pos-opt, mas sem SA/L2O):
#   python -m pip install -U -e .
# - Com JAX (sem notebooks):
#   python -m pip install -U -e ".[train]"

# Opcional (acelera `polygons_intersect` no score local):
python scripts/build/build_fastcollide.py
```

Se voce tiver GPU/CUDA, instale o pacote JAX adequado ao seu ambiente.

### Notebooks (VS Code/Jupyter)

Depois do setup, selecione o interpretador/kernel da `.venv` (assim o `ipykernel` instalado via `.[notebooks]` sera usado).

Notebook recomendado (1 clique: gerar + score + log):
* `notebooks/01_generate_and_score.ipynb`

### Estrutura de codigo

* `santa_packing/`: pacote principal (geometria, colisao, SA/L2O, scorer, etc.).
  * `santa_packing/main.py`: runner do SA batch (requer JAX; gera `best_packing.png`; exemplo em `assets/best_packing.png`).
  * `santa_packing/cli/`: CLIs oficiais (`generate_submission`, `score_submission`, `make_submit`, `sweep_ensemble`, treino, etc.).
* `scripts/build/`: build/compilacao (ex.: `scripts/build/build_fastcollide.py`).
* `scripts/submission/` e `scripts/evaluation/`: shims/wrappers (delegam para `python -m santa_packing.cli...`).
* `scripts/training/`: treino (L2O, meta-init, heatmap).
* `scripts/data/`: geracao de datasets (behavior cloning).
* `scripts/bench/`: benchmarks.
* `assets/`: recursos estaticos (ex.: `assets/best_packing.png`).

### Comandos rapidos

Instalar em modo desenvolvimento (opcional, mas recomendado para imports limpos):

```bash
python -m pip install -e ".[dev]"
```

Atalhos (Makefile):

```bash
make test
make submit NAME=baseline
```

Rodar SA isolado (uma instancia):

```bash
python3 -m santa_packing.main --n_trees 25 --batch_size 128 --n_steps 1000 --rot_prob 0.3 --proposal mixed --neighborhood --objective packing
```

Notas (qualidade sem custo grande):
* O SA usa por padrao: **push-to-center** (move deterministico leve), **adaptacao de sigma** (alvo de aceitacao) e **reheating** (se estagnar). Veja flags em `santa_packing/main.py` (ex.: `--no-adapt-sigma`, `--push_prob`, `--reheat_patience`).
* Para “SA com vizinhanca” (swap/teleport/compact), use `--neighborhood` no runner e `--sa-neighborhood`/`--refine-neighborhood` no gerador de submission.

Gerar submission (hibrido SA + lattice):

```bash
python -m santa_packing.cli.generate_submission --out submission.csv --nmax 200 --sa-nmax 30 --sa-steps 400 --sa-batch 64 --sa-proposal mixed --sa-neighborhood --sa-objective packing
```

Gerar + validar strict + arquivar (make submit; recomendado):

```bash
python -m santa_packing.cli.make_submit --config configs/submit.json --name baseline
# -> submissions/<timestamp>_<sha>.../{submission.csv,score.json,meta.json,*.log}
```

Config central (JSON) para evitar flags longas:

```bash
python -m santa_packing.cli.generate_submission --config configs/submit.json --out submission.csv --nmax 200
```

Sweep + ensemble por instancia (multi-start; escolhe o melhor `s_n` por `n` entre varias tentativas):

```bash
python -m santa_packing.cli.sweep_ensemble --nmax 200 --seeds 1,2,3 \\
  --recipe hex:"--lattice-pattern hex --lattice-rotations 0,15,30" \\
  --recipe square:"--lattice-pattern square --lattice-rotations 0,15,30" \\
  --out submission_ensemble.csv
```

Ensemble “inteligente” (mistura o melhor por `n` entre varios candidatos: lattice/SA/GA/hill-climb, etc.):

```bash
python -m santa_packing.cli.sweep_ensemble --nmax 200 --seeds 1..3 --jobs 3 \\
  --recipes-json scripts/submission/portfolios/mixed.json \\
  --out submission_ensemble.csv
```

Notas (lattice):
* `--lattice-rotate-mode` suporta `constant`, `row`, `checker`, `ring`. Em modo `constant`, `--lattice-rotations` tenta varias rotacoes; nos outros modos, vira uma sequencia repetida.
* Para um ajuste rapido apos o lattice (sem SA), use `--lattice-post-nmax 200 --lattice-post-steps 30` (hill-climb curto focado na borda do bbox).

Gerar submission usando L2O para n pequeno (exige policy treinada):

```bash
python -m santa_packing.cli.train_l2o --n 10 --train-steps 200 --out runs/l2o_policy.npz
python -m santa_packing.cli.generate_submission --out submission.csv --nmax 200 --l2o-model runs/l2o_policy.npz --l2o-nmax 10
```

Treinar L2O com GNN (kNN simples):

```bash
python -m santa_packing.cli.train_l2o --n 10 --train-steps 200 --policy gnn --knn-k 4 --out runs/l2o_gnn_policy.npz
python -m santa_packing.cli.generate_submission --out submission.csv --nmax 200 --l2o-model runs/l2o_gnn_policy.npz --l2o-nmax 10
```

Para alinhar o reward/objetivo ao score oficial (prefixo), use:
* L2O: `--preset submission` (ou `--reward prefix`).
* SA: `--sa-objective prefix` (no `--mother-prefix`, isso vira o default automaticamente; `--refine-objective/--block-objective` seguem o mesmo default).
Opcionalmente, experimente `--feature-mode rich`, `--gnn-attention`, `--gnn-steps` maior e `--overlap-lambda` pequeno (penalidade suave por overlap via circulos).

Treinamento com dataset multi-N (ex.: N=25,50,100) e diferentes inicializacoes:

```bash
python -m santa_packing.cli.train_l2o --n-list 25,50,100 --train-steps 200 \\
  --init mix --dataset-size 128 --dataset-out runs/l2o_dataset.npz \\
  --policy gnn --knn-k 4 --out runs/l2o_gnn_policy.npz
```

Treinamento supervisado (imitacao de SA / behavior cloning):

1) Colete um dataset de SA (guarda estados + deslocamentos aceitos):

```bash
python -m santa_packing.cli.collect_sa_dataset --n-list 25,50,100 --runs-per-n 5 --steps 400 \\
  --init mix --best-only --out runs/sa_bc_dataset.npz
```

2) Treine o modelo para imitar os deslocamentos aceitos:

```bash
python -m santa_packing.cli.train_l2o_bc --dataset runs/sa_bc_dataset.npz --policy gnn --knn-k 4 --train-steps 500 \\
  --out runs/l2o_bc_policy.npz
```

Treinamento baseado em mapas de calor (inspiracao MoCo, meta-otimizador + ES):

```bash
python -m santa_packing.cli.train_heatmap_meta --n-list 25,50,100 --train-steps 50 --es-pop 6 \\
  --heatmap-steps 200 --policy gnn --knn-k 4 --out runs/heatmap_meta.npz
```

Meta-inicializacao (gera um bom start para o SA; requer JAX):

```bash
python -m santa_packing.cli.train_meta_init --n-list 25,50,100 --train-steps 50 --es-pop 6 \\
  --sa-steps 100 --sa-batch 16 --objective packing --out runs/meta_init.npz

python -m santa_packing.cli.generate_submission --out submission.csv --nmax 200 \\
  --sa-nmax 50 --sa-steps 400 --meta-init-model runs/meta_init.npz
```

Heatmap meta (prioriza quais arvores mover; nao requer JAX):

```bash
python -m santa_packing.cli.train_heatmap_meta --n-list 10,20,30 --train-steps 50 --es-pop 6 \\
  --heatmap-steps 200 --out runs/heatmap_meta.npz

python -m santa_packing.cli.generate_submission --out submission.csv --nmax 200 \\
  --heatmap-model runs/heatmap_meta.npz --heatmap-nmax 20 --heatmap-steps 400
```

Uso no gerador (heatmap para n pequeno):

```bash
python -m santa_packing.cli.generate_submission --out submission.csv --nmax 200 \\
  --heatmap-model runs/heatmap_meta.npz --heatmap-nmax 10 --heatmap-steps 200
```

Scorar um submission:

```bash
python -m santa_packing.cli.score_submission submission.csv --pretty
```

Para acelerar (sem checar overlap):

```bash
python -m santa_packing.cli.score_submission submission.csv --no-overlap
```

Obs: `--no-overlap` e so para estimativa rapida; o gerador (`python -m santa_packing.cli.generate_submission`) roda validacao strict automaticamente ao final.

Observacao: a forma da arvore esta em `santa_packing/tree_data.py` (poligono oficial de 15 vertices). Se a geometria mudar no Kaggle, ajuste ali.

Testes rapidos:

```bash
python -m unittest discover -s tests
```

--- 

## L2O: modelo que substitui ou ajusta o SA (2.1)

**Ideia:** substituir o movimento aleatorio do SA por um modelo neural `f_theta` que, dado o estado atual, sugere deslocamentos `(dx, dy, dtheta)` e/ou a escolha da arvore a mover.

**Estado (entrada):** tensor `P in R^{N x 3}` com `(x, y, theta)` de cada arvore + features globais opcionais (bbox, area ocupada, densidade).

**Acoes (saida):**
* **Contínuas:** a rede prevê parametros de uma Gaussiana e amostramos `dx, dy, dtheta`.
* **Discretas:** a rede produz logits para escolher a arvore e o tipo de movimento.

**Arquiteturas candidatas:**
* **GNN/Graph Attention:** nos = arvores, arestas = proximas (kNN ou raio), embeddings capturam interacoes/colisoes e produzem deslocamentos por arvore.
* **MLP simples:** entrada concatenada das poses + features globais; saida gera deslocamento para 1 arvore (ou para todas).

**Objetivo (recompensa/perda):** minimizar `packing_score` e penalizar colisoes (ex.: `reward = -packing_score - lambda * overlap`).

Este repo ja possui um esqueleto L2O MLP em `santa_packing/l2o.py` (REINFORCE). Para trocar por GNN/Transformer, a ideia e manter a mesma API: `policy_apply(params, poses) -> (logits, mean)`.

---

## Ensembling / sweep / repair (Python)

A versao anterior em C++ tinha pipelines de **sweep**, **ensemble por instancia** e **repair**. Nesta versao Python, isso foi portado como scripts simples e reprodutiveis:

* **Sweep + ensemble por instancia:** `python -m santa_packing.cli.sweep_ensemble` roda varios `python -m santa_packing.cli.generate_submission` (seeds/receitas) e monta um submission final escolhendo o menor `s_n` por `n`.
* **Repair / refinamento local:** `python -m santa_packing.cli.generate_submission` ja inclui hill-climb e GA com reparo simples de overlaps para `n` pequenos (`--hc-*`, `--ga-*`).

---

## Meta-aprendizagem (inicializacao antes do SA)

Para melhorar a generalizacao a diferentes `N`, um regime de meta-aprendizagem treina uma rede para gerar **uma boa inicializacao**, e o SA faz o refinamento por um numero fixo de passos.

Pipeline sugerido:

```bash
python -m santa_packing.cli.train_meta_init --n-list 25,50,100 --train-steps 50 --sa-steps 100 --es-pop 6 \\
  --delta-xy 0.2 --delta-theta 10.0 --out runs/meta_init.npz
```

Uso na geracao de submissions (aplica meta-init antes do SA):

```bash
python -m santa_packing.cli.generate_submission --out submission.csv --nmax 200 \\
  --sa-nmax 30 --sa-steps 400 --sa-batch 64 --meta-init-model runs/meta_init.npz
```

[1]: https://www.competehub.dev/en/competitions/kagglesanta-2025 "Santa 2025 - Christmas Tree Packing Challenge - CompeteHub"
[2]: https://www.kaggle.com/code/jekiwantaufik/santa-2025-christmas-tree-packing-challenge?utm_source=chatgpt.com "Santa 2025 - Christmas Tree Packing Challenge"
[4]: https://www.kaggle.com/code/jekiwantaufik/santa-2025-christmas-tree-packing-challenge-2?utm_source=chatgpt.com "Santa 2025 - Christmas Tree Packing Challenge 2"
[5]: https://www.kaggle.com/code/koushikkumardinda/santa-2025-christmas-tree-packing-challenge?utm_source=chatgpt.com "Santa 2025 - Christmas Tree Packing Challenge"
[6]: https://www.kaggle.com/code/muhammadibrahim3093/ensembling-santa-2025?utm_source=chatgpt.com "Ensembling_Santa_2025"
