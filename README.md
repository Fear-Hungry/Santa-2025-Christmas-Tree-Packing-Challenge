## MODO ESTRATÉGIA DE COMPETIÇÃO — Santa 2025 (Christmas Tree Packing)

### [Resumo da competição]

* **Tipo de problema:** otimização geométrica 2D (packing). Você não “treina modelo”; você **constrói uma solução geométrica** (posições + rotações) que minimize a função de score. ([AI Competition Hub][1])
* **Instâncias:** você precisa dar uma configuração para **n = 1 até 200 árvores**. ([AI Competition Hub][1])
* **O que vai no submission:** para cada `id` (ex.: `002_1`) você entrega **posição `(x,y)` e rotação `deg`**. E tem uma pegadinha: os valores precisam ser **string com prefixo `s`** (ex.: `s0.0`). ([AI Competition Hub][1])
* **Restrições importantes:**

  * **Não pode haver sobreposição** (overlap) — se houver, sua submissão pode dar erro. ([AI Competition Hub][1])
  * `-100 ≤ x,y ≤ 100`. ([AI Competition Hub][1])
* **Métrica (minimização):**
  [
  \text{score} = \sum_{n=1}^{200}\frac{s_n^2}{n}
  ]
  onde `s_n` é o lado do **quadrado** que “encaixota” a configuração daquele `n`. ([AI Competition Hub][1])
  Em notebooks públicos, isso costuma ser implementado via **bounding box axis-aligned** (pega `min/max` de todos os vértices, e `s_n = max(width, height)`). ([Kaggle][2])

---

## Status atual do projeto (laboratório)

* **Score local atual (fonte de verdade: `./bin/score_submission`):** `84.394057199`
  * **Arquivo:** `submissions/submission_best.csv` (ver `submissions/BEST.txt` e `submissions/scores.tsv`)
  * Observação: é **score local** (minimização); LB público pode variar.
* **Como chegamos nele:**
  * Pipeline **auto80** em `runs/auto80_20251219_233144` (script `auto80.sh` + logs).
  * Cada iteração roda um **sweep de 200 runs** combinando `solver_tile` + `solver_tessellation` com variação de parâmetros/offsets (via `santa_pipeline.py sweep` / `scripts/sweep_blend.py`).
  * Depois do sweep, faz **ensemble + repair** contra o best atual e **promove** se melhorar.
  * Melhor atual surgiu na `iter_0005` (seed0 `10801`), ver `runs/auto80_20251219_233144/progress.tsv` e `auto80.log`.

## O que “dá score alto” aqui (na prática)

Você sobe score com 3 coisas (ordem de importância):

1. **Geometria correta + checagem de colisão robusta**
   Se sua checagem falha (falso negativo), você perde tempo e toma erro no submission. Se for conservadora demais, você reduz densidade e piora score.

2. **Boa configuração base (packing “estrutural”)**
   Tipicamente: algum **tiling/lattice** (triangular/hex) + padrão de rotações (ex.: alternar rotações por linha/coluna) e depois ajuste fino.

3. **Metaheurística para refino** (quase sempre necessário para “score alto”)
   O espaço de busca é contínuo e grande (≈ `3n` variáveis por instância: `x,y,deg`). Abordagens comuns: **Simulated Annealing (SA)**, hill-climb, mutações estilo GA, etc.

---

## Checklist inicial (pra você não travar)

1. **Reproduzir a métrica localmente** (igual à do Kaggle) e ter um `score(submission)` rápido. Use a própria descrição/fórmula do score para validar. ([AI Competition Hub][1])
2. **Implementar a forma do “tree” como polígono** e transformação rígida (rotação + translação). Em notebooks públicos, aparece como polígono com **NV=15 vértices** (bom sinal: dá pra otimizar rápido). ([Kaggle][4])
3. **Implementar colisão eficiente**:

   * filtro grosso: **círculo envolvente** (raio = max distância do vértice ao centro)
   * filtro fino: **interseção de polígonos** (SAT se for convexo; se for côncavo, triangule ou decomponha).
4. **Fazer um baseline determinístico** que sempre produz solução válida (mesmo que ruim). Só depois colocar SA/GA.

---

## Roadmap de otimização (alto ROI)

### 1) Baseline que “não erra”

* Coloque centros em **grade** ou **lattice triangular** (densidade melhor que grade).
* Use uma rotação fixa (ex.: `deg=0`) inicialmente.
* Gere solução para todos `n` pegando os **n primeiros pontos** de uma lista ordenada por “proximidade do centro” (isso ajuda os `n` pequenos a terem caixa menor).
  Esse padrão aparece em abordagens públicas (“ordenar pontos por distância ao centro para compactar”). ([Kaggle][5])

### 2) Insight que geralmente vale muito: “solução-mãe + recorte”

Em vez de resolver 200 problemas independentes, faça:

* Otimize bem um packing para **N=200**.
* Defina uma **ordem de inclusão** das árvores (camadas do centro para fora).
* Para cada `n`, use o prefixo das `n` primeiras árvores.

Isso costuma dar um salto grande porque você gasta compute pesado em **uma** configuração e “reaproveita” para todas.

### 3) SA (Simulated Annealing) / Hill-climb em cima da base

Movimentos típicos:

* escolher árvore `i`
* propor `dx, dy, dθ` pequenos
* aceitar se:

  * não colide
  * melhora `s_n` (ou melhora um custo suavizado)
  * ou aceita por temperatura (SA)

**Custo recomendado para SA:**
Se você otimizar diretamente `s_n` (max de extents) pode ser “não suave”. Uma alternativa é usar um custo com penalidade:

* `cost = max_x - min_x` e `max_y - min_y` e penalizar o maior; ou
* `cost = max(width, height) + λ * overlap_penalty` (overlap_penalty grande).

### 4) “Ensembling” aqui é seleção por instância (não média)

Como o score é soma por `n`, dá pra fazer um ensemble extremamente forte:

* Rode 5–20 variações do solver (seeds, parâmetros, padrões de lattice, etc.)
* Para cada `n`, **pegue a configuração com menor `s_n`** dentre as candidatas
* Monte um submission final misturando “o melhor de cada”

Isso aparece explicitamente como estratégia em notebooks públicos (ex.: “Ensembling_Santa_2025”). ([Kaggle][6])

### 5) Micro-otimizações que decidem leaderboard

* **Spatial hashing / grid de vizinhança**: checar colisão só com árvores próximas (corta O(n²)).
* **Numba/C++**: collision check é o gargalo; NV pequeno (ex.: 15) favorece otimização agressiva. ([Kaggle][4])
* **Precisão numérica / margens**: você quer encostar sem “quase-colidir” (evitar erro na avaliação). A exigência do prefixo `s` e o foco em não-overlap deixam claro que precisão importa. ([AI Competition Hub][1])

---

## Snippet mínimo (formato de submission)

Formato exigido (com `s` prefixado) ([AI Competition Hub][1]):

```python
import pandas as pd

def fmt(x: float) -> str:
    # Ajuste casas conforme sua necessidade (mais casas = mais controle)
    return f"s{x:.9f}"

rows = []
for n in range(1, 201):
    for i in range(n):
        x, y, deg = 0.0, 0.0, 0.0  # <- preencher com sua solução
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

## Se você quer top 5%: o plano “curto” e eficaz

1. Pegar um baseline público (tiling/greedy) e **rodar local score** (garantir que você consegue submeter sem erro).
2. Implementar **SA com vizinhança** + multi-start só para `N=200`.
3. Definir uma boa **ordem de inclusão** (centro → borda) e gerar todos `n` por recorte.
4. Rodar várias seeds e fazer **seleção por `n`** (ensemble por instância). ([Kaggle][6])

---

## Para eu te levar do seu baseline para “score alto”

Cole aqui:

* seu score atual (LB público e/ou local),
* qual baseline está usando (grid/hex/greedy/SA/GA),
* tempo de execução que você está aceitando (ex.: “quero algo que rode em 10 min” vs “posso rodar horas offline”),
* se você já tem checagem de colisão (SAT/triangulação/shapely/numba).

---

## Implementação base neste repositório (C++)

Neste repo vamos usar **C++** como linguagem principal do solver:

Build (CMake):

```bash
cmake -S . -B build
cmake --build build -j
```

Depois de configurar, use `cmake --build build -j --target <alvo>` para compilar um binário específico.

Observação: o build “de verdade” é via CMake; o `Makefile` na raiz é apenas um wrapper de conveniência (ele inclui `convenience.mk`) para compatibilidade com scripts.

* Código em C++ dividido em módulos:
  * `include/geom.hpp` / `src/geometry/geom.cpp`: tipos (`Point`, `Polygon`, `TreePose`), transformação de polígonos, bounding box / bounding square (`s_n`), formatação `s...`.
  * `include/collision.hpp` / `src/geometry/collision.cpp` + `src/geometry/collision_polygons.cpp`: checagem de colisão (broad-phase por círculo envolvente + narrow-phase por interseção de segmentos).
  * `include/baseline.hpp` / `src/solvers/baseline.cpp`: baseline em grade que gera posições sem overlap.
  * `apps/solver_baseline.cpp`: `main` que usa os módulos acima e escreve um `submission_baseline_cpp.csv` válido (use `--output` para mudar o caminho).

Para compilar e gerar o submission baseline:

```bash
cmake --build build -j --target solver_baseline
./bin/solver_baseline --output runs/tmp/submission_baseline_cpp.csv
```

Saída padrão: `submission_baseline_cpp.csv` (+ score local no terminal). No exemplo acima: `runs/tmp/submission_baseline_cpp.csv`.

### Solver de tesselação (lattice hexagonal)

Este solver gera uma solução “estrutural” baseada em **tesselação hexagonal** (tiling) e escolhe, para cada `n`, o melhor ângulo dentre alguns candidatos para minimizar o quadrado axis-aligned. Ele também imprime o **score local** no terminal.

Compilar e rodar:

```bash
cmake --build build -j --target solver_tessellation
./bin/solver_tessellation --output runs/tmp/submission_tessellation_cpp.csv
```

Saída padrão: `submission_tessellation_cpp.csv`. No exemplo acima: `runs/tmp/submission_tessellation_cpp.csv`.

Opções úteis (SA é opcional; padrão = desligado):

* `--preset quick|balanced|quality` (atalhos de orçamento; aliases: `fast`, `medium`, `moderate`, `slow`)
* `--use-ga` (ativa GA como busca global; gera um candidato extra via GA e compara com a tesselação padrão)
* `--ga-pop 40` / `--ga-gens 60` / `--ga-elite 2` / `--ga-tournament 3`
* `--ga-spacing-min 1.000` / `--ga-spacing-max 1.010`
* `--ga-rots 0,180` (ângulos candidatos por árvore no GA)
* `--sa-restarts 3`
* `--sa-base-iters 500`
* `--sa-iters-per-n 20`
* `--sa-w-micro 1.0` / `--sa-w-swap-rot 0.25` / `--sa-w-relocate 0.15` (pesos do portfólio)
* `--sa-w-block-translate 0.05` / `--sa-w-block-rotate 0.02` / `--sa-w-lns 0.001` (vizinhanças grandes; `lns` é caro)
* `--sa-w-push-contact 0.2` (macro determinístico: empurra árvore da casca até contato no eixo dominante)
* `--sa-w-squeeze 0.05` / `--sa-squeeze-pushes 6` (repete `push-contact` algumas vezes; tende a igualar width≈height)
* `--sa-block-size 6` (tamanho do bloco nos macro-movimentos)
* `--sa-lns-remove 6` (quantas árvores remover no LNS; o remove mira a **casca**/extremos)
* `--sa-hh-segment 50` / `--sa-hh-reaction 0.20` (controlador adaptativo de vizinhanças)
  * Dica prática: pra reduzir `s_n`, normalmente vale **subir macro** (`block-translate`, `lns`) e **baixar micro**.
* `--no-final-rigid` (desliga o pós-processamento “final rigid” por `n`)
* `--seed 1` (reprodutibilidade)
* `--angles 0,15,30,45,60,75`
* `--spacing-safety 1.001`
* `--shift-a 0.0` / `--shift-b 0.0` / `--shift a,b` (offset da lattice; ótimo pra sweeps/ensembling)
* `--output submission_tessellation_cpp.csv`

Auto-seleção simples (escolhe o melhor preset pelo score local):

```bash
python3 scripts/auto_tessellation.py --output runs/tmp/submission_tessellation_auto.csv
```

### Solver de tile (motif) + translação (+ refino de fronteiras)

Este solver implementa a ideia de **“tile pequeno + translação”**:

* define um **motif** com `k` árvores dentro de uma célula fundamental (tile),
* acha o menor `spacing` seguro para o tile via checagem periódica,
* replica o tile em uma lattice (default: hex/triangular) e gera um **pool de candidatos** (`--pool-size`, default: 600),
* gera `n=1..200` via uma **ordem de prefixo** (`--prefix-order`):
  * `central`: ordem por centralidade (primeiro `max(|x|,|y|)`, depois distância),
  * `greedy`: ordem gulosa que adiciona a próxima árvore que minimiza o novo `s_n` (bounding square),
* se `prune` estiver ligado (padrão), também tenta um **pruning guloso da borda** e escolhe o melhor (prefixo vs. prune) por `n`,
* opcionalmente faz um **refino local de fronteira**: compaction “puxa a casca pra dentro” com backtracking + um SA leve.

Compilar e rodar:

```bash
cmake --build build -j --target solver_tile
./bin/solver_tile --output runs/tmp/submission_tile_cpp.csv
```

Opções úteis:

* `--k 4` (tamanho do tile / motif)
* `--pool-size 800` (tamanho do pool de candidatos para o prefixo/prune)
* `--prefix-order greedy` (prefixo “score-aware” por minimização incremental de `s_n`)
* `--tile-iters 5000` (busca aleatória simples para “apertar” o tile; offline)
* `--lattice-v-ratio 1.0` / `--lattice-theta 60` (reticulado geral: |v|/|u| e ângulo(u,v); default é hex)
* `--no-tile-opt-lattice` (otimiza só o motif, não mexe no reticulado durante `--tile-iters`)
* `--refine-iters 20000` (refino local nas fronteiras; offline)
* `--sa-restarts 3` / `--sa-base-iters 500` / `--sa-iters-per-n 20` (SA opcional)
* `--sa-w-micro 1.0` / `--sa-w-swap-rot 0.25` / `--sa-w-relocate 0.15` (pesos do portfólio)
* `--sa-w-block-translate 0.05` / `--sa-w-block-rotate 0.02` / `--sa-w-lns 0.001` (vizinhanças grandes; `lns` é caro)
* `--sa-block-size 6` (tamanho do bloco nos macro-movimentos)
* `--sa-lns-remove 6` (quantas árvores remover no LNS; o remove mira a **casca**/extremos)
* `--sa-hh-segment 50` / `--sa-hh-reaction 0.20` (controlador adaptativo de vizinhanças)
  * Dica prática: pra reduzir `s_n`, normalmente vale **subir macro** (`block-translate`, `lns`) e **baixar micro**.
* `--sa-chain` (modo “encadeado”: começa em `n=200`, remove 1 árvore e roda um SA curtíssimo na **casca**; repete para `n` menores)
  * `--sa-chain-base-iters 40` / `--sa-chain-iters-per-n 0` (iters por `n` no chain; usa os mesmos pesos do SA)
  * `--sa-chain-band-layers 2.5` (espessura da boundary-layer em “camadas” do spacing; interior fica congelado)
  * `--sa-chain-min-n 1` (até qual `n` aplicar SA; abaixo disso só faz as remoções)
* `--no-final-rigid` (desliga o pós-processamento “final rigid” por `n`; alias: `--no-sa-rigid`)
* `--no-prune` (desliga o pruning guloso e usa apenas o prefixo)
* `--seed 1` (reprodutibilidade)
* `--shift-a 0.0` / `--shift-b 0.0` / `--shift a,b` (offset do tile; ótimo pra sweeps/ensembling)
* `--output submission_tile_cpp.csv`

Saída padrão: `submission_tile_cpp.csv`. No exemplo acima: `runs/tmp/submission_tile_cpp.csv`.

### Simulador local de score

Também há um **simulador local de score** em C++ que lê um `submission.csv` no formato oficial, reconstrói as árvores como polígonos e calcula:

```text
score = ∑(s_n^2 / n)
```

onde `s_n` é o lado do quadrado axis-aligned que contém todas as árvores da instância `n`. Submissões com overlap geram erro local (similar à validação do Kaggle).

Compilar e rodar o simulador:

```bash
cmake --build build -j --target score_submission
./bin/score_submission submission.csv
```

Observação: a forma da árvore é definida em `get_tree_polygon()` (`src/geometry/geom.cpp`). Atualmente já usamos o polígono oficial (15 vértices) da árvore; se o Kaggle atualizar a geometria, ajuste aqui.

Depois, vamos evoluir esse baseline em C++ com heurísticas (SA, hill-climb, etc.) em cima das `TreePose`.

### Ensembling por instância (pegar o melhor por `n`)

Como cada `n` é uma instância independente, dá para combinar vários `submission*.csv` e, para cada `n`, escolher o que tiver menor `s_n` (sem overlap). Isso costuma dar um salto grande.

Compilar e rodar:

```bash
cmake --build build -j --target ensemble_submissions
./bin/ensemble_submissions submission_ensemble.csv run1.csv run2.csv run3.csv
./bin/score_submission submission_ensemble.csv
```

Observação: `ensemble_submissions` aplica por padrão um pós-processamento “final rigid” por `n` (rotação global que pode reduzir o quadrado axis-aligned). Para desligar: `--no-final-rigid`.
Detalhe: o “final rigid” é implementado em `src/geometry/geom.cpp` usando **convex hull + rotating calipers** (minimum bounding square), evitando depender de ângulos discretos.

### Sweep + blend (modo simples para ganhar score)

Se você quer rodar M variantes e fazer o blend por `n` automaticamente, use `scripts/sweep_blend.py`.

Placeholders do `--cmd`: `{seed} {shift_a} {shift_b} {out} {run}` + quaisquer variáveis definidas por `--set/--choice/--uniform/--randint`.

```bash
./scripts/sweep_blend.py --runs 10 --seed0 1 \
  --cmd './bin/solver_tessellation --seed {seed} --shift {shift_a},{shift_b} --output {out}' \
  --runs-dir runs_tess \
  --out submission_blended.csv
```

Exemplo variando parâmetros (tesselação + offsets + blocos de SA):

```bash
./scripts/sweep_blend.py --runs 20 --seed0 1 \
  --choice 'sa_block_size=4|6|8' \
  --choice 'angles=0,15,30,45,60,75|0,10,20,30,40,50,60' \
  --uniform spacing_safety=1.000,1.010 \
  --cmd './bin/solver_tessellation --seed {seed} --shift {shift_a},{shift_b} --sa-block-size {sa_block_size} --angles {angles} --spacing-safety {spacing_safety} --output {out}' \
  --runs-dir runs_tess \
  --out submission_blended.csv
```

Dica: você pode passar múltiplos `--cmd` para misturar solvers (por padrão é round-robin; use `--cmd-mode random` para escolher aleatoriamente).

Novo: `scripts/sweep_blend.py --blend-repair` roda `./bin/blend_repair` após o ensemble e scoreia o CSV reparado (saída default: `submission_repair.csv` se `--out submission_ensemble.csv`, senão `<out>_repair.csv`; log em `blend_repair.log`). Por padrão usa um preset forte (`--blend-repair-preset hunt`) e, se existir `submissions/submission_best.csv`, aplica `--base ... --target-top 20` para só mexer em um subconjunto do best.

### Blend + repair (top-K por n) + (SA curto) + final rigid

Depois do ensemble simples, dá para tentar um salto extra misturando **mais granularmente** e reparando colisões:

* para cada `n`, mantém as **top-K** soluções candidatas (por `s_n`),
* gera “filhos” substituindo um subconjunto de árvores da **borda** por poses vindas de outra candidata,
* roda um **repair** rápido para eliminar overlaps,
* opcionalmente roda um **SA curto** e aplica **final rigid**.

Compilar e rodar:

```bash
cmake --build build -j --target blend_repair
./bin/blend_repair submission_repair.csv runs_tess/run_*.csv \
  --topk-per-n 30 --blend-iters 200 \
  --boundary-topk 20 --replace-min 3 --replace-max 16 \
  --repair-passes 400 --repair-attempts 60 \
  --sa-iters 200 --sa-restarts 1
./bin/score_submission submission_repair.csv
```

Principais knobs: `--topk-per-n`, `--blend-iters`, `--repair-passes`, `--repair-attempts`, `--replace-min/--replace-max` e `--boundary-topk`.

[1]: https://www.competehub.dev/en/competitions/kagglesanta-2025 "Santa 2025 - Christmas Tree Packing Challenge - CompeteHub"
[2]: https://www.kaggle.com/code/jekiwantaufik/santa-2025-christmas-tree-packing-challenge?utm_source=chatgpt.com "Santa 2025 - Christmas Tree Packing Challenge"
[4]: https://www.kaggle.com/code/jekiwantaufik/santa-2025-christmas-tree-packing-challenge-2?utm_source=chatgpt.com "Santa 2025 - Christmas Tree Packing Challenge 2"
[5]: https://www.kaggle.com/code/koushikkumardinda/santa-2025-christmas-tree-packing-challenge?utm_source=chatgpt.com "Santa 2025 - Christmas Tree Packing Challenge"
[6]: https://www.kaggle.com/code/muhammadibrahim3093/ensembling-santa-2025?utm_source=chatgpt.com "Ensembling_Santa_2025"
