# Santa 2025 — Christmas Tree Packing (C++ only)

Base C++20 (sem dependências) para modelagem geométrica, checagem de colisão e heurísticas construtivas do desafio **Santa 2025 – Christmas Tree Packing**.

## Forma da árvore (polígono oficial)

O polígono base (15 vértices) está hardcoded em `include/santa2025/tree_polygon.hpp`:

```cpp
{0.0, 0.8},
{0.125, 0.5},
{0.0625, 0.5},
{0.2, 0.25},
{0.1, 0.25},
{0.35, 0.0},
{0.075, 0.0},
{0.075, -0.2},
{-0.075, -0.2},
{-0.075, 0.0},
{-0.35, 0.0},
{-0.1, 0.25},
{-0.2, 0.25},
{-0.0625, 0.5},
{-0.125, 0.5},
```

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

Binários saem em `./bin/`.

## Melhor score local (até agora)

**Menor é melhor.** Medido com `./bin/score_submission` para `n=1..200`, sem merge e sem pós-processamento de ordem.

- 2025-12-26: `score = 89.033538667437242`
- CLIs usados para reproduzir:

```bash
cmake -S . -B build
cmake --build build -j

./bin/solve_all --out runs/best_89/submission.csv --out-dir runs/best_89 --nmax 200 --init bottom-left --refine none \
  --seed 1 --angles 0,45,90,135,180,225,270,315 --gap 1e-6 --safety-eps 0

./bin/score_submission runs/best_89/submission.csv --breakdown
```

## Ferramentas

Inspecionar propriedades (área, bounding box e melhor rotação para minimizar o bounding square):

```bash
./bin/tree_info
```

Validar NFP (no-fit polygon) contra checagem geométrica direta (interseção de polígonos):

```bash
./bin/nfp_sanity --samples 200
```

## Construção de solução inicial

### Bottom-left (NFP + slide)

```bash
./bin/bottom_left_pack --n 200
```

Alternar orientações “complementares” (ex.: 0° e 180°):

```bash
./bin/bottom_left_pack --n 200 --mode cycle --cycle 0,180 --angles 0,180
```

### Grid + shake-down (baseline simples)

Gera uma grade inicial (sem overlaps) e faz passes de “shake-down” (desliza down/left cada item até encostar):

```bash
./bin/grid_shake_pack --n 200
```

## Refino por Simulated Annealing (SA)

Roda SA em cima de uma solução inicial (`bottom-left` ou `grid-shake`) para tentar reduzir o bounding square.

Por padrão, a rotação é tratada como **conjunto discreto de ângulos** (`--angles`) para evitar explosão do cache de NFP.

```bash
./bin/sa_opt --init bottom-left --n 200 --iters 200000 --seed 1 --log-every 5000
```

Schedule alternativo (polinomial) e opção de escala adaptiva (alvo de taxa de aceitação):

```bash
./bin/sa_opt --init bottom-left --n 200 --schedule poly --poly-power 2 --adaptive-window 200 --target-accept 0.35
```

Aceitação com Δ baseado em `s²` (e tie-breaker quando `s` não muda):

```bash
./bin/sa_opt --init bottom-left --n 200 --delta-mode squared_over_s --secondary perimeter --secondary-weight 0.01
```

Moves para “escapar de mínimos locais” (cluster + foco em árvores de borda + touch-best-of):

```bash
./bin/sa_opt --init bottom-left --n 200 --cluster-prob 0.25 --cluster-min 3 --cluster-max 8 --boundary-prob 0.7 --touch-best-of 4
```

Multi-start (várias rodadas com seeds diferentes) e execução em paralelo:

```bash
./bin/sa_opt --inits bottom-left,grid-shake --n 200 --runs 8 --threads 8 --seed 1 --iters 200000
```

Iterative shrink-wrapping (tenta “fechar” o lado do quadrado por estágios):

```bash
./bin/sa_opt --init bottom-left --n 200 --shrink-wrap --shrink-stage-iters 5000 --shrink-factor 0.999 --shrink-delta 0.0
```

LNS (Large Neighborhood Search) como pós-pass (destroy & repair) para tentar “fechar” ainda mais:

```bash
./bin/sa_opt --init bottom-left --n 200 --iters 200000 --seed 1 --lns --lns-stages 50 --lns-stage-attempts 50 --lns-remove-frac 0.10 --lns-shrink-factor 0.999
```

## Submissão (CSV) e verificação

### Solve-all (puzzles independentes)

Se você quiser otimizar cada puzzle `n` separadamente (sem impor que `n=1..200` sejam prefixos do `n=200`),
use `solve_all` para gerar um `submission.csv` completo (20100 linhas):

```bash
./bin/solve_all --out submission.csv --seed 1 --refine sa --runs-per-n 1 --threads 8
```

Validar/score local:

```bash
./bin/score_submission submission.csv --breakdown
```

Para “hardening” (evitar overlaps microscópicos no avaliador), você pode exigir uma separação mínima:

```bash
./bin/score_submission submission.csv --min-sep 1e-6
```

### Merge (best-of por puzzle)

Se diferentes runs/estratégias produzem soluções melhores em `n` diferentes, você pode combinar CSVs escolhendo
o melhor bloco por puzzle (menor `s_n`, equivalente a menor `s_n²/n`):

```bash
./bin/merge_submissions --out merged.csv a.csv b.csv c.csv --out-json merge_report.json
./bin/score_submission merged.csv --breakdown
```

### Prefixo de `n=200` (modo prefix)

Gerar `submission.csv` (puzzles `001..200`) como **prefixos** do packing com `n=200`:

```bash
./bin/sa_opt --init bottom-left --n 200 --iters 200000 --seed 1 --out-csv submission.csv --csv-nmax 200 --csv-precision 17
```

Score local + checagem de overlaps (usa `polygons_overlap_strict`, então “touch” é permitido, overlap com área > 0 não):

```bash
./bin/score_submission submission.csv --breakdown
```

## Pós-pass: otimização da ordem (prefix-score)

Este pós-pass só faz sentido no **modo prefix** acima: com a geometria fixa (mesmas posições/rotações),
**a ordem das árvores** afeta o score porque cada puzzle `n` usa apenas as `n` primeiras árvores.

O binário `order_opt` lê um `submission.csv`, pega o packing do puzzle `N` (default `200`), e roda uma hiper-heurística
barata no espaço de **permutações** (swap/reverse/shuffle/reinsert) para reduzir o **prefix-score**:

```bash
./bin/order_opt --in submission.csv --out submission_ordered.csv --iters 50000 --seed 1
./bin/score_submission submission_ordered.csv --breakdown
```

## Colisão e índice espacial

- `santa2025::NFPCache` + `trees_overlap_nfp()` em `include/santa2025/nfp.hpp`
- `santa2025::SpatialHashGrid` em `include/santa2025/spatial_hash_grid.hpp`
- `santa2025::CollisionIndex` em `include/santa2025/collision_index.hpp` (grid + NFP)
- `santa2025::lns_shrink_wrap()` em `include/santa2025/lns.hpp`

## Restrições de coordenadas

Helpers para manter `x,y ∈ [-100, 100]`:

- `santa2025::within_coord_bounds()` e `santa2025::clamp_pose_xy()` em `include/santa2025/constraints.hpp`

## Métrica (referência)

No desafio, o score (minimização) costuma ser definido como:

\[
\text{score} = \sum_{n=1}^{200}\frac{s_n^2}{n}
\]

onde `s_n = max(width, height)` do bounding box axis-aligned de todos os vértices das `n` árvores.
