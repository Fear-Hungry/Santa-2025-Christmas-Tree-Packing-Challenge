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

- 2025-12-27: `score = 87.619057066525926` (`runs/try_sa60k_r16_n200_20251227_124010/submission.csv`)
- 2025-12-26: `score = 89.033538667437242` (`runs/best_89/submission.csv`)
- CLIs usados para reproduzir:

```bash
cmake -S . -B build
cmake --build build -j

OUT_DIR=runs/try_sa60k_r16_n200_20251227_124010
mkdir -p "$OUT_DIR"
./bin/solve_all --out "$OUT_DIR/submission.csv" --out-dir "$OUT_DIR" --out-json "$OUT_DIR/run.json" \
  --nmax 200 --init bottom-left --refine sa --warm-start-feed initial \
  --runs-per-n 16 --threads 16 \
  --sa-iters 60000 --sa-iters-mode linear --sa-tries 8 \
  --seed 1 --angles 0,45,90,135,180,225,270,315 --gap 1e-6 --safety-eps 0 \
  --compact-final --compact-passes 1 --log-every 50

./bin/score_submission "$OUT_DIR/submission.csv"
```

## Benchmarks rápidos (lab)

Benchmarks curtos (menor é melhor) para comparar heurísticas sem gastar o budget do `n=200`.
O score é sempre calculado com `./bin/score_submission` (fonte de verdade).

### 2025-12-26 — LNS gap destroy (n<=40)

Comparação controlada (seeds fixas, budget fixo):

- Baseline: commit `c4704ce` (pré-merge)
  - seeds 1,2,3: `21.396470160711008`, `21.511813315095495`, `21.2637152072427`
  - mean: `21.390666227683067`
- Candidate: commit `5e188ae` + `--lns-destroy-mode gap`
  - seeds 1,2,3: `21.35681648803555`, `21.31341494631857`, `21.322161260198037`
  - mean: `21.33079756485072` (Δ `-0.05986866283234704`)

CLI (candidate, HEAD) usado:

```bash
cmake -S . -B build
cmake --build build -j

# Repetir para seed=1,2,3
OUT_DIR=runs/lab_lns_gap_n40/head_seed1
mkdir -p "$OUT_DIR"
./bin/solve_all --out "$OUT_DIR/submission.csv" --out-dir "$OUT_DIR" --out-json "$OUT_DIR/run.json" \
  --nmax 40 --init bottom-left --refine none --seed 1 --threads 1 \
  --angles 0,90,180,270 --gap 1e-6 --safety-eps 0 \
  --lns --lns-stages 2 --lns-stage-attempts 5 --lns-remove-frac 0.12 --lns-boundary-prob 0.7 \
  --lns-destroy-mode gap --lns-gap-grid 48 --lns-gap-try-hole-center 1 \
  --lns-slide-iters 60 --lns-shrink-factor 0.999 --lns-shrink-delta 0 \
  --log-every 0
./bin/score_submission "$OUT_DIR/submission.csv" --nmax 40 --breakdown
```

CLI (baseline, commit `c4704ce`) usado:

```bash
# Rodar baseline em worktree separado (para não mexer no checkout atual)
git worktree add --detach runs/_worktrees/c4704ce c4704ce
( cd runs/_worktrees/c4704ce && cmake -S . -B build && cmake --build build -j )

# Repetir para seed=1,2,3
OUT_DIR=runs/lab_lns_gap_n40/baseline_seed1
mkdir -p "$OUT_DIR"
runs/_worktrees/c4704ce/bin/solve_all --out "$OUT_DIR/submission.csv" --out-dir "$OUT_DIR" --out-json "$OUT_DIR/run.json" \
  --nmax 40 --init bottom-left --refine none --seed 1 --threads 1 \
  --angles 0,90,180,270 --gap 1e-6 --safety-eps 0 \
  --lns --lns-stages 2 --lns-stage-attempts 5 --lns-remove-frac 0.12 --lns-boundary-prob 0.7 \
  --lns-slide-iters 60 --lns-shrink-factor 0.999 --lns-shrink-delta 0 \
  --log-every 0

# Score sempre com o scorer do HEAD (para isolar mudanças do solver)
./bin/score_submission "$OUT_DIR/submission.csv" --nmax 40 --breakdown
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

### Budgets por faixa de `n` (experimento)

Para gastar mais compute em `n` médios/grandes (e menos em `n` pequenos), use as flags:

- `--sa-iters-ranges lo-hi=value,...` (override do `--sa-iters` por faixa)
- `--runs-per-n-ranges lo-hi=value,...` (override do `--runs-per-n` por faixa)

O último match vence (“last match wins”). Faixas fora do `--nmax` são ignoradas.

Exemplo (mais multi-start + mais SA conforme `n` cresce):

```bash
OUT_DIR=runs/ranges_20251227
mkdir -p "$OUT_DIR"

./bin/solve_all --out "$OUT_DIR/submission.csv" --out-dir "$OUT_DIR" --out-json "$OUT_DIR/run.json" \
  --nmax 200 --init bottom-left --refine sa --warm-start-feed initial \
  --angles 0,45,90,135,180,225,270,315 --gap 1e-6 --safety-eps 0 \
  --sa-iters-mode linear --sa-iters 20000 \
  --sa-iters-ranges 1-80=30000,81-160=60000,161-200=120000 \
  --runs-per-n 2 --runs-per-n-ranges 1-80=4,81-160=8,161-200=16 \
  --threads 16 --seed 1 --log-every 10

./bin/score_submission "$OUT_DIR/submission.csv"
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

Se você quiser fazer “pós-processamento” só em uma faixa pequena (ex.: otimizar `n<=30` com ângulos mais densos,
ou até bruteforce/exato para `n` muito pequenos) e depois **colar** isso em cima de uma solução completa,
use `--allow-partial` no merge:

```bash
./bin/solve_all --out base.csv --nmax 200 --seed 1 --init bottom-left --refine none --angles 0,45,90,135,180,225,270,315
./bin/solve_all --out small.csv --nmax 30  --seed 1 --init bottom-left --refine none --angles 0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345

./bin/merge_submissions --out merged.csv --nmax 200 --allow-partial base.csv small.csv --out-json merge_report.json
./bin/score_submission merged.csv --breakdown
```

Para automatizar um “portfolio” de configs (ex.: 0°/180°, 8 ângulos, e ângulos densos para poucos `n`) e já
gerar o merged final em `runs/`, use:

```bash
python3 scripts/portfolio_merge.py --tag portfolio1 --nmax 200 --dense-nmax 30 --dense-step 5
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
