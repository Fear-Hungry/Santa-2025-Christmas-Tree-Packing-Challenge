# Architecture

## Data flow (solver_tile)
1. Generate a periodic pattern (motif + lattice) with tiling search.
2. Build a candidate pool and pick a prefix (central or greedy).
3. Optional prune and boundary refine to remove overlap and shrink the side.
4. Optional local search layers (SA / ILS / MZ) for n up to 200.
5. Optional final rigid rotation to reduce the bounding square.
6. Quantize to CSV format and write the submission.

## Modules (where the work lives)
- apps/: thin CLIs that parse flags, call into modules, and write output.
- src/solvers/tiling_pool.cpp: pattern search, spacing, pool generation.
- src/solvers/prefix_prune.cpp: prefix selection + prune + local repair paths.
- src/solvers/sa/sa.cpp: SARefiner implementation and move portfolio.
- src/solvers/sa/sa_refine.cpp: SARefiner local search and operator implementations.
- src/solvers/boundary_refine.cpp: boundary-only refinements on a pool.
- src/utils/submission_io.cpp: CSV parsing and quantization helpers.
- src/geometry/geom.cpp + src/geometry/collision.cpp + src/geometry/collision_polygons.cpp: geometry and overlap checks.

## Adding new operators (preferred locations)
- Tile/pattern moves: src/solvers/tiling_pool.cpp (optimize_tile_by_spacing).
- Prefix / prune logic: src/solvers/prefix_prune.cpp (build_* helpers).
- SA moves or acceptance logic: src/solvers/sa/sa.cpp (SARefiner move blocks).
- Output/quantization tweaks: src/utils/submission_io.cpp.

## Notes
- Keep CLI flags stable; add new flags instead of changing semantics.
- Run regression checks before and after each refactor step.
