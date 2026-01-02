from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from geom_np import polygon_bbox, polygon_radius, shift_poses_to_origin, transform_polygon
from scoring import polygons_intersect


def _packing_score_from_bboxes(bboxes: np.ndarray) -> float:
    min_x = float(np.min(bboxes[:, 0]))
    min_y = float(np.min(bboxes[:, 1]))
    max_x = float(np.max(bboxes[:, 2]))
    max_y = float(np.max(bboxes[:, 3]))
    return float(max(max_x - min_x, max_y - min_y))


def _has_overlaps(points: np.ndarray, poses: np.ndarray) -> bool:
    poses = np.array(poses, dtype=float, copy=False)
    if poses.shape[0] <= 1:
        return False

    centers = poses[:, :2]
    rad = float(polygon_radius(points))
    thr2 = (2.0 * rad) ** 2

    polys = [transform_polygon(points, pose) for pose in poses]
    n = poses.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            dx = float(centers[i, 0] - centers[j, 0])
            dy = float(centers[i, 1] - centers[j, 1])
            if dx * dx + dy * dy > thr2:
                continue
            if polygons_intersect(polys[i], polys[j]):
                return True
    return False


def has_overlaps(points: np.ndarray, poses: np.ndarray) -> bool:
    """Public overlap checker used by sweep/ensemble scripts."""
    return _has_overlaps(points, poses)


@dataclass
class _PackingState:
    poses: np.ndarray
    centers: np.ndarray
    polys: list[np.ndarray]
    bboxes: np.ndarray
    score: float
    thr2: float


def _build_state(points: np.ndarray, poses: np.ndarray) -> _PackingState:
    poses = np.array(poses, dtype=float, copy=True)
    polys = [transform_polygon(points, pose) for pose in poses]
    bboxes = np.stack([polygon_bbox(p) for p in polys], axis=0)
    score = _packing_score_from_bboxes(bboxes)
    rad = float(polygon_radius(points))
    thr2 = (2.0 * rad) ** 2
    return _PackingState(
        poses=poses,
        centers=poses[:, :2].copy(),
        polys=polys,
        bboxes=bboxes,
        score=score,
        thr2=thr2,
    )


def _state_has_overlaps(state: _PackingState) -> bool:
    n = state.poses.shape[0]
    if n <= 1:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = float(state.centers[i, 0] - state.centers[j, 0])
            dy = float(state.centers[i, 1] - state.centers[j, 1])
            if dx * dx + dy * dy > state.thr2:
                continue
            if polygons_intersect(state.polys[i], state.polys[j]):
                return True
    return False


def _collides_one_vs_all(state: _PackingState, idx: int, cand_center: np.ndarray, cand_poly: np.ndarray) -> bool:
    n = state.poses.shape[0]
    for j in range(n):
        if j == idx:
            continue
        dx = float(cand_center[0] - state.centers[j, 0])
        dy = float(cand_center[1] - state.centers[j, 1])
        if dx * dx + dy * dy > state.thr2:
            continue
        if polygons_intersect(cand_poly, state.polys[j]):
            return True
    return False


def hill_climb(
    points: np.ndarray,
    poses: np.ndarray,
    *,
    step_xy: float = 0.01,
    step_deg: float = 2.0,
    max_passes: int = 2,
    tol: float = 1e-12,
) -> np.ndarray:
    """Deterministic local search: for each tree, try +/-x, +/-y, +/-deg and accept improvements only."""

    state = _build_state(points, poses)
    n = state.poses.shape[0]
    if n <= 1:
        return shift_poses_to_origin(points, state.poses)

    moves: list[tuple[float, float, float]] = [
        (step_xy, 0.0, 0.0),
        (-step_xy, 0.0, 0.0),
        (0.0, step_xy, 0.0),
        (0.0, -step_xy, 0.0),
        (0.0, 0.0, step_deg),
        (0.0, 0.0, -step_deg),
    ]

    for _pass in range(max_passes):
        improved_pass = False
        for idx in range(n):
            best_score = state.score
            best_pose: np.ndarray | None = None
            best_poly: np.ndarray | None = None
            best_bbox: np.ndarray | None = None

            base_pose = state.poses[idx].copy()
            for dx, dy, ddeg in moves:
                cand_pose = base_pose.copy()
                cand_pose[0] += dx
                cand_pose[1] += dy
                cand_pose[2] = float(math.fmod(cand_pose[2] + ddeg, 360.0))
                if cand_pose[2] < 0.0:
                    cand_pose[2] += 360.0

                cand_poly = transform_polygon(points, cand_pose)
                cand_center = cand_pose[:2]
                if _collides_one_vs_all(state, idx, cand_center, cand_poly):
                    continue

                cand_bbox = polygon_bbox(cand_poly)
                tmp = state.bboxes.copy()
                tmp[idx] = cand_bbox
                cand_score = _packing_score_from_bboxes(tmp)

                if cand_score + tol < best_score:
                    best_score = cand_score
                    best_pose = cand_pose
                    best_poly = cand_poly
                    best_bbox = cand_bbox

            if best_pose is not None:
                state.poses[idx] = best_pose
                state.centers[idx] = best_pose[:2]
                state.polys[idx] = best_poly  # type: ignore[assignment]
                state.bboxes[idx] = best_bbox  # type: ignore[assignment]
                state.score = best_score
                improved_pass = True

        if not improved_pass:
            break

    return shift_poses_to_origin(points, state.poses)


def _pick_boundary_tree(
    bboxes: np.ndarray,
    *,
    rng: np.random.Generator,
    k: int = 8,
) -> tuple[int, int, np.ndarray]:
    """Pick a tree near the current packing AABB boundary.

    Returns: (idx, axis, center_xy) where axis is 0 (x) or 1 (y).
    """
    min_x = float(np.min(bboxes[:, 0]))
    min_y = float(np.min(bboxes[:, 1]))
    max_x = float(np.max(bboxes[:, 2]))
    max_y = float(np.max(bboxes[:, 3]))
    width = max_x - min_x
    height = max_y - min_y
    axis = 0 if width >= height else 1
    center = np.array([(min_x + max_x) * 0.5, (min_y + max_y) * 0.5], dtype=float)

    if axis == 0:
        slack = np.minimum(bboxes[:, 0] - min_x, max_x - bboxes[:, 2])
        scale = max(width, 1e-9)
    else:
        slack = np.minimum(bboxes[:, 1] - min_y, max_y - bboxes[:, 3])
        scale = max(height, 1e-9)
    slack = np.maximum(slack, 0.0)

    # Sample from the k most boundary-ish trees to avoid always picking the same index.
    order = np.argsort(slack, kind="mergesort")
    topk = order[: max(1, min(int(k), order.shape[0]))]
    weights = np.exp(-(slack[topk] / scale) * 8.0)
    weights = weights / np.sum(weights)
    idx = int(rng.choice(topk, p=weights))
    return idx, axis, center


def _pick_boundary_tree_from_centers(
    centers: np.ndarray,
    *,
    rng: np.random.Generator,
    k: int = 8,
) -> tuple[int, int, np.ndarray]:
    min_x = float(np.min(centers[:, 0]))
    min_y = float(np.min(centers[:, 1]))
    max_x = float(np.max(centers[:, 0]))
    max_y = float(np.max(centers[:, 1]))
    width = max_x - min_x
    height = max_y - min_y
    axis = 0 if width >= height else 1
    center = np.array([(min_x + max_x) * 0.5, (min_y + max_y) * 0.5], dtype=float)

    if axis == 0:
        slack = np.minimum(centers[:, 0] - min_x, max_x - centers[:, 0])
        scale = max(width, 1e-9)
    else:
        slack = np.minimum(centers[:, 1] - min_y, max_y - centers[:, 1])
        scale = max(height, 1e-9)
    slack = np.maximum(slack, 0.0)

    order = np.argsort(slack, kind="mergesort")
    topk = order[: max(1, min(int(k), order.shape[0]))]
    weights = np.exp(-(slack[topk] / scale) * 8.0)
    weights = weights / np.sum(weights)
    idx = int(rng.choice(topk, p=weights))
    return idx, axis, center


def _repair_overlaps(
    points: np.ndarray,
    poses: np.ndarray,
    *,
    rng: np.random.Generator,
    max_iters: int = 200,
    step_xy: float = 0.01,
    step_deg: float = 0.0,
) -> np.ndarray | None:
    """Try to repair overlaps by nudging colliding trees apart."""
    state = _build_state(points, poses)
    n = state.poses.shape[0]
    if n <= 1:
        return shift_poses_to_origin(points, state.poses)

    for _ in range(max_iters):
        pair: tuple[int, int] | None = None
        for i in range(n):
            for j in range(i + 1, n):
                dx = float(state.centers[i, 0] - state.centers[j, 0])
                dy = float(state.centers[i, 1] - state.centers[j, 1])
                if dx * dx + dy * dy > state.thr2:
                    continue
                if polygons_intersect(state.polys[i], state.polys[j]):
                    pair = (i, j)
                    break
            if pair is not None:
                break
        if pair is None:
            return shift_poses_to_origin(points, state.poses)

        i, j = pair
        move = i if rng.random() < 0.5 else j
        other = j if move == i else i
        direction = state.centers[move] - state.centers[other]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            ang = float(rng.uniform(0.0, 2.0 * math.pi))
            direction = np.array([math.cos(ang), math.sin(ang)], dtype=float)
            norm = 1.0
        unit = direction / norm
        delta_xy = unit * float(step_xy)
        noise = rng.normal(0.0, float(step_xy) * 0.25, size=(2,))
        cand_pose = state.poses[move].copy()
        cand_pose[0:2] = cand_pose[0:2] + delta_xy + noise
        if step_deg != 0.0:
            cand_pose[2] = float(math.fmod(cand_pose[2] + rng.normal(0.0, float(step_deg)), 360.0))
            if cand_pose[2] < 0.0:
                cand_pose[2] += 360.0

        cand_poly = transform_polygon(points, cand_pose)
        cand_center = cand_pose[:2]
        if _collides_one_vs_all(state, move, cand_center, cand_poly):
            continue

        state.poses[move] = cand_pose
        state.centers[move] = cand_center
        state.polys[move] = cand_poly
        state.bboxes[move] = polygon_bbox(cand_poly)

    return None


def genetic_optimize(
    points: np.ndarray,
    seeds: Sequence[np.ndarray],
    *,
    seed: int = 1,
    pop_size: int = 24,
    generations: int = 20,
    elite_frac: float = 0.25,
    crossover_prob: float = 0.5,
    mutation_sigma_xy: float = 0.01,
    mutation_sigma_deg: float = 2.0,
    directed_mut_prob: float = 0.5,
    directed_step_xy: float = 0.02,
    directed_k: int = 8,
    repair_iters: int = 200,
    hill_climb_passes: int = 0,
    hill_climb_step_xy: float = 0.01,
    hill_climb_step_deg: float = 2.0,
    max_child_attempts: int = 50,
) -> np.ndarray:
    """Simple GA for 1 instance: selection + (optional) crossover + mutations (+ optional hill-climb).

    Keeps solutions feasible by repairing (or retrying) colliding children.
    """

    if not seeds:
        raise ValueError("empty seeds")
    rng = np.random.default_rng(int(seed))

    seeds_arr = [np.array(p, dtype=float, copy=True) for p in seeds]
    n = int(seeds_arr[0].shape[0])
    for p in seeds_arr:
        if p.shape != (n, 3):
            raise ValueError("all seeds must have the same shape (n,3)")

    def _eval(p: np.ndarray) -> float:
        state = _build_state(points, p)
        if _state_has_overlaps(state):
            return float("inf")
        return state.score

    def _tournament(pop: list[np.ndarray], scores: np.ndarray, k: int = 3) -> np.ndarray:
        idxs = rng.integers(0, len(pop), size=(k,))
        best = idxs[0]
        for ii in idxs[1:]:
            if scores[ii] < scores[best]:
                best = ii
        return pop[int(best)]

    def _mutate_directed(p: np.ndarray) -> np.ndarray:
        child = np.array(p, dtype=float, copy=True)
        idx, axis, center = _pick_boundary_tree_from_centers(child[:, :2], rng=rng, k=directed_k)

        drift = np.zeros((2,), dtype=float)
        diff = center - child[idx, :2]
        if axis == 0:
            drift[0] = math.copysign(directed_step_xy, float(diff[0])) if abs(float(diff[0])) > 1e-12 else float(directed_step_xy)
        else:
            drift[1] = math.copysign(directed_step_xy, float(diff[1])) if abs(float(diff[1])) > 1e-12 else float(directed_step_xy)

        noise = rng.normal(0.0, mutation_sigma_xy, size=(2,))
        child[idx, 0:2] = child[idx, 0:2] + drift + noise
        child[idx, 2] = float(math.fmod(child[idx, 2] + rng.normal(0.0, mutation_sigma_deg), 360.0))
        if child[idx, 2] < 0.0:
            child[idx, 2] += 360.0
        return child

    def _mutate_random(p: np.ndarray) -> np.ndarray:
        child = np.array(p, dtype=float, copy=True)
        idx = int(rng.integers(0, n))
        child[idx, 0:2] = child[idx, 0:2] + rng.normal(0.0, mutation_sigma_xy, size=(2,))
        child[idx, 2] = float(math.fmod(child[idx, 2] + rng.normal(0.0, mutation_sigma_deg), 360.0))
        if child[idx, 2] < 0.0:
            child[idx, 2] += 360.0
        return child

    # --- Initialize population
    population: list[np.ndarray] = []
    for p in seeds_arr:
        population.append(shift_poses_to_origin(points, p))
        if len(population) >= pop_size:
            break

    base = population[0]
    while len(population) < pop_size:
        cand = _mutate_directed(base) if rng.random() < directed_mut_prob else _mutate_random(base)
        repaired = _repair_overlaps(points, cand, rng=rng, max_iters=repair_iters, step_xy=mutation_sigma_xy)
        if repaired is None:
            continue
        population.append(repaired)

    scores = np.array([_eval(p) for p in population], dtype=float)

    elite_n = max(1, int(round(pop_size * float(elite_frac))))
    best_pose = population[int(np.argmin(scores))]
    best_score = float(np.min(scores))

    # --- Evolve
    for _gen in range(generations):
        order = np.argsort(scores, kind="mergesort")
        population = [population[int(i)] for i in order]
        scores = scores[order]

        next_pop: list[np.ndarray] = [population[i].copy() for i in range(elite_n)]

        while len(next_pop) < pop_size:
            parent_a = _tournament(population, scores)
            parent_b = _tournament(population, scores)

            if rng.random() < crossover_prob:
                mask = rng.random(n) < 0.5
                child = np.where(mask[:, None], parent_a, parent_b).astype(float, copy=True)
            else:
                child = parent_a.copy()

            for _attempt in range(max_child_attempts):
                cand = _mutate_directed(child) if rng.random() < directed_mut_prob else _mutate_random(child)
                repaired = _repair_overlaps(points, cand, rng=rng, max_iters=repair_iters, step_xy=mutation_sigma_xy)
                if repaired is None:
                    continue
                cand = repaired
                if hill_climb_passes > 0:
                    cand = hill_climb(
                        points,
                        cand,
                        step_xy=hill_climb_step_xy,
                        step_deg=hill_climb_step_deg,
                        max_passes=hill_climb_passes,
                    )
                child = cand
                break

            next_pop.append(child)

        population = next_pop
        scores = np.array([_eval(p) for p in population], dtype=float)
        gen_best_idx = int(np.argmin(scores))
        gen_best = float(scores[gen_best_idx])
        if gen_best < best_score:
            best_score = gen_best
            best_pose = population[gen_best_idx]

    if hill_climb_passes > 0 and not math.isinf(best_score):
        best_pose = hill_climb(
            points,
            best_pose,
            step_xy=hill_climb_step_xy,
            step_deg=hill_climb_step_deg,
            max_passes=hill_climb_passes,
        )
    return shift_poses_to_origin(points, best_pose)
