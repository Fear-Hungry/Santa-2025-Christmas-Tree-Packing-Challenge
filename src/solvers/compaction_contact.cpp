#include "compaction_contact.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "collision.hpp"
#include "geom.hpp"
#include "spatial_grid.hpp"
#include "submission_io.hpp"

namespace compaction_contact {
namespace {

struct Extents {
    double min_x;
    double max_x;
    double min_y;
    double max_y;
};

Extents compute_extents(const std::vector<BoundingBox>& bbs) {
    Extents e{std::numeric_limits<double>::infinity(),
              -std::numeric_limits<double>::infinity(),
              std::numeric_limits<double>::infinity(),
              -std::numeric_limits<double>::infinity()};
    for (const auto& bb : bbs) {
        e.min_x = std::min(e.min_x, bb.min_x);
        e.max_x = std::max(e.max_x, bb.max_x);
        e.min_y = std::min(e.min_y, bb.min_y);
        e.max_y = std::max(e.max_y, bb.max_y);
    }
    return e;
}

double side_from_extents(const Extents& e) {
    return std::max(e.max_x - e.min_x, e.max_y - e.min_y);
}

double min_dim_from_extents(const Extents& e) {
    return std::min(e.max_x - e.min_x, e.max_y - e.min_y);
}

double area_from_extents(const Extents& e) {
    return (e.max_x - e.min_x) * (e.max_y - e.min_y);
}

bool aabb_overlap(const BoundingBox& a, const BoundingBox& b) {
    if (a.max_x < b.min_x || b.max_x < a.min_x) {
        return false;
    }
    if (a.max_y < b.min_y || b.max_y < a.min_y) {
        return false;
    }
    return true;
}

std::vector<BoundingBox> bounding_boxes_for_poses(const Polygon& base_poly,
                                                  const std::vector<TreePose>& poses) {
    std::vector<BoundingBox> bbs;
    bbs.reserve(poses.size());
    for (const auto& pose : poses) {
        bbs.push_back(bounding_box(transform_polygon(base_poly, pose)));
    }
    return bbs;
}

std::vector<int> build_extreme_pool(const std::vector<BoundingBox>& bbs, int topk) {
    const int n = static_cast<int>(bbs.size());
    if (n <= 0) {
        return {};
    }
    topk = std::max(1, std::min(topk, n));

    std::vector<int> idx(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        idx[static_cast<size_t>(i)] = i;
    }

    std::vector<char> mark(static_cast<size_t>(n), 0);
    auto add = [&](auto cmp) {
        std::vector<int> v = idx;
        std::sort(v.begin(), v.end(), cmp);
        for (int i = 0; i < topk; ++i) {
            mark[static_cast<size_t>(v[static_cast<size_t>(i)])] = 1;
        }
    };

    add([&](int a, int b) {
        return bbs[static_cast<size_t>(a)].min_x <
               bbs[static_cast<size_t>(b)].min_x;
    });
    add([&](int a, int b) {
        return bbs[static_cast<size_t>(a)].max_x >
               bbs[static_cast<size_t>(b)].max_x;
    });
    add([&](int a, int b) {
        return bbs[static_cast<size_t>(a)].min_y <
               bbs[static_cast<size_t>(b)].min_y;
    });
    add([&](int a, int b) {
        return bbs[static_cast<size_t>(a)].max_y >
               bbs[static_cast<size_t>(b)].max_y;
    });

    std::vector<int> pool;
    pool.reserve(static_cast<size_t>(4 * topk));
    for (int i = 0; i < n; ++i) {
        if (mark[static_cast<size_t>(i)]) {
            pool.push_back(i);
        }
    }
    if (pool.empty()) {
        pool = idx;
    }
    return pool;
}

struct Eval {
    double side = 0.0;
    double area = 0.0;
};

bool improves_eval(const Eval& cand, const Eval& best, double plateau_eps) {
    if (!std::isfinite(best.side)) {
        return std::isfinite(cand.side);
    }
    if (cand.side < best.side - 1e-12) {
        return true;
    }
    if (cand.side <= best.side + plateau_eps && cand.area < best.area - 1e-12) {
        return true;
    }
    return false;
}

Point normalize_dir(const Point& p) {
    const double norm = std::hypot(p.x, p.y);
    if (!(norm > 1e-12)) {
        return Point{0.0, 0.0};
    }
    return Point{p.x / norm, p.y / norm};
}

bool push_to_contact_dir(const Polygon& base_poly,
                         const std::vector<TreePose>& poses,
                         const std::vector<Polygon>& polys,
                         const std::vector<BoundingBox>& bbs,
                         const UniformGridIndex& grid,
                         int idx,
                         const Point& dir_in,
                         double max_step,
                         int bisect_iters,
                         double thr_sq,
                         int quantize_decimals,
                         TreePose& pose_out,
                         Polygon& poly_out,
                         BoundingBox& bb_out,
                         double& delta_out) {
    Point dir = normalize_dir(dir_in);
    if (!(max_step > 1e-12) ||
        (std::abs(dir.x) < 1e-12 && std::abs(dir.y) < 1e-12)) {
        return false;
    }

    const TreePose& base_pose = poses[static_cast<size_t>(idx)];
    double t_max = max_step;
    if (dir.x > 1e-12) {
        t_max = std::min(t_max, (100.0 - base_pose.x) / dir.x);
    } else if (dir.x < -1e-12) {
        t_max = std::min(t_max, (-100.0 - base_pose.x) / dir.x);
    }
    if (dir.y > 1e-12) {
        t_max = std::min(t_max, (100.0 - base_pose.y) / dir.y);
    } else if (dir.y < -1e-12) {
        t_max = std::min(t_max, (-100.0 - base_pose.y) / dir.y);
    }

    if (!(t_max > 1e-12)) {
        return false;
    }

    std::vector<int> neigh;
    neigh.reserve(32);

    auto valid_at = [&](double delta,
                        TreePose& cand_pose,
                        Polygon& cand_poly,
                        BoundingBox& cand_bb) -> bool {
        cand_pose = base_pose;
        cand_pose.x += dir.x * delta;
        cand_pose.y += dir.y * delta;
        cand_pose = quantize_pose(cand_pose, quantize_decimals);
        if (cand_pose.x < -100.0 || cand_pose.x > 100.0 ||
            cand_pose.y < -100.0 || cand_pose.y > 100.0) {
            return false;
        }

        cand_poly = transform_polygon(base_poly, cand_pose);
        cand_bb = bounding_box(cand_poly);

        grid.gather(cand_pose.x, cand_pose.y, neigh);
        for (int j : neigh) {
            if (j == idx) {
                continue;
            }
            const double dx = cand_pose.x - poses[static_cast<size_t>(j)].x;
            const double dy = cand_pose.y - poses[static_cast<size_t>(j)].y;
            if (dx * dx + dy * dy > thr_sq) {
                continue;
            }
            if (!aabb_overlap(cand_bb, bbs[static_cast<size_t>(j)])) {
                continue;
            }
            if (polygons_intersect(cand_poly, polys[static_cast<size_t>(j)])) {
                return false;
            }
        }
        return true;
    };

    TreePose best_pose;
    Polygon best_poly;
    BoundingBox best_bb;
    double best_delta = 0.0;

    TreePose pose_hi;
    Polygon poly_hi;
    BoundingBox bb_hi;
    if (valid_at(t_max, pose_hi, poly_hi, bb_hi)) {
        best_pose = pose_hi;
        best_poly = std::move(poly_hi);
        best_bb = bb_hi;
        best_delta = t_max;
    } else {
        double hi_invalid = t_max;
        double lo_valid = 0.0;
        TreePose pose_lo;
        Polygon poly_lo;
        BoundingBox bb_lo;
        bool found = false;
        double step = t_max;
        for (int bt = 0; bt < 14; ++bt) {
            step *= 0.5;
            if (!(step > 1e-12)) {
                break;
            }
            if (valid_at(step, pose_lo, poly_lo, bb_lo)) {
                lo_valid = step;
                found = true;
                break;
            }
            hi_invalid = step;
        }
        if (!found) {
            return false;
        }

        double lo = lo_valid;
        double hi = hi_invalid;
        best_pose = pose_lo;
        best_poly = std::move(poly_lo);
        best_bb = bb_lo;
        best_delta = lo;

        for (int it = 0; it < bisect_iters; ++it) {
            const double mid = 0.5 * (lo + hi);
            TreePose pose_mid;
            Polygon poly_mid;
            BoundingBox bb_mid;
            if (valid_at(mid, pose_mid, poly_mid, bb_mid)) {
                lo = mid;
                best_pose = pose_mid;
                best_poly = std::move(poly_mid);
                best_bb = bb_mid;
                best_delta = lo;
            } else {
                hi = mid;
            }
        }
    }

    if (!(best_delta > 1e-12)) {
        return false;
    }

    pose_out = best_pose;
    poly_out = std::move(best_poly);
    bb_out = best_bb;
    delta_out = best_delta;
    return true;
}

}  // namespace

Stats compact_contact(const Polygon& base_poly,
                      std::vector<TreePose>& poses,
                      const Params& params,
                      std::mt19937_64& rng) {
    Stats result;
    if (poses.empty()) {
        result.ok = true;
        return result;
    }

    poses = quantize_poses(poses, params.quantize_decimals);
    std::vector<Polygon> polys = transformed_polygons(base_poly, poses);
    std::vector<BoundingBox> bbs = bounding_boxes_for_poses(base_poly, poses);

    Extents curr_ext = compute_extents(bbs);
    result.side_before = side_from_extents(curr_ext);
    result.area_before = area_from_extents(curr_ext);

    const double radius = enclosing_circle_radius(base_poly);
    const double thr = 2.0 * radius + 1e-9;
    const double thr_sq = thr * thr;
    UniformGridIndex grid(static_cast<int>(poses.size()), thr);
    grid.rebuild(poses);

    auto cost_from_extents = [&](const Extents& e) -> double {
        const double side = side_from_extents(e);
        const double min_dim = min_dim_from_extents(e);
        return side + params.plateau_eps * min_dim;
    };

    double curr_cost = cost_from_extents(curr_ext);
    int no_improve_passes = 0;

    Eval best_eval{result.side_before, result.area_before};
    int last_improve_pass = 0;
    const bool use_early_stop =
        params.early_stop.enabled && params.early_stop.patience_passes > 0;

    std::uniform_int_distribution<int> coin(0, 1);
    std::uniform_real_distribution<double> uni01(0.0, 1.0);
    std::normal_distribution<double> normal(0.0, 1.0);

    std::vector<int> neigh;
    neigh.reserve(64);

    auto try_shake = [&](int idx) {
        if ((params.shake_pos <= 0.0 && params.shake_rot_deg <= 0.0) ||
            uni01(rng) > params.shake_prob) {
            return;
        }

        TreePose cand = poses[static_cast<size_t>(idx)];
        cand.x += normal(rng) * params.shake_pos;
        cand.y += normal(rng) * params.shake_pos;
        if (params.shake_rot_deg > 0.0) {
            cand.deg += normal(rng) * params.shake_rot_deg;
        }
        cand = quantize_pose_wrap_deg(cand, params.quantize_decimals);
        if (cand.x < -100.0 || cand.x > 100.0 || cand.y < -100.0 || cand.y > 100.0) {
            return;
        }

        Polygon poly = transform_polygon(base_poly, cand);
        BoundingBox bb = bounding_box(poly);
        bool collide = false;
        grid.gather(cand.x, cand.y, neigh);
        for (int j : neigh) {
            if (j == idx) {
                continue;
            }
            const double dx = cand.x - poses[static_cast<size_t>(j)].x;
            const double dy = cand.y - poses[static_cast<size_t>(j)].y;
            if (dx * dx + dy * dy > thr_sq) {
                continue;
            }
            if (!aabb_overlap(bb, bbs[static_cast<size_t>(j)])) {
                continue;
            }
            if (polygons_intersect(poly, polys[static_cast<size_t>(j)])) {
                collide = true;
                break;
            }
        }
        if (collide) {
            return;
        }

        poses[static_cast<size_t>(idx)] = cand;
        polys[static_cast<size_t>(idx)] = std::move(poly);
        bbs[static_cast<size_t>(idx)] = bb;
        grid.update_position(idx, cand.x, cand.y);
    };

    for (int pass = 0; pass < params.passes; ++pass) {
        const double width = curr_ext.max_x - curr_ext.min_x;
        const double height = curr_ext.max_y - curr_ext.min_y;
        bool axis_x = (width >= height);
        if (params.alt_axis && (pass % 2 == 1)) {
            axis_x = !axis_x;
        }
        const double center_x = 0.5 * (curr_ext.min_x + curr_ext.max_x);
        const double center_y = 0.5 * (curr_ext.min_y + curr_ext.max_y);

        std::vector<int> pool = build_extreme_pool(bbs, params.boundary_topk);
        std::vector<int> candidates;
        candidates.reserve(pool.size());
        const double tol = 1e-9;
        for (int idx : pool) {
            if (axis_x) {
                if (bbs[static_cast<size_t>(idx)].min_x <= curr_ext.min_x + tol ||
                    bbs[static_cast<size_t>(idx)].max_x >= curr_ext.max_x - tol) {
                    candidates.push_back(idx);
                }
            } else {
                if (bbs[static_cast<size_t>(idx)].min_y <= curr_ext.min_y + tol ||
                    bbs[static_cast<size_t>(idx)].max_y >= curr_ext.max_y - tol) {
                    candidates.push_back(idx);
                }
            }
        }
        if (candidates.empty()) {
            candidates = pool;
        }
        if (candidates.empty()) {
            break;
        }

        std::uniform_int_distribution<int> pick(0, static_cast<int>(candidates.size()) - 1);
        std::uniform_int_distribution<int> pick_all(
            0, static_cast<int>(poses.size()) - 1);
        bool moved_any = false;

        const double max_step =
            params.push_max_step_frac * std::max(1e-9, side_from_extents(curr_ext));

        for (int attempt = 0; attempt < params.attempts_per_pass; ++attempt) {
            int i = candidates[static_cast<size_t>(pick(rng))];
            if (params.interior_prob > 0.0 && uni01(rng) < params.interior_prob) {
                i = pick_all(rng);
            }
            try_shake(i);
            const BoundingBox& bb = bbs[static_cast<size_t>(i)];

            Point dir{0.0, 0.0};
            if (axis_x) {
                const bool at_min = bb.min_x <= curr_ext.min_x + tol;
                const bool at_max = bb.max_x >= curr_ext.max_x - tol;
                if (at_min && at_max) {
                    dir.x = (coin(rng) == 0) ? 1.0 : -1.0;
                } else if (at_min) {
                    dir.x = 1.0;
                } else if (at_max) {
                    dir.x = -1.0;
                } else {
                    dir.x = (poses[static_cast<size_t>(i)].x >= center_x) ? -1.0 : 1.0;
                }
                double cross = 0.0;
                if (params.center_bias > 0.0) {
                    cross += params.center_bias *
                             (center_y - poses[static_cast<size_t>(i)].y) /
                             std::max(1e-9, height);
                }
                if (params.diag_frac > 0.0) {
                    cross += (coin(rng) == 0 ? -1.0 : 1.0) * params.diag_frac;
                }
                if (params.diag_rand > 0.0) {
                    cross += (2.0 * uni01(rng) - 1.0) * params.diag_rand;
                }
                dir.y = cross;
            } else {
                const bool at_min = bb.min_y <= curr_ext.min_y + tol;
                const bool at_max = bb.max_y >= curr_ext.max_y - tol;
                if (at_min && at_max) {
                    dir.y = (coin(rng) == 0) ? 1.0 : -1.0;
                } else if (at_min) {
                    dir.y = 1.0;
                } else if (at_max) {
                    dir.y = -1.0;
                } else {
                    dir.y = (poses[static_cast<size_t>(i)].y >= center_y) ? -1.0 : 1.0;
                }
                double cross = 0.0;
                if (params.center_bias > 0.0) {
                    cross += params.center_bias *
                             (center_x - poses[static_cast<size_t>(i)].x) /
                             std::max(1e-9, width);
                }
                if (params.diag_frac > 0.0) {
                    cross += (coin(rng) == 0 ? -1.0 : 1.0) * params.diag_frac;
                }
                if (params.diag_rand > 0.0) {
                    cross += (2.0 * uni01(rng) - 1.0) * params.diag_rand;
                }
                dir.x = cross;
            }

            TreePose cand_pose;
            Polygon cand_poly;
            BoundingBox cand_bb;
            double cand_delta = 0.0;
            if (!push_to_contact_dir(base_poly, poses, polys, bbs, grid, i, dir, max_step,
                                     params.push_bisect_iters, thr_sq, params.quantize_decimals,
                                     cand_pose, cand_poly, cand_bb, cand_delta)) {
                continue;
            }

            const TreePose old_pose = poses[static_cast<size_t>(i)];
            Polygon old_poly = polys[static_cast<size_t>(i)];
            const BoundingBox old_bb = bbs[static_cast<size_t>(i)];

            poses[static_cast<size_t>(i)] = cand_pose;
            polys[static_cast<size_t>(i)] = std::move(cand_poly);
            bbs[static_cast<size_t>(i)] = cand_bb;

            Extents new_ext = compute_extents(bbs);
            const double new_cost = cost_from_extents(new_ext);

            if (new_cost <= curr_cost + 1e-15) {
                grid.update_position(i, cand_pose.x, cand_pose.y);
                curr_ext = new_ext;
                curr_cost = new_cost;
                moved_any = true;
                result.moves++;
            } else {
                poses[static_cast<size_t>(i)] = old_pose;
                polys[static_cast<size_t>(i)] = std::move(old_poly);
                bbs[static_cast<size_t>(i)] = old_bb;
            }
        }

        result.passes_run = pass + 1;
        if (use_early_stop) {
            Eval curr_eval{side_from_extents(curr_ext), area_from_extents(curr_ext)};
            if (improves_eval(curr_eval, best_eval, params.plateau_eps)) {
                best_eval = curr_eval;
                last_improve_pass = result.passes_run;
            }
            const int min_passes = std::max(0, params.early_stop.min_passes);
            const int patience = std::max(1, params.early_stop.patience_passes);
            if (result.passes_run >= min_passes &&
                (result.passes_run - last_improve_pass) >= patience) {
                break;
            }
        } else {
            if (!moved_any) {
                no_improve_passes++;
                if (no_improve_passes >= params.patience) {
                    break;
                }
            } else {
                no_improve_passes = 0;
            }
        }
    }

    if (params.final_rigid) {
        std::vector<TreePose> rotated = poses;
        optimize_rigid_rotation(base_poly, rotated);
        rotated = quantize_poses_wrap_deg(rotated, params.quantize_decimals);
        if (!any_overlap(base_poly, rotated)) {
            poses = std::move(rotated);
        }
    }

    poses = quantize_poses_wrap_deg(poses, params.quantize_decimals);
    std::vector<BoundingBox> final_bbs = bounding_boxes_for_poses(base_poly, poses);
    Extents final_ext = compute_extents(final_bbs);
    result.side_after = side_from_extents(final_ext);
    result.area_after = area_from_extents(final_ext);
    result.ok = true;
    return result;
}

}  // namespace compaction_contact
