#include "boundary_refine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "collision.hpp"
#include "submission_io.hpp"
#include "spatial_grid.hpp"
#include "wrap_utils.hpp"

void refine_boundary(const Polygon& base_poly,
                     double radius,
                     std::vector<TreePose>& poses,
                     int iters,
                     uint64_t seed,
                     double step_hint) {
    if (iters <= 0 || poses.empty()) {
        return;
    }

    for (auto& p : poses) {
        p = quantize_pose_wrap_deg(p);
    }

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::normal_distribution<double> normal(0.0, 1.0);

    const double eps = 1e-9;
    const double limit = 2.0 * radius + eps;
    const double limit_sq = limit * limit;
    auto aabb_overlap = [](const BoundingBox& a, const BoundingBox& b) -> bool {
        if (a.max_x < b.min_x || b.max_x < a.min_x) {
            return false;
        }
        if (a.max_y < b.min_y || b.max_y < a.min_y) {
            return false;
        }
        return true;
    };

    auto polys = transformed_polygons(base_poly, poses);
    std::vector<BoundingBox> bbs;
    bbs.reserve(polys.size());
    for (const auto& poly : polys) {
        bbs.push_back(bounding_box(poly));
    }

    UniformGridIndex grid(static_cast<int>(poses.size()), limit);
    grid.rebuild(poses);
    std::vector<int> neigh;
    neigh.reserve(64);

    auto recompute_side_and_extents =
        [&](double& min_x, double& max_x, double& min_y, double& max_y) -> double {
        min_x = std::numeric_limits<double>::infinity();
        max_x = -std::numeric_limits<double>::infinity();
        min_y = std::numeric_limits<double>::infinity();
        max_y = -std::numeric_limits<double>::infinity();
        for (const auto& bb : bbs) {
            min_x = std::min(min_x, bb.min_x);
            max_x = std::max(max_x, bb.max_x);
            min_y = std::min(min_y, bb.min_y);
            max_y = std::max(max_y, bb.max_y);
        }
        return std::max(max_x - min_x, max_y - min_y);
    };

    double min_x = 0.0, max_x = 0.0, min_y = 0.0, max_y = 0.0;
    double current_s = recompute_side_and_extents(min_x, max_x, min_y, max_y);

    const double initial_step = std::max(step_hint * 0.35, radius * 0.20);
    const double final_step = std::max(step_hint * 0.03, radius * 0.02);
    const double initial_T = std::max(0.01 * current_s, 1e-6);
    const double final_T = initial_T * 1e-3;
    const int compact_iters = iters / 2;

    for (int it = 0; it < iters; ++it) {
        double t = static_cast<double>(it) / std::max(1, iters - 1);
        double step = initial_step * std::pow(final_step / initial_step, t);
        double T = initial_T * std::pow(final_T / initial_T, t);

        double width = max_x - min_x;
        double height = max_y - min_y;
        bool shrink_x = (width >= height);

        std::vector<int> boundary;
        boundary.reserve(poses.size());
        const double tol = 1e-9;

        if (shrink_x) {
            bool left = (uni(rng) < 0.5);
            for (size_t i = 0; i < bbs.size(); ++i) {
                if (left) {
                    if (bbs[i].min_x <= min_x + tol) {
                        boundary.push_back(static_cast<int>(i));
                    }
                } else {
                    if (bbs[i].max_x >= max_x - tol) {
                        boundary.push_back(static_cast<int>(i));
                    }
                }
            }
        } else {
            bool bottom = (uni(rng) < 0.5);
            for (size_t i = 0; i < bbs.size(); ++i) {
                if (bottom) {
                    if (bbs[i].min_y <= min_y + tol) {
                        boundary.push_back(static_cast<int>(i));
                    }
                } else {
                    if (bbs[i].max_y >= max_y - tol) {
                        boundary.push_back(static_cast<int>(i));
                    }
                }
            }
        }

        int idx = 0;
        if (!boundary.empty()) {
            std::uniform_int_distribution<int> pick(0, static_cast<int>(boundary.size()) - 1);
            idx = boundary[static_cast<size_t>(pick(rng))];
        } else {
            std::uniform_int_distribution<int> pick(0, static_cast<int>(poses.size()) - 1);
            idx = pick(rng);
        }

        const bool deterministic = (it < compact_iters);

        const TreePose base = poses[static_cast<size_t>(idx)];
        double dir_x = 0.0;
        double dir_y = 0.0;
        if (shrink_x) {
            if (bbs[static_cast<size_t>(idx)].min_x <= min_x + tol) {
                dir_x = +1.0;
            } else if (bbs[static_cast<size_t>(idx)].max_x >= max_x - tol) {
                dir_x = -1.0;
            } else {
                dir_x = (uni(rng) < 0.5) ? +1.0 : -1.0;
            }
        } else {
            if (bbs[static_cast<size_t>(idx)].min_y <= min_y + tol) {
                dir_y = +1.0;
            } else if (bbs[static_cast<size_t>(idx)].max_y >= max_y - tol) {
                dir_y = -1.0;
            } else {
                dir_y = (uni(rng) < 0.5) ? +1.0 : -1.0;
            }
        }

        double dx0 = dir_x * step;
        double dy0 = dir_y * step;
        double ddeg0 = 0.0;
        if (!deterministic) {
            dx0 += normal(rng) * step * 0.15;
            dy0 += normal(rng) * step * 0.15;
            ddeg0 = normal(rng) * 5.0;
        }

        double scale = 1.0;
        for (int bt = 0; bt < 6; ++bt) {
            TreePose cand = base;
            cand.x += dx0 * scale;
            cand.y += dy0 * scale;
            cand.deg = wrap_deg(cand.deg + ddeg0 * scale);
            cand = quantize_pose_wrap_deg(cand);

            if (cand.x < -100.0 || cand.x > 100.0 || cand.y < -100.0 || cand.y > 100.0) {
                scale *= 0.5;
                continue;
            }

            Polygon cand_poly = transform_polygon(base_poly, cand);
            BoundingBox cand_bb = bounding_box(cand_poly);

            bool collide = false;
            grid.gather(cand.x, cand.y, neigh);
            for (int j : neigh) {
                if (j == idx) {
                    continue;
                }
                double dx = cand.x - poses[static_cast<size_t>(j)].x;
                double dy = cand.y - poses[static_cast<size_t>(j)].y;
                if (dx * dx + dy * dy > limit_sq) {
                    continue;
                }
                if (!aabb_overlap(cand_bb, bbs[static_cast<size_t>(j)])) {
                    continue;
                }
                if (polygons_intersect(cand_poly, polys[static_cast<size_t>(j)])) {
                    collide = true;
                    break;
                }
            }
            if (collide) {
                scale *= 0.5;
                continue;
            }

            BoundingBox old_bb = bbs[static_cast<size_t>(idx)];
            bbs[static_cast<size_t>(idx)] = cand_bb;
            double new_min_x = 0.0, new_max_x = 0.0, new_min_y = 0.0, new_max_y = 0.0;
            double new_s = recompute_side_and_extents(new_min_x, new_max_x, new_min_y, new_max_y);
            bbs[static_cast<size_t>(idx)] = old_bb;

            bool accept = false;
            if (new_s + 1e-12 < current_s) {
                accept = true;
            } else if (!deterministic && T > 0.0) {
                double prob = std::exp((current_s - new_s) / T);
                if (uni(rng) < prob) {
                    accept = true;
                }
            }

            if (!accept) {
                if (deterministic) {
                    scale *= 0.5;
                    continue;
                }
                break;
            }

            poses[static_cast<size_t>(idx)] = cand;
            polys[static_cast<size_t>(idx)] = std::move(cand_poly);
            bbs[static_cast<size_t>(idx)] = cand_bb;
            grid.update_position(idx, cand.x, cand.y);
            current_s = recompute_side_and_extents(min_x, max_x, min_y, max_y);
            break;
        }
    }
}
