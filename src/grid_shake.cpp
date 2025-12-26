#include "santa2025/grid_shake.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "santa2025/collision_index.hpp"
#include "santa2025/constraints.hpp"

namespace santa2025 {
namespace {

struct RotatedBBoxCache {
    std::unordered_map<std::int64_t, BoundingBox> bboxes;
};

constexpr double kQuant = 1e6;

std::int64_t quant_deg(double deg) {
    return static_cast<std::int64_t>(std::llround(deg * kQuant));
}

BoundingBox rotated_bbox_cached(const Polygon& base, double deg, RotatedBBoxCache& cache) {
    const std::int64_t k = quant_deg(deg);
    auto it = cache.bboxes.find(k);
    if (it != cache.bboxes.end()) {
        return it->second;
    }
    const BoundingBox bb = polygon_bbox(rotate_polygon(base, deg));
    cache.bboxes.emplace(k, bb);
    return bb;
}

BoundingBox bbox_for_pose(const BoundingBox& local, const Pose& p) {
    return BoundingBox{
        local.min_x + p.x,
        local.min_y + p.y,
        local.max_x + p.x,
        local.max_y + p.y,
    };
}

bool within_container_quadrant(const BoundingBox& bb, double eps) {
    return bb.min_x >= -eps && bb.min_y >= -eps;
}

bool within_container_square(const BoundingBox& bb, double side, double eps) {
    return within_container_quadrant(bb, eps) && (bb.max_x <= side + eps) && (bb.max_y <= side + eps);
}

bool is_valid_pose(
    int id,
    const Pose& p,
    const BoundingBox& local_bb,
    const CollisionIndex& index,
    double side,
    double safety_eps,
    double eps
) {
    if (!within_coord_bounds(p, kCoordMin, kCoordMax, eps)) {
        return false;
    }
    const BoundingBox bb = bbox_for_pose(local_bb, p);
    if (!within_container_square(bb, side, eps)) {
        return false;
    }
    return !index.collides_with_any(p, id, safety_eps);
}

Pose slide_down_left(
    int id,
    Pose p,
    const BoundingBox& local_bb,
    const CollisionIndex& index,
    double side,
    double safety_eps,
    int iters,
    double eps
) {
    double x_low = -local_bb.min_x;
    double x_high = side - local_bb.max_x;
    double y_low = -local_bb.min_y;
    double y_high = side - local_bb.max_y;

    auto normalize_interval = [&](double& lo, double& hi) -> bool {
        if (lo <= hi) {
            return true;
        }
        if (lo <= hi + eps) {
            const double mid = 0.5 * (lo + hi);
            lo = mid;
            hi = mid;
            return true;
        }
        return false;
    };

    if (!normalize_interval(x_low, x_high) || !normalize_interval(y_low, y_high)) {
        return Pose{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), p.deg};
    }

    // Clamp to container.
    p.x = std::clamp(p.x, x_low, x_high);
    p.y = std::clamp(p.y, y_low, y_high);

    if (!is_valid_pose(id, p, local_bb, index, side, safety_eps, eps)) {
        return Pose{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), p.deg};
    }

    // Drop down (min y).
    {
        double lo = y_low;
        double hi = p.y;
        for (int k = 0; k < iters; ++k) {
            const double mid = 0.5 * (lo + hi);
            Pose tmp = p;
            tmp.y = mid;
            if (is_valid_pose(id, tmp, local_bb, index, side, safety_eps, eps)) {
                hi = mid;
                p = tmp;
            } else {
                lo = mid;
            }
        }
        p.y = hi;
    }

    // Push left (min x).
    {
        double lo = x_low;
        double hi = p.x;
        for (int k = 0; k < iters; ++k) {
            const double mid = 0.5 * (lo + hi);
            Pose tmp = p;
            tmp.x = mid;
            if (is_valid_pose(id, tmp, local_bb, index, side, safety_eps, eps)) {
                hi = mid;
                p = tmp;
            } else {
                lo = mid;
            }
        }
        p.x = hi;
    }

    return p;
}

bool cycle_applies(const GridShakeOptions& opt, int i) {
    if (opt.cycle_deg.empty()) {
        return false;
    }
    if (opt.cycle_prefix <= 0) {
        return true;
    }
    return i < opt.cycle_prefix;
}

std::vector<double> angle_candidates(const GridShakeOptions& opt, int i) {
    if (opt.orientation_mode == OrientationMode::kTryAll) {
        return opt.angles_deg;
    }
    if (!cycle_applies(opt, i)) {
        return opt.angles_deg;
    }
    const double cyc = opt.cycle_deg[static_cast<size_t>(i) % opt.cycle_deg.size()];
    if (opt.orientation_mode == OrientationMode::kCycle) {
        return {cyc};
    }

    std::vector<double> out;
    out.reserve(opt.angles_deg.size() + 1);
    out.push_back(cyc);
    for (const double a : opt.angles_deg) {
        if (std::abs(a - cyc) <= 1e-12) {
            continue;
        }
        out.push_back(a);
    }
    return out;
}

double initial_angle(const GridShakeOptions& opt, int i) {
    if (opt.orientation_mode != OrientationMode::kTryAll && cycle_applies(opt, i)) {
        return opt.cycle_deg[static_cast<size_t>(i) % opt.cycle_deg.size()];
    }
    return opt.angles_deg.front();
}

}  // namespace

std::vector<Pose> grid_shake_pack(const Polygon& tree_poly, const GridShakeOptions& opt, double eps) {
    if (opt.n <= 0) {
        return {};
    }
    if (opt.angles_deg.empty()) {
        throw std::invalid_argument("grid_shake_pack: angles_deg must be non-empty");
    }
    if (opt.orientation_mode != OrientationMode::kTryAll && opt.cycle_deg.empty()) {
        throw std::invalid_argument("grid_shake_pack: cycle_deg must be set when using cycle modes");
    }
    if (opt.slide_iters <= 0) {
        throw std::invalid_argument("grid_shake_pack: slide_iters must be > 0");
    }
    if (opt.passes < 0) {
        throw std::invalid_argument("grid_shake_pack: passes must be >= 0");
    }
    if (!(opt.gap >= 0.0)) {
        throw std::invalid_argument("grid_shake_pack: gap must be >= 0");
    }
    if (!(opt.side_grow > 1.0)) {
        throw std::invalid_argument("grid_shake_pack: side_grow must be > 1");
    }
    if (opt.max_restarts <= 0) {
        throw std::invalid_argument("grid_shake_pack: max_restarts must be > 0");
    }

    RotatedBBoxCache bbox_cache;
    const double radius = polygon_max_radius(tree_poly);
    const double step = (opt.step > 0.0) ? opt.step : (2.0 * radius + std::max(1e-3, opt.gap));

    // Square grid: dim x dim.
    const int dim = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(opt.n))));
    const double required_side = 2.0 * radius + step * static_cast<double>(std::max(0, dim - 1)) + 1e-9;

    double side = opt.side;
    if (!(side > 0.0)) {
        side = required_side;
    }

    std::vector<Pose> poses;
    poses.reserve(static_cast<size_t>(opt.n));

    for (int attempt = 0; attempt < opt.max_restarts; ++attempt) {
        if (side < required_side - eps) {
            side *= opt.side_grow;
            continue;
        }

        poses.clear();
        bool ok = true;

        for (int i = 0; i < opt.n; ++i) {
            const int r = i / dim;
            const int c = i % dim;
            Pose p;
            p.deg = initial_angle(opt, i);
            p.x = radius + static_cast<double>(c) * step;
            p.y = radius + static_cast<double>(r) * step;

            const BoundingBox local_bb = rotated_bbox_cached(tree_poly, p.deg, bbox_cache);
            const BoundingBox bb = bbox_for_pose(local_bb, p);
            if (!within_coord_bounds(p, kCoordMin, kCoordMax, eps) || !within_container_square(bb, side, eps)) {
                ok = false;
                break;
            }
            poses.push_back(p);
        }

        if (!ok) {
            side *= opt.side_grow;
            continue;
        }

        CollisionIndex index(tree_poly, eps);
        index.resize(opt.n);
        for (int i = 0; i < opt.n; ++i) {
            index.set_pose(i, poses[static_cast<size_t>(i)]);
        }

        // Validate the initial layout (user might provide a too-small step).
        for (int i = 0; i < opt.n; ++i) {
            if (index.collides_with_any(poses[static_cast<size_t>(i)], i, opt.safety_eps)) {
                throw std::runtime_error("grid_shake_pack: initial grid overlaps (increase --step or --gap)");
            }
        }

        // Shake-down compaction: iterative slide down/left for each item.
        for (int pass = 0; pass < opt.passes; ++pass) {
            bool improved = false;

            for (int i = 0; i < opt.n; ++i) {
                const Pose cur = poses[static_cast<size_t>(i)];
                Pose best{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), cur.deg};

                for (const double ang : angle_candidates(opt, i)) {
                    const BoundingBox local_bb = rotated_bbox_cached(tree_poly, ang, bbox_cache);
                    Pose cand = cur;
                    cand.deg = ang;
                    cand = slide_down_left(i, cand, local_bb, index, side, opt.safety_eps, opt.slide_iters, eps);
                    if (!std::isfinite(cand.x) || !std::isfinite(cand.y)) {
                        continue;
                    }
                    if (cand.y < best.y - 1e-12 ||
                        (std::abs(cand.y - best.y) <= 1e-12 && cand.x < best.x)) {
                        best = cand;
                    }
                }

                if (!std::isfinite(best.x)) {
                    throw std::runtime_error("grid_shake_pack: failed to compact (no valid pose found)");
                }

                if (std::abs(best.x - cur.x) > 1e-15 || std::abs(best.y - cur.y) > 1e-15 ||
                    std::abs(best.deg - cur.deg) > 1e-12) {
                    improved = true;
                }

                poses[static_cast<size_t>(i)] = best;
                index.set_pose(i, best);
            }

            if (!improved) {
                break;
            }
        }

        return poses;
    }

    throw std::runtime_error("grid_shake_pack: failed to construct a feasible packing within max_restarts");
}

}  // namespace santa2025
