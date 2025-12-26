#include "santa2025/bottom_left.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

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

std::uint64_t pack_xy_key(std::int32_t qx, std::int32_t qy) {
    const std::uint64_t ux = static_cast<std::uint32_t>(qx);
    const std::uint64_t uy = static_cast<std::uint32_t>(qy);
    return (ux << 32) | uy;
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
    return !index.collides_with_any(p, -1, safety_eps);
}

Pose slide_down_left(
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

    if (!is_valid_pose(p, local_bb, index, side, safety_eps, eps)) {
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
            if (is_valid_pose(tmp, local_bb, index, side, safety_eps, eps)) {
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
            if (is_valid_pose(tmp, local_bb, index, side, safety_eps, eps)) {
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

std::vector<Point> nfp_vertices_sorted_limited(const NFP& nfp, int max_n, double eps) {
    std::unordered_set<std::uint64_t> seen;
    std::vector<Point> verts;

    for (const auto& piece : nfp.pieces) {
        for (const auto& v : piece) {
            const auto qx = static_cast<std::int32_t>(std::llround(v.x * kQuant));
            const auto qy = static_cast<std::int32_t>(std::llround(v.y * kQuant));
            const std::uint64_t key = pack_xy_key(qx, qy);
            if (seen.insert(key).second) {
                verts.push_back(v);
            }
        }
    }

    std::sort(verts.begin(), verts.end(), [](const Point& a, const Point& b) {
        if (a.y != b.y) {
            return a.y < b.y;
        }
        return a.x < b.x;
    });

    if (max_n > 0 && static_cast<int>(verts.size()) > max_n) {
        verts.resize(static_cast<size_t>(max_n));
    }

    // Filter out tiny vectors that break unit-direction computation.
    verts.erase(std::remove_if(verts.begin(),
                               verts.end(),
                               [&](const Point& v) { return std::hypot(v.x, v.y) <= eps; }),
                verts.end());
    return verts;
}

}  // namespace

std::vector<Pose> bottom_left_pack(const Polygon& tree_poly, const BottomLeftOptions& opt, double eps) {
    if (opt.n <= 0) {
        return {};
    }
    if (opt.angles_deg.empty()) {
        throw std::invalid_argument("bottom_left_pack: angles_deg must be non-empty");
    }
    if (opt.orientation_mode != OrientationMode::kTryAll && opt.cycle_deg.empty()) {
        throw std::invalid_argument("bottom_left_pack: cycle_deg must be set when using cycle modes");
    }
    if (opt.slide_iters <= 0) {
        throw std::invalid_argument("bottom_left_pack: slide_iters must be > 0");
    }
    if (!(opt.gap >= 0.0)) {
        throw std::invalid_argument("bottom_left_pack: gap must be >= 0");
    }
    if (!(opt.side_grow > 1.0)) {
        throw std::invalid_argument("bottom_left_pack: side_grow must be > 1");
    }
    if (opt.max_restarts <= 0) {
        throw std::invalid_argument("bottom_left_pack: max_restarts must be > 0");
    }

    RotatedBBoxCache bbox_cache;
    CollisionIndex index(tree_poly, eps);
    index.resize(opt.n);

    // Cache NFP vertex candidates per relative rotation (keyed by quantized degrees).
    std::unordered_map<std::int64_t, std::vector<Point>> verts_cache;

    auto get_offsets = [&](double delta_deg) -> const std::vector<Point>& {
        const std::int64_t key = quant_deg(delta_deg);
        auto it = verts_cache.find(key);
        if (it != verts_cache.end()) {
            return it->second;
        }
        const auto& nfp = index.nfp(delta_deg);
        auto verts = nfp_vertices_sorted_limited(nfp, opt.max_offsets_per_delta, eps);
        auto [ins, _ok] = verts_cache.emplace(key, std::move(verts));
        return ins->second;
    };

    std::vector<Pose> poses;
    poses.reserve(static_cast<size_t>(opt.n));

    // Initial side guess.
    double side = opt.side;
    if (!(side > 0.0)) {
        const double area = polygon_area(tree_poly);
        const double dens = std::max(1e-6, opt.density_guess);
        side = std::sqrt(static_cast<double>(opt.n) * area / dens);
    }
    {
        double need = std::numeric_limits<double>::infinity();
        auto consider_need = [&](double ang) {
            const BoundingBox bb = rotated_bbox_cached(tree_poly, ang, bbox_cache);
            const double s = std::max(bb.max_x - bb.min_x, bb.max_y - bb.min_y);
            need = std::min(need, s);
        };

        if (opt.orientation_mode == OrientationMode::kCycle) {
            consider_need(opt.cycle_deg.front());
        } else {
            for (const double ang : opt.angles_deg) {
                consider_need(ang);
            }
            if (opt.orientation_mode == OrientationMode::kCycleThenAll && !opt.cycle_deg.empty()) {
                consider_need(opt.cycle_deg.front());
            }
        }

        side = std::max(side, need + 1e-9);
    }

    auto cycle_applies = [&](int i) -> bool {
        if (opt.cycle_deg.empty()) {
            return false;
        }
        if (opt.cycle_prefix <= 0) {
            return true;
        }
        return i < opt.cycle_prefix;
    };

    for (int attempt = 0; attempt < opt.max_restarts; ++attempt) {
        poses.clear();
        index.resize(opt.n);

        bool ok = true;
        for (int i = 0; i < opt.n; ++i) {
            Pose best{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), 0.0};

            auto process_angle = [&](double ang) -> bool {
                bool found_any = false;
                const BoundingBox local_bb = rotated_bbox_cached(tree_poly, ang, bbox_cache);
                if (local_bb.max_x - local_bb.min_x > side + eps || local_bb.max_y - local_bb.min_y > side + eps) {
                    return false;
                }

                const double x_low = -local_bb.min_x;
                const double y_low = -local_bb.min_y;

                auto consider = [&](Pose cand_start) {
                    cand_start.deg = ang;
                    Pose cand = slide_down_left(cand_start, local_bb, index, side, opt.safety_eps, opt.slide_iters, eps);
                    if (!std::isfinite(cand.x) || !std::isfinite(cand.y)) {
                        return;
                    }
                    found_any = true;
                    if (cand.y < best.y - 1e-12 ||
                        (std::abs(cand.y - best.y) <= 1e-12 && cand.x < best.x)) {
                        best = cand;
                    }
                };

                // Always try the bottom-left corner of the container.
                consider(Pose{x_low, y_low, ang});

                for (int j = 0; j < i; ++j) {
                    const Pose& anchor = poses[static_cast<size_t>(j)];
                    const BoundingBox anchor_local_bb = rotated_bbox_cached(tree_poly, anchor.deg, bbox_cache);
                    const BoundingBox anchor_world_bb = bbox_for_pose(anchor_local_bb, anchor);
                    const double delta = ang - anchor.deg;

                    // Bounding-box corner candidates (classic bottom-left for rectangles).
                    consider(Pose{anchor_world_bb.max_x - local_bb.min_x + opt.gap, y_low, ang});
                    consider(Pose{x_low, anchor_world_bb.max_y - local_bb.min_y + opt.gap, ang});
                    consider(Pose{anchor_world_bb.max_x - local_bb.min_x + opt.gap,
                                  anchor_world_bb.max_y - local_bb.min_y + opt.gap,
                                  ang});

                    const auto& offsets = get_offsets(delta);
                    for (const auto& v : offsets) {
                        const double nrm = std::hypot(v.x, v.y);
                        if (!(nrm > 0.0)) {
                            continue;
                        }
                        const Point u{v.x / nrm, v.y / nrm};
                        const Point v_out{v.x + opt.gap * u.x, v.y + opt.gap * u.y};

                        const Point t_world = rotate_point(v_out, anchor.deg);
                        consider(Pose{anchor.x + t_world.x, anchor.y + t_world.y, ang});
                    }
                }
                return found_any;
            };

            auto try_all_angles = [&]() -> bool {
                bool any = false;
                for (const double ang : opt.angles_deg) {
                    any = process_angle(ang) || any;
                }
                return any;
            };

            bool found = false;
            if (opt.orientation_mode != OrientationMode::kTryAll && cycle_applies(i)) {
                const double ang = opt.cycle_deg[static_cast<size_t>(i) % opt.cycle_deg.size()];
                found = process_angle(ang);
                if (!found && opt.orientation_mode == OrientationMode::kCycleThenAll) {
                    found = try_all_angles();
                }
            } else {
                found = try_all_angles();
            }

            if (!found || !std::isfinite(best.x)) {
                ok = false;
                break;
            }

            poses.push_back(best);
            index.set_pose(i, best);
        }

        if (ok) {
            return poses;
        }

        side *= opt.side_grow;
    }

    throw std::runtime_error("bottom_left_pack: failed to construct a feasible packing within max_restarts");
}

}  // namespace santa2025
