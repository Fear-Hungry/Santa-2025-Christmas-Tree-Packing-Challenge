#include "santa2025/lns.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "santa2025/collision_index.hpp"
#include "santa2025/constraints.hpp"
#include "santa2025/logging.hpp"
#include "santa2025/simulated_annealing.hpp"

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

struct BoundsMetrics {
    double min_x = 0.0;
    double min_y = 0.0;
    double max_x = 0.0;
    double max_y = 0.0;
    double width = 0.0;
    double height = 0.0;
    double s = 0.0;
};

BoundsMetrics bounds_metrics(const std::vector<BoundingBox>& world_bbs) {
    BoundsMetrics m;
    if (world_bbs.empty()) {
        return m;
    }
    m.min_x = std::numeric_limits<double>::infinity();
    m.min_y = std::numeric_limits<double>::infinity();
    m.max_x = -std::numeric_limits<double>::infinity();
    m.max_y = -std::numeric_limits<double>::infinity();
    for (const auto& bb : world_bbs) {
        m.min_x = std::min(m.min_x, bb.min_x);
        m.min_y = std::min(m.min_y, bb.min_y);
        m.max_x = std::max(m.max_x, bb.max_x);
        m.max_y = std::max(m.max_y, bb.max_y);
    }
    m.width = m.max_x - m.min_x;
    m.height = m.max_y - m.min_y;
    m.s = std::max(m.width, m.height);
    return m;
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

    verts.erase(std::remove_if(verts.begin(),
                               verts.end(),
                               [&](const Point& v) { return std::hypot(v.x, v.y) <= eps; }),
                verts.end());
    return verts;
}

bool cycle_applies(const LNSOptions& opt, int i) {
    if (opt.cycle_deg.empty()) {
        return false;
    }
    if (opt.cycle_prefix <= 0) {
        return true;
    }
    return i < opt.cycle_prefix;
}

std::vector<double> angle_candidates(const LNSOptions& opt, int i) {
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

std::vector<int> boundary_ids(const std::vector<BoundingBox>& world_bbs, const BoundsMetrics& m, double tol) {
    std::vector<int> ids;
    ids.reserve(world_bbs.size() / 3 + 1);
    for (size_t i = 0; i < world_bbs.size(); ++i) {
        const auto& bb = world_bbs[i];
        if (bb.min_x <= m.min_x + tol || bb.max_x >= m.max_x - tol || bb.min_y <= m.min_y + tol ||
            bb.max_y >= m.max_y - tol) {
            ids.push_back(static_cast<int>(i));
        }
    }
    return ids;
}

bool normalize_to_quadrant(
    const Polygon& tree_poly,
    std::vector<Pose>& poses,
    CollisionIndex* index,
    RotatedBBoxCache& bbox_cache,
    double eps
) {
    if (poses.empty()) {
        return true;
    }
    std::vector<BoundingBox> world;
    world.reserve(poses.size());
    for (const auto& p : poses) {
        const BoundingBox local = rotated_bbox_cached(tree_poly, p.deg, bbox_cache);
        world.push_back(bbox_for_pose(local, p));
    }
    const auto m = bounds_metrics(world);
    const double dx = -m.min_x;
    const double dy = -m.min_y;
    if (std::abs(dx) == 0.0 && std::abs(dy) == 0.0) {
        return true;
    }
    for (const auto& p : poses) {
        if (!within_coord_bounds(p.x + dx, p.y + dy, kCoordMin, kCoordMax, eps)) {
            return false;
        }
    }
    for (auto& p : poses) {
        p.x += dx;
        p.y += dy;
    }
    if (index) {
        index->translate_all(dx, dy);
    }
    return true;
}

bool destroy_and_repair(
    const Polygon& tree_poly,
    std::vector<Pose>& poses,
    CollisionIndex& index,
    RotatedBBoxCache& bbox_cache,
    std::unordered_map<std::int64_t, std::vector<Point>>& verts_cache,
    const LNSOptions& opt,
    double side,
    std::mt19937_64& rng,
    std::uniform_real_distribution<double>& unif01,
    double eps
) {
    const int n = static_cast<int>(poses.size());
    if (n <= 0) {
        return true;
    }

    std::vector<BoundingBox> world;
    world.reserve(static_cast<size_t>(n));
    for (const auto& p : poses) {
        const BoundingBox local = rotated_bbox_cached(tree_poly, p.deg, bbox_cache);
        world.push_back(bbox_for_pose(local, p));
    }
    const auto m = bounds_metrics(world);
    const double tol = std::max(1e-9, 1e-6 * m.s);
    const auto boundary = boundary_ids(world, m, tol);

    int k = static_cast<int>(std::ceil(opt.remove_frac * static_cast<double>(n)));
    k = std::clamp(k, 1, n);

    std::vector<int> removed;
    removed.reserve(static_cast<size_t>(k));
    std::vector<char> picked(static_cast<size_t>(n), 0);

    std::uniform_int_distribution<int> pick_any(0, n - 1);
    std::uniform_int_distribution<int> pick_bnd(0, boundary.empty() ? 0 : (static_cast<int>(boundary.size()) - 1));

    auto take = [&](int id) {
        if (id < 0 || id >= n) {
            return;
        }
        if (picked[static_cast<size_t>(id)]) {
            return;
        }
        picked[static_cast<size_t>(id)] = 1;
        removed.push_back(id);
    };

    int guard = 0;
    switch (opt.destroy_mode) {
    case LNSDestroyMode::kCluster: {
        int seed_id = -1;
        if (!boundary.empty() && unif01(rng) < opt.boundary_prob) {
            seed_id = boundary[static_cast<size_t>(pick_bnd(rng))];
        } else {
            seed_id = pick_any(rng);
        }
        seed_id = std::clamp(seed_id, 0, n - 1);

        std::vector<std::pair<double, int>> dist;
        dist.reserve(static_cast<size_t>(n));
        const Pose& s = poses[static_cast<size_t>(seed_id)];
        for (int i = 0; i < n; ++i) {
            const Pose& p = poses[static_cast<size_t>(i)];
            const double dx = p.x - s.x;
            const double dy = p.y - s.y;
            dist.emplace_back(dx * dx + dy * dy, i);
        }
        std::sort(dist.begin(), dist.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        for (int i = 0; i < std::min(k, n); ++i) {
            take(dist[static_cast<size_t>(i)].second);
        }
        break;
    }
    case LNSDestroyMode::kBoundary: {
        if (!boundary.empty()) {
            std::vector<int> b = boundary;
            std::shuffle(b.begin(), b.end(), rng);
            for (const int id : b) {
                if (static_cast<int>(removed.size()) >= k) {
                    break;
                }
                take(id);
            }
        }
        while (static_cast<int>(removed.size()) < k && guard++ < 10 * n) {
            take(pick_any(rng));
        }
        break;
    }
    case LNSDestroyMode::kRandom: {
        while (static_cast<int>(removed.size()) < k && guard++ < 10 * n) {
            take(pick_any(rng));
        }
        break;
    }
    case LNSDestroyMode::kMixRandomBoundary:
    default: {
        while (static_cast<int>(removed.size()) < k && guard++ < 10 * n) {
            int id = -1;
            if (!boundary.empty() && unif01(rng) < opt.boundary_prob) {
                id = boundary[static_cast<size_t>(pick_bnd(rng))];
            } else {
                id = pick_any(rng);
            }
            take(id);
        }
        break;
    }
    }
    if (removed.empty()) {
        return false;
    }

    for (const int id : removed) {
        index.remove(id);
    }

    std::shuffle(removed.begin(), removed.end(), rng);

    auto offsets_for = [&](double delta_deg) -> const std::vector<Point>& {
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

    for (const int id : removed) {
        Pose best{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), 0.0};

        const auto angs = angle_candidates(opt, id);
        bool any_angle = false;
        for (const double ang : angs) {
            const BoundingBox local_bb = rotated_bbox_cached(tree_poly, ang, bbox_cache);
            if (local_bb.max_x - local_bb.min_x > side + eps || local_bb.max_y - local_bb.min_y > side + eps) {
                continue;
            }

            const double x_low = -local_bb.min_x;
            const double y_low = -local_bb.min_y;

            auto consider = [&](Pose cand_start) {
                cand_start.deg = ang;
                Pose cand = slide_down_left(cand_start, local_bb, index, side, opt.safety_eps, opt.slide_iters, eps);
                if (!std::isfinite(cand.x) || !std::isfinite(cand.y)) {
                    return;
                }
                any_angle = true;
                if (cand.y < best.y - 1e-12 || (std::abs(cand.y - best.y) <= 1e-12 && cand.x < best.x)) {
                    best = cand;
                }
            };

            // Bottom-left corner.
            consider(Pose{x_low, y_low, ang});

            // Touch existing trees.
            for (int j = 0; j < n; ++j) {
                if (picked[static_cast<size_t>(j)]) {
                    continue;
                }
                const Pose& anchor = poses[static_cast<size_t>(j)];
                const BoundingBox anchor_local_bb = rotated_bbox_cached(tree_poly, anchor.deg, bbox_cache);
                const BoundingBox anchor_world_bb = bbox_for_pose(anchor_local_bb, anchor);
                const double delta = ang - anchor.deg;

                // Bounding box corners.
                consider(Pose{anchor_world_bb.max_x - local_bb.min_x + opt.gap, y_low, ang});
                consider(Pose{x_low, anchor_world_bb.max_y - local_bb.min_y + opt.gap, ang});
                consider(Pose{anchor_world_bb.max_x - local_bb.min_x + opt.gap,
                              anchor_world_bb.max_y - local_bb.min_y + opt.gap,
                              ang});

                const auto& offsets = offsets_for(delta);
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
        }

        if (!any_angle || !std::isfinite(best.x)) {
            return false;
        }

        poses[static_cast<size_t>(id)] = best;
        index.set_pose(id, best);
        picked[static_cast<size_t>(id)] = 0;
    }

    return true;
}

}  // namespace

LNSResult lns_shrink_wrap(const Polygon& tree_poly, const std::vector<Pose>& initial, const LNSOptions& opt, double eps) {
    if (opt.n <= 0) {
        return LNSResult{};
    }
    if (static_cast<int>(initial.size()) != opt.n) {
        throw std::invalid_argument("lns_shrink_wrap: initial.size() must equal opt.n");
    }
    if (opt.angles_deg.empty()) {
        throw std::invalid_argument("lns_shrink_wrap: angles_deg must be non-empty");
    }
    if (opt.orientation_mode != OrientationMode::kTryAll && opt.cycle_deg.empty()) {
        throw std::invalid_argument("lns_shrink_wrap: cycle_deg must be set when using cycle modes");
    }
    if (opt.stages < 0 || opt.stage_attempts <= 0) {
        throw std::invalid_argument("lns_shrink_wrap: invalid stages/stage_attempts");
    }
    if (!(opt.shrink_factor > 0.0 && opt.shrink_factor <= 1.0)) {
        throw std::invalid_argument("lns_shrink_wrap: shrink_factor must be in (0,1]");
    }
    if (!(opt.shrink_delta >= 0.0)) {
        throw std::invalid_argument("lns_shrink_wrap: shrink_delta must be >= 0");
    }
    if (!(opt.remove_frac > 0.0 && opt.remove_frac <= 1.0)) {
        throw std::invalid_argument("lns_shrink_wrap: remove_frac must be in (0,1]");
    }
    if (!(opt.boundary_prob >= 0.0 && opt.boundary_prob <= 1.0)) {
        throw std::invalid_argument("lns_shrink_wrap: boundary_prob must be in [0,1]");
    }
    if (opt.slide_iters <= 0) {
        throw std::invalid_argument("lns_shrink_wrap: slide_iters must be > 0");
    }
    if (!(opt.gap >= 0.0)) {
        throw std::invalid_argument("lns_shrink_wrap: gap must be >= 0");
    }
    if (opt.max_offsets_per_delta < 0) {
        throw std::invalid_argument("lns_shrink_wrap: max_offsets_per_delta must be >= 0");
    }

    RotatedBBoxCache bbox_cache;
    std::vector<Pose> cur = initial;

    // Normalize to [0, s] quadrant (keeps objective, helps bottom-left repair).
    CollisionIndex tmp_index(tree_poly, eps);
    tmp_index.resize(opt.n);
    for (int i = 0; i < opt.n; ++i) {
        tmp_index.set_pose(i, cur[static_cast<size_t>(i)]);
    }
    if (!normalize_to_quadrant(tree_poly, cur, &tmp_index, bbox_cache, eps)) {
        throw std::runtime_error("lns_shrink_wrap: failed to normalize initial packing to quadrant within coord bounds");
    }

    const double init_s = packing_s200(tree_poly, cur);

    LNSResult out;
    out.best_poses = cur;
    out.init_s200 = init_s;
    out.best_s200 = init_s;

    const std::string prefix = opt.log_prefix.empty() ? std::string("[lns]") : opt.log_prefix;
    if (opt.log_every > 0) {
        std::lock_guard<std::mutex> lk(log_mutex());
        std::cerr << prefix << " start n=" << opt.n << " init_s200=" << init_s << " stages=" << opt.stages
                  << " stage_attempts=" << opt.stage_attempts << " remove_frac=" << opt.remove_frac
                  << " boundary_prob=" << opt.boundary_prob << " shrink_factor=" << opt.shrink_factor
                  << " shrink_delta=" << opt.shrink_delta << "\n";
    }

    std::mt19937_64 rng(opt.seed);
    std::uniform_real_distribution<double> unif01(0.0, 1.0);

    CollisionIndex index(tree_poly, eps);
    index.resize(opt.n);
    std::unordered_map<std::int64_t, std::vector<Point>> verts_cache;

    double cur_s = init_s;
    for (int stage = 0; stage < opt.stages; ++stage) {
        const double target = cur_s * opt.shrink_factor - opt.shrink_delta;
        if (!(target > 0.0)) {
            break;
        }
        out.last_target = target;

        if (opt.log_every > 0) {
            std::lock_guard<std::mutex> lk(log_mutex());
            std::cerr << prefix << " stage=" << stage << "/" << opt.stages << " target=" << target
                      << " cur_best_s200=" << cur_s << "\n";
        }

        bool stage_ok = false;
        for (int attempt = 0; attempt < opt.stage_attempts; ++attempt) {
            out.attempted++;

            std::vector<Pose> cand = out.best_poses;
            index.resize(opt.n);
            for (int i = 0; i < opt.n; ++i) {
                index.set_pose(i, cand[static_cast<size_t>(i)]);
            }

            if (!normalize_to_quadrant(tree_poly, cand, &index, bbox_cache, eps)) {
                continue;
            }

            if (!destroy_and_repair(tree_poly,
                                    cand,
                                    index,
                                    bbox_cache,
                                    verts_cache,
                                    opt,
                                    target,
                                    rng,
                                    unif01,
                                    eps)) {
                continue;
            }

            if (!normalize_to_quadrant(tree_poly, cand, &index, bbox_cache, eps)) {
                continue;
            }
            const double s_new = packing_s200(tree_poly, cand);
            if (s_new <= target + 1e-12) {
                out.succeeded++;
                out.best_poses = std::move(cand);
                out.best_s200 = s_new;
                cur_s = s_new;
                stage_ok = true;
                if (opt.log_every > 0) {
                    std::lock_guard<std::mutex> lk(log_mutex());
                    std::cerr << prefix << " stage=" << stage << " success attempt=" << attempt << " s200=" << s_new
                              << " target=" << target << "\n";
                }
                break;
            }
        }

        if (!stage_ok) {
            if (opt.log_every > 0) {
                std::lock_guard<std::mutex> lk(log_mutex());
                std::cerr << prefix << " stage=" << stage << " fail attempts=" << opt.stage_attempts
                          << " cur_best_s200=" << cur_s << " target=" << target << "\n";
            }
            break;
        }
        out.stages_done++;
    }

    if (opt.log_every > 0) {
        std::lock_guard<std::mutex> lk(log_mutex());
        std::cerr << prefix << " done stages_done=" << out.stages_done << " best_s200=" << out.best_s200
                  << " attempted=" << out.attempted << " succeeded=" << out.succeeded << "\n";
    }

    return out;
}

}  // namespace santa2025
