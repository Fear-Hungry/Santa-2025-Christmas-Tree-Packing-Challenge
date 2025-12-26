#include "santa2025/simulated_annealing.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "santa2025/collision_index.hpp"
#include "santa2025/constraints.hpp"
#include "santa2025/logging.hpp"

namespace santa2025 {
namespace {

struct RotatedBBoxCache {
    std::unordered_map<std::int64_t, BoundingBox> bboxes;
};

struct BoundsMetrics {
    double min_x = 0.0;
    double min_y = 0.0;
    double max_x = 0.0;
    double max_y = 0.0;
    double width = 0.0;
    double height = 0.0;
    double s = 0.0;
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

double prefix_score_from_bbs(const std::vector<BoundingBox>& world_bbs, int nmax) {
    if (nmax <= 0) {
        return 0.0;
    }
    if (world_bbs.empty()) {
        return 0.0;
    }
    const int n = std::min(nmax, static_cast<int>(world_bbs.size()));

    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();

    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        const auto& bb = world_bbs[static_cast<size_t>(i)];
        min_x = std::min(min_x, bb.min_x);
        min_y = std::min(min_y, bb.min_y);
        max_x = std::max(max_x, bb.max_x);
        max_y = std::max(max_y, bb.max_y);
        const double s = std::max(max_x - min_x, max_y - min_y);
        total += (s * s) / static_cast<double>(i + 1);
    }
    return total;
}

double secondary_value(const SAOptions& opt, const BoundsMetrics& m) {
    switch (opt.secondary) {
    case SASecondary::kNone:
        return 0.0;
    case SASecondary::kPerimeter:
        return m.width + m.height;
    case SASecondary::kArea:
        return m.width * m.height;
    case SASecondary::kAspect:
        return std::abs(m.width - m.height);
    }
    return 0.0;
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

bool cycle_applies(const SAOptions& opt, int i) {
    if (opt.cycle_deg.empty()) {
        return false;
    }
    if (opt.cycle_prefix <= 0) {
        return true;
    }
    return i < opt.cycle_prefix;
}

double pick_angle(
    const SAOptions& opt,
    int i,
    double current_deg,
    std::mt19937_64& rng,
    std::uniform_real_distribution<double>& unif01
) {
    if (opt.orientation_mode == OrientationMode::kCycle && cycle_applies(opt, i)) {
        return opt.cycle_deg[static_cast<size_t>(i) % opt.cycle_deg.size()];
    }

    const bool bias_cycle =
        (opt.orientation_mode == OrientationMode::kCycleThenAll && cycle_applies(opt, i) && !opt.cycle_deg.empty());
    if (bias_cycle && unif01(rng) < 0.5) {
        return opt.cycle_deg[static_cast<size_t>(i) % opt.cycle_deg.size()];
    }

    if (opt.angles_deg.empty()) {
        return current_deg;
    }
    std::uniform_int_distribution<int> pick(0, static_cast<int>(opt.angles_deg.size()) - 1);
    return opt.angles_deg[static_cast<size_t>(pick(rng))];
}

std::vector<Point> nfp_vertices_unique_limited(const NFP& nfp, int max_n, double eps) {
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
    verts.erase(std::remove_if(verts.begin(),
                               verts.end(),
                               [&](const Point& v) { return std::hypot(v.x, v.y) <= eps; }),
                verts.end());
    if (max_n > 0 && static_cast<int>(verts.size()) > max_n) {
        verts.resize(static_cast<size_t>(max_n));
    }
    return verts;
}

BoundsMetrics bounds_metrics_replace_one(const std::vector<BoundingBox>& world_bbs, int id, const BoundingBox& bb_new) {
    BoundsMetrics m;
    if (world_bbs.empty()) {
        return m;
    }
    m.min_x = std::numeric_limits<double>::infinity();
    m.min_y = std::numeric_limits<double>::infinity();
    m.max_x = -std::numeric_limits<double>::infinity();
    m.max_y = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < world_bbs.size(); ++i) {
        const auto& bb = (static_cast<int>(i) == id) ? bb_new : world_bbs[i];
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

double prefix_score_replace_one(
    const std::vector<BoundingBox>& world_bbs,
    int id,
    const BoundingBox& bb_new,
    int nmax
) {
    if (nmax <= 0) {
        return 0.0;
    }
    if (world_bbs.empty()) {
        return 0.0;
    }
    const int n = std::min(nmax, static_cast<int>(world_bbs.size()));

    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();

    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        const auto& bb = (i == id) ? bb_new : world_bbs[static_cast<size_t>(i)];
        min_x = std::min(min_x, bb.min_x);
        min_y = std::min(min_y, bb.min_y);
        max_x = std::max(max_x, bb.max_x);
        max_y = std::max(max_y, bb.max_y);
        const double s = std::max(max_x - min_x, max_y - min_y);
        total += (s * s) / static_cast<double>(i + 1);
    }
    return total;
}

}  // namespace

double packing_s200(const Polygon& tree_poly, const std::vector<Pose>& poses) {
    RotatedBBoxCache cache;
    std::vector<BoundingBox> world;
    world.reserve(poses.size());
    for (const auto& p : poses) {
        const BoundingBox local = rotated_bbox_cached(tree_poly, p.deg, cache);
        world.push_back(bbox_for_pose(local, p));
    }
    return bounds_metrics(world).s;
}

double packing_prefix_score(const Polygon& tree_poly, const std::vector<Pose>& poses, int nmax) {
    RotatedBBoxCache cache;
    std::vector<BoundingBox> world;
    world.reserve(poses.size());
    for (const auto& p : poses) {
        const BoundingBox local = rotated_bbox_cached(tree_poly, p.deg, cache);
        world.push_back(bbox_for_pose(local, p));
    }
    return prefix_score_from_bbs(world, nmax);
}

SAResult simulated_annealing(const Polygon& tree_poly, const std::vector<Pose>& initial, const SAOptions& opt, double eps) {
    if (opt.n <= 0) {
        return SAResult{};
    }
    if (static_cast<int>(initial.size()) != opt.n) {
        throw std::invalid_argument("simulated_annealing: initial.size() must equal opt.n");
    }
    if (opt.angles_deg.empty()) {
        throw std::invalid_argument("simulated_annealing: angles_deg must be non-empty");
    }
    if (opt.orientation_mode != OrientationMode::kTryAll && opt.cycle_deg.empty()) {
        throw std::invalid_argument("simulated_annealing: cycle_deg must be set when using cycle modes");
    }
    if (opt.iters <= 0) {
        throw std::invalid_argument("simulated_annealing: iters must be > 0");
    }
    if (opt.tries_per_iter <= 0) {
        throw std::invalid_argument("simulated_annealing: tries_per_iter must be > 0");
    }
    if (!(opt.t0 > 0.0) || !(opt.t1 > 0.0)) {
        throw std::invalid_argument("simulated_annealing: t0/t1 must be > 0");
    }
    if (opt.schedule == SASchedule::kPolynomial && !(opt.poly_power > 0.0)) {
        throw std::invalid_argument("simulated_annealing: poly_power must be > 0");
    }
    if (opt.adaptive_window < 0) {
        throw std::invalid_argument("simulated_annealing: adaptive_window must be >= 0");
    }
    if (opt.adaptive_window > 0 && !(opt.target_accept >= 0.0 && opt.target_accept <= 1.0)) {
        throw std::invalid_argument("simulated_annealing: target_accept must be in [0,1]");
    }
    if (opt.max_offsets_per_delta < 0) {
        throw std::invalid_argument("simulated_annealing: max_offsets_per_delta must be >= 0");
    }
    if (!(opt.delta_scale > 0.0)) {
        throw std::invalid_argument("simulated_annealing: delta_scale must be > 0");
    }
    if (opt.secondary_weight < 0.0) {
        throw std::invalid_argument("simulated_annealing: secondary_weight must be >= 0");
    }
    if (!(opt.boundary_prob >= 0.0 && opt.boundary_prob <= 1.0)) {
        throw std::invalid_argument("simulated_annealing: boundary_prob must be in [0,1]");
    }
    if (!(opt.cluster_prob >= 0.0 && opt.cluster_prob <= 1.0)) {
        throw std::invalid_argument("simulated_annealing: cluster_prob must be in [0,1]");
    }
    if (opt.cluster_min < 1) {
        throw std::invalid_argument("simulated_annealing: cluster_min must be >= 1");
    }
    if (opt.cluster_max < opt.cluster_min) {
        throw std::invalid_argument("simulated_annealing: cluster_max must be >= cluster_min");
    }
    if (!(opt.cluster_radius_mult > 0.0)) {
        throw std::invalid_argument("simulated_annealing: cluster_radius_mult must be > 0");
    }
    if (!(opt.cluster_sigma_mult > 0.0)) {
        throw std::invalid_argument("simulated_annealing: cluster_sigma_mult must be > 0");
    }
    if (opt.touch_best_of <= 0) {
        throw std::invalid_argument("simulated_annealing: touch_best_of must be >= 1");
    }
    if (opt.objective == SAObjective::kTargetSide) {
        if (!(opt.target_side > 0.0)) {
            throw std::invalid_argument("simulated_annealing: target_side must be > 0 when objective==kTargetSide");
        }
        if (opt.target_power != 1 && opt.target_power != 2) {
            throw std::invalid_argument("simulated_annealing: target_power must be 1 or 2");
        }
    }

    std::vector<Pose> poses = initial;

    RotatedBBoxCache bbox_cache;
    std::vector<BoundingBox> local_bbs(static_cast<size_t>(opt.n));
    std::vector<BoundingBox> world_bbs(static_cast<size_t>(opt.n));
    for (int i = 0; i < opt.n; ++i) {
        const Pose& p = poses[static_cast<size_t>(i)];
        local_bbs[static_cast<size_t>(i)] = rotated_bbox_cached(tree_poly, p.deg, bbox_cache);
        world_bbs[static_cast<size_t>(i)] = bbox_for_pose(local_bbs[static_cast<size_t>(i)], p);
    }

    CollisionIndex index(tree_poly, eps);
    index.resize(opt.n);
    for (int i = 0; i < opt.n; ++i) {
        index.set_pose(i, poses[static_cast<size_t>(i)]);
    }

    // NFP vertex caches keyed by quantized delta degrees.
    std::unordered_map<std::int64_t, std::vector<Point>> nfp_verts_cache;
    auto get_offsets = [&](double delta_deg) -> const std::vector<Point>& {
        const std::int64_t key = quant_deg(delta_deg);
        auto it = nfp_verts_cache.find(key);
        if (it != nfp_verts_cache.end()) {
            return it->second;
        }
        const auto& nfp = index.nfp(delta_deg);
        auto verts = nfp_vertices_unique_limited(nfp, opt.max_offsets_per_delta, eps);
        auto [ins, _ok] = nfp_verts_cache.emplace(key, std::move(verts));
        return ins->second;
    };

    auto clamp_pose = [&](Pose p) {
        p = clamp_pose_xy(p, kCoordMin, kCoordMax);
        if (opt.rot_jitter_deg != 0.0) {
            // Caller is responsible for jitter in a deterministic/reproducible way if they want it.
        }
        return p;
    };

    SAResult res;
    res.best_poses = poses;
    const BoundsMetrics init_bounds = bounds_metrics(world_bbs);
    const double init_score = prefix_score_from_bbs(world_bbs, 200);
    res.init_s200 = init_bounds.s;
    res.best_s200 = res.init_s200;
    res.init_prefix_score = init_score;
    res.best_prefix_score = res.init_prefix_score;

    BoundsMetrics cur_bounds = init_bounds;
    double cur_secondary = secondary_value(opt, cur_bounds);
    double current_cost = 0.0;
    if (opt.objective == SAObjective::kPrefixScore) {
        current_cost = prefix_score_from_bbs(world_bbs, opt.nmax_score);
    } else if (opt.objective == SAObjective::kTargetSide) {
        const double v = std::max(0.0, cur_bounds.s - opt.target_side);
        current_cost = (opt.target_power == 1) ? v : (v * v);
    } else {
        current_cost = cur_bounds.s;
    }
    res.init_cost = current_cost;
    res.best_cost = current_cost;
    double best_s = cur_bounds.s;
    double best_secondary = cur_secondary;

    const double log_ratio = std::log(opt.t1 / opt.t0);
    const double poly_c =
        (opt.schedule == SASchedule::kPolynomial)
            ? (std::pow(opt.t0 / opt.t1, 1.0 / opt.poly_power) - 1.0)
            : 0.0;

    std::mt19937_64 rng(opt.seed);
    std::uniform_real_distribution<double> unif01(0.0, 1.0);
    std::normal_distribution<double> norm01(0.0, 1.0);
    std::uniform_int_distribution<int> pick_id(0, opt.n - 1);

    int feasible_this_log = 0;
    int accepted_this_log = 0;
    int attempted_this_log = 0;
    int feasible_window = 0;
    int accepted_window = 0;
    double adapt_scale = 1.0;

    for (int it = 0; it < opt.iters; ++it) {
        const double frac = (opt.iters == 1) ? 1.0 : (static_cast<double>(it) / static_cast<double>(opt.iters - 1));
        if (opt.adaptive_window > 0 && it > 0 && (it % opt.adaptive_window) == 0) {
            const double acc_rate = static_cast<double>(accepted_window) / static_cast<double>(std::max(1, feasible_window));
            if (acc_rate < opt.target_accept) {
                adapt_scale *= opt.adapt_up;
            } else if (acc_rate > opt.target_accept) {
                adapt_scale *= opt.adapt_down;
            }
            adapt_scale = std::clamp(adapt_scale, opt.adapt_min_scale, opt.adapt_max_scale);
            feasible_window = 0;
            accepted_window = 0;
        }

        double T_base = opt.t0 * std::exp(log_ratio * frac);
        if (opt.schedule == SASchedule::kPolynomial) {
            T_base = opt.t0 / std::pow(1.0 + poly_c * frac, opt.poly_power);
        }
        const double T = T_base * adapt_scale;
        const double sigma = opt.trans_sigma0 + (opt.trans_sigma1 - opt.trans_sigma0) * frac;

        const std::vector<int> b_ids =
            (opt.boundary_prob > 0.0) ? boundary_ids(world_bbs, cur_bounds, 1e-9) : std::vector<int>{};

        auto pick_focus_id = [&]() -> int {
            if (!b_ids.empty() && unif01(rng) < opt.boundary_prob) {
                std::uniform_int_distribution<int> pickb(0, static_cast<int>(b_ids.size()) - 1);
                return b_ids[static_cast<size_t>(pickb(rng))];
            }
            return pick_id(rng);
        };

        for (int t = 0; t < opt.tries_per_iter; ++t) {
            res.attempted++;
            attempted_this_log++;

            const bool do_cluster = (opt.n >= opt.cluster_min && opt.cluster_min >= 2) && (unif01(rng) < opt.cluster_prob);
            if (do_cluster) {
                struct Saved {
                    int id = -1;
                    Pose pose;
                    BoundingBox world;
                };

                const int seed_id = pick_focus_id();
                const double r = opt.cluster_radius_mult * index.tree_radius();
                const double r2 = r * r;

                std::vector<int> cluster;
                cluster.reserve(static_cast<size_t>(opt.cluster_max));
                cluster.push_back(seed_id);
                const Pose& seed_pose = poses[static_cast<size_t>(seed_id)];
                for (int j = 0; j < opt.n; ++j) {
                    if (j == seed_id) {
                        continue;
                    }
                    const Pose& pj = poses[static_cast<size_t>(j)];
                    const double dx = pj.x - seed_pose.x;
                    const double dy = pj.y - seed_pose.y;
                    if (dx * dx + dy * dy <= r2) {
                        cluster.push_back(j);
                    }
                }

                if (static_cast<int>(cluster.size()) < opt.cluster_min) {
                    continue;
                }

                if (static_cast<int>(cluster.size()) > opt.cluster_max) {
                    std::shuffle(cluster.begin() + 1, cluster.end(), rng);
                    cluster.resize(static_cast<size_t>(opt.cluster_max));
                }

                std::vector<Saved> saved;
                saved.reserve(cluster.size());
                for (const int id : cluster) {
                    saved.push_back(Saved{id, poses[static_cast<size_t>(id)], world_bbs[static_cast<size_t>(id)]});
                    index.remove(id);
                }

                const double s_cluster = sigma * opt.cluster_sigma_mult;
                const double dx = norm01(rng) * s_cluster;
                const double dy = norm01(rng) * s_cluster;

                bool feasible = true;
                for (const auto& sv : saved) {
                    Pose p = sv.pose;
                    p.x += dx;
                    p.y += dy;
                    if (!within_coord_bounds(p, kCoordMin, kCoordMax, eps)) {
                        feasible = false;
                        break;
                    }
                    if (index.collides_with_any(p, -1, opt.safety_eps)) {
                        feasible = false;
                        break;
                    }
                }

                if (!feasible) {
                    for (const auto& sv : saved) {
                        index.set_pose(sv.id, sv.pose);
                    }
                    continue;
                }

                res.feasible++;
                feasible_this_log++;
                feasible_window++;

                for (const auto& sv : saved) {
                    Pose p = sv.pose;
                    p.x += dx;
                    p.y += dy;
                    poses[static_cast<size_t>(sv.id)] = p;
                    world_bbs[static_cast<size_t>(sv.id)] = BoundingBox{
                        sv.world.min_x + dx,
                        sv.world.min_y + dy,
                        sv.world.max_x + dx,
                        sv.world.max_y + dy,
                    };
                    index.set_pose(sv.id, p);
                }

                const BoundsMetrics new_bounds = bounds_metrics(world_bbs);
                const double new_secondary = secondary_value(opt, new_bounds);

                double new_cost = 0.0;
                double d = 0.0;
                if (opt.objective == SAObjective::kPrefixScore) {
                    new_cost = prefix_score_from_bbs(world_bbs, opt.nmax_score);
                    d = (new_cost - current_cost) * opt.delta_scale;
                } else if (opt.objective == SAObjective::kTargetSide) {
                    const double v_old = std::max(0.0, cur_bounds.s - opt.target_side);
                    const double v_new = std::max(0.0, new_bounds.s - opt.target_side);
                    new_cost = (opt.target_power == 1) ? v_new : (v_new * v_new);
                    d = (new_cost - current_cost) * opt.delta_scale;
                    if (opt.secondary != SASecondary::kNone && opt.secondary_weight > 0.0 &&
                        std::abs(v_new - v_old) <= 1e-12) {
                        d += opt.secondary_weight * (new_secondary - cur_secondary);
                    }
                } else {
                    const double s_old = cur_bounds.s;
                    const double s_new = new_bounds.s;
                    new_cost = s_new;

                    const double s_old2 = s_old * s_old;
                    const double s_new2 = s_new * s_new;
                    double delta = 0.0;
                    switch (opt.delta_mode) {
                    case SADeltaMode::kLinear:
                        delta = s_new - s_old;
                        break;
                    case SADeltaMode::kSquared:
                        delta = s_new2 - s_old2;
                        break;
                    case SADeltaMode::kSquaredOverS:
                        delta = (s_new2 - s_old2) / std::max(1e-12, s_old);
                        break;
                    case SADeltaMode::kSquaredOverS2:
                        delta = (s_new2 - s_old2) / std::max(1e-12, s_old2);
                        break;
                    }

                    d = delta * opt.delta_scale;
                    if (opt.secondary != SASecondary::kNone && opt.secondary_weight > 0.0 &&
                        std::abs(s_new - s_old) <= 1e-12) {
                        d += opt.secondary_weight * (new_secondary - cur_secondary);
                    }
                }

                bool accept = false;
                if (d <= 0.0) {
                    accept = true;
                } else {
                    const double p = std::exp(-d / T);
                    accept = unif01(rng) < p;
                }

                if (accept) {
                    res.accepted++;
                    accepted_this_log++;
                    accepted_window++;
                    current_cost = new_cost;
                    cur_bounds = new_bounds;
                    cur_secondary = new_secondary;

                    const double s200 = new_bounds.s;
                    const double score = prefix_score_from_bbs(world_bbs, 200);

                    bool improved_best = false;
                    if (opt.objective == SAObjective::kPrefixScore) {
                        improved_best = (new_cost < res.best_cost - 1e-15);
                    } else if (opt.objective == SAObjective::kTargetSide) {
                        if (new_cost < res.best_cost - 1e-15) {
                            improved_best = true;
                        } else if (std::abs(new_cost - res.best_cost) <= 1e-15) {
                            if (s200 < best_s - 1e-12) {
                                improved_best = true;
                            } else if (std::abs(s200 - best_s) <= 1e-12 &&
                                       opt.secondary != SASecondary::kNone && opt.secondary_weight > 0.0 &&
                                       new_secondary < best_secondary - 1e-12) {
                                improved_best = true;
                            }
                        }
                    } else {
                        if (s200 < best_s - 1e-12) {
                            improved_best = true;
                        } else if (std::abs(s200 - best_s) <= 1e-12 &&
                                   opt.secondary != SASecondary::kNone && opt.secondary_weight > 0.0 &&
                                   new_secondary < best_secondary - 1e-12) {
                            improved_best = true;
                        }
                    }

                    if (improved_best) {
                        if (opt.objective == SAObjective::kPrefixScore || opt.objective == SAObjective::kTargetSide) {
                            res.best_cost = new_cost;
                        } else {
                            res.best_cost = s200;
                        }
                        res.best_poses = poses;
                        res.best_s200 = s200;
                        res.best_prefix_score = score;
                        best_s = s200;
                        best_secondary = new_secondary;
                    }

                    break;
                }

                // Reject cluster move: restore.
                for (const auto& sv : saved) {
                    poses[static_cast<size_t>(sv.id)] = sv.pose;
                    world_bbs[static_cast<size_t>(sv.id)] = sv.world;
                    index.set_pose(sv.id, sv.pose);
                }
                continue;
            }

            const int i = pick_focus_id();
            const Pose old = poses[static_cast<size_t>(i)];
            const BoundingBox old_local = local_bbs[static_cast<size_t>(i)];
            const BoundingBox old_world = world_bbs[static_cast<size_t>(i)];

            Pose cand = old;
            BoundingBox cand_local;
            BoundingBox cand_world;
            bool cand_has_bbs = false;

            const bool do_touch = unif01(rng) < opt.touch_prob && opt.n > 1;
            if (do_touch) {
                int j = pick_id(rng);
                if (j == i) {
                    j = (j + 1) % opt.n;
                }
                const Pose& anchor = poses[static_cast<size_t>(j)];
                const double ang = pick_angle(opt, i, old.deg, rng, unif01);

                const double delta = ang - anchor.deg;
                const auto& offsets = get_offsets(delta);
                if (offsets.empty()) {
                    continue;
                }

                const int k = std::max(1, opt.touch_best_of);
                std::uniform_int_distribution<int> pick_off(0, static_cast<int>(offsets.size()) - 1);

                Pose best_pose;
                BoundingBox best_local;
                BoundingBox best_world;
                BoundsMetrics best_bounds;
                double best_secondary_val = 0.0;
                double best_primary = 0.0;
                bool found_any = false;

                for (int s = 0; s < k; ++s) {
                    const Point v = offsets[static_cast<size_t>(pick_off(rng))];
                    const double nrm = std::hypot(v.x, v.y);
                    if (!(nrm > 0.0)) {
                        continue;
                    }
                    const Point u{v.x / nrm, v.y / nrm};
                    const Point v_out{v.x + opt.gap * u.x, v.y + opt.gap * u.y};
                    const Point t_world = rotate_point(v_out, anchor.deg);

                    Pose tmp;
                    tmp.deg = ang;
                    tmp.x = anchor.x + t_world.x;
                    tmp.y = anchor.y + t_world.y;

                    // Small jitter around the contact point.
                    const double j_sigma = 0.25 * sigma;
                    tmp.x += norm01(rng) * j_sigma;
                    tmp.y += norm01(rng) * j_sigma;

                    tmp = clamp_pose(tmp);
                    if (!within_coord_bounds(tmp, kCoordMin, kCoordMax, eps)) {
                        continue;
                    }
                    if (index.collides_with_any(tmp, i, opt.safety_eps)) {
                        continue;
                    }

                    const BoundingBox tmp_local = rotated_bbox_cached(tree_poly, tmp.deg, bbox_cache);
                    const BoundingBox tmp_world = bbox_for_pose(tmp_local, tmp);
                    const BoundsMetrics tmp_bounds = bounds_metrics_replace_one(world_bbs, i, tmp_world);
                    const double tmp_secondary = secondary_value(opt, tmp_bounds);
                    double tmp_primary = 0.0;
                    if (opt.objective == SAObjective::kPrefixScore) {
                        tmp_primary = prefix_score_replace_one(world_bbs, i, tmp_world, opt.nmax_score);
                    } else if (opt.objective == SAObjective::kTargetSide) {
                        const double v = std::max(0.0, tmp_bounds.s - opt.target_side);
                        tmp_primary = (opt.target_power == 1) ? v : (v * v);
                    } else {
                        tmp_primary = tmp_bounds.s;
                    }

                    if (!found_any) {
                        best_pose = tmp;
                        best_local = tmp_local;
                        best_world = tmp_world;
                        best_bounds = tmp_bounds;
                        best_secondary_val = tmp_secondary;
                        best_primary = tmp_primary;
                        found_any = true;
                        continue;
                    }

                    bool better = false;
                    if (opt.objective == SAObjective::kPrefixScore) {
                        better = (tmp_primary < best_primary - 1e-15);
                    } else if (opt.objective == SAObjective::kTargetSide) {
                        if (tmp_primary < best_primary - 1e-15) {
                            better = true;
                        } else if (std::abs(tmp_primary - best_primary) <= 1e-15) {
                            if (tmp_bounds.s < best_bounds.s - 1e-12) {
                                better = true;
                            } else if (std::abs(tmp_bounds.s - best_bounds.s) <= 1e-12 &&
                                       opt.secondary != SASecondary::kNone && opt.secondary_weight > 0.0 &&
                                       tmp_secondary < best_secondary_val - 1e-12) {
                                better = true;
                            }
                        }
                    } else {
                        const double s_best = best_bounds.s;
                        const double s_tmp = tmp_bounds.s;
                        if (s_tmp < s_best - 1e-12) {
                            better = true;
                        } else if (std::abs(s_tmp - s_best) <= 1e-12 &&
                                   opt.secondary != SASecondary::kNone && opt.secondary_weight > 0.0 &&
                                   tmp_secondary < best_secondary_val - 1e-12) {
                            better = true;
                        }
                    }
                    if (better) {
                        best_pose = tmp;
                        best_local = tmp_local;
                        best_world = tmp_world;
                        best_bounds = tmp_bounds;
                        best_secondary_val = tmp_secondary;
                        best_primary = tmp_primary;
                    }
                }

                if (!found_any) {
                    continue;
                }

                cand = best_pose;
                cand_local = best_local;
                cand_world = best_world;
                cand_has_bbs = true;
            } else {
                if (unif01(rng) < opt.rot_prob) {
                    cand.deg = pick_angle(opt, i, old.deg, rng, unif01);
                }
                cand.x += norm01(rng) * sigma;
                cand.y += norm01(rng) * sigma;
                cand = clamp_pose(cand);
            }

            if (!within_coord_bounds(cand, kCoordMin, kCoordMax, eps)) {
                continue;
            }
            if (index.collides_with_any(cand, i, opt.safety_eps)) {
                continue;
            }

            res.feasible++;
            feasible_this_log++;
            feasible_window++;

            poses[static_cast<size_t>(i)] = cand;
            if (!cand_has_bbs) {
                cand_local = rotated_bbox_cached(tree_poly, cand.deg, bbox_cache);
                cand_world = bbox_for_pose(cand_local, cand);
            }
            local_bbs[static_cast<size_t>(i)] = cand_local;
            world_bbs[static_cast<size_t>(i)] = cand_world;
            index.set_pose(i, cand);
            const BoundsMetrics new_bounds = bounds_metrics(world_bbs);
            const double new_secondary = secondary_value(opt, new_bounds);

            double new_cost = 0.0;
            double d = 0.0;
            if (opt.objective == SAObjective::kPrefixScore) {
                new_cost = prefix_score_from_bbs(world_bbs, opt.nmax_score);
                d = (new_cost - current_cost) * opt.delta_scale;
            } else if (opt.objective == SAObjective::kTargetSide) {
                const double v_old = std::max(0.0, cur_bounds.s - opt.target_side);
                const double v_new = std::max(0.0, new_bounds.s - opt.target_side);
                new_cost = (opt.target_power == 1) ? v_new : (v_new * v_new);
                d = (new_cost - current_cost) * opt.delta_scale;
                if (opt.secondary != SASecondary::kNone && opt.secondary_weight > 0.0 &&
                    std::abs(v_new - v_old) <= 1e-12) {
                    d += opt.secondary_weight * (new_secondary - cur_secondary);
                }
            } else {
                const double s_old = cur_bounds.s;
                const double s_new = new_bounds.s;
                new_cost = s_new;

                const double s_old2 = s_old * s_old;
                const double s_new2 = s_new * s_new;
                double delta = 0.0;
                switch (opt.delta_mode) {
                case SADeltaMode::kLinear:
                    delta = s_new - s_old;
                    break;
                case SADeltaMode::kSquared:
                    delta = s_new2 - s_old2;
                    break;
                case SADeltaMode::kSquaredOverS:
                    delta = (s_new2 - s_old2) / std::max(1e-12, s_old);
                    break;
                case SADeltaMode::kSquaredOverS2:
                    delta = (s_new2 - s_old2) / std::max(1e-12, s_old2);
                    break;
                }

                d = delta * opt.delta_scale;
                if (opt.secondary != SASecondary::kNone && opt.secondary_weight > 0.0 &&
                    std::abs(s_new - s_old) <= 1e-12) {
                    d += opt.secondary_weight * (new_secondary - cur_secondary);
                }
            }

            bool accept = false;
            if (d <= 0.0) {
                accept = true;
            } else {
                const double p = std::exp(-d / T);
                accept = unif01(rng) < p;
            }

            if (accept) {
                res.accepted++;
                accepted_this_log++;
                accepted_window++;
                current_cost = new_cost;
                cur_bounds = new_bounds;
                cur_secondary = new_secondary;

                const double s200 = new_bounds.s;
                const double score = prefix_score_from_bbs(world_bbs, 200);

                bool improved_best = false;
                if (opt.objective == SAObjective::kPrefixScore) {
                    improved_best = (new_cost < res.best_cost - 1e-15);
                } else if (opt.objective == SAObjective::kTargetSide) {
                    if (new_cost < res.best_cost - 1e-15) {
                        improved_best = true;
                    } else if (std::abs(new_cost - res.best_cost) <= 1e-15) {
                        if (s200 < best_s - 1e-12) {
                            improved_best = true;
                        } else if (std::abs(s200 - best_s) <= 1e-12 &&
                                   opt.secondary != SASecondary::kNone && opt.secondary_weight > 0.0 &&
                                   new_secondary < best_secondary - 1e-12) {
                            improved_best = true;
                        }
                    }
                } else {
                    if (s200 < best_s - 1e-12) {
                        improved_best = true;
                    } else if (std::abs(s200 - best_s) <= 1e-12 &&
                               opt.secondary != SASecondary::kNone && opt.secondary_weight > 0.0 &&
                               new_secondary < best_secondary - 1e-12) {
                        improved_best = true;
                    }
                }

                if (improved_best) {
                    if (opt.objective == SAObjective::kPrefixScore || opt.objective == SAObjective::kTargetSide) {
                        res.best_cost = new_cost;
                    } else {
                        res.best_cost = s200;
                    }
                    res.best_poses = poses;
                    res.best_s200 = s200;
                    res.best_prefix_score = score;
                    best_s = s200;
                    best_secondary = new_secondary;
                }
                break;
            }

            // Reject: revert.
            poses[static_cast<size_t>(i)] = old;
            local_bbs[static_cast<size_t>(i)] = old_local;
            world_bbs[static_cast<size_t>(i)] = old_world;
            index.set_pose(i, old);
        }

        if (opt.log_every > 0 && (it % opt.log_every) == 0) {
            const std::string prefix = opt.log_prefix.empty() ? std::string("[sa]") : opt.log_prefix;
            const double cur_s200 = bounds_metrics(world_bbs).s;
            const double cur_score200 = prefix_score_from_bbs(world_bbs, 200);
            const double acc = static_cast<double>(accepted_this_log) / static_cast<double>(std::max(1, feasible_this_log));
            {
                std::lock_guard<std::mutex> lk(log_mutex());
                std::cerr << prefix << " it=" << it << "/" << opt.iters << " frac=" << frac << " T=" << T
                          << " sigma=" << sigma << " scale=" << adapt_scale << " attempted=" << attempted_this_log
                          << " feasible=" << feasible_this_log << " accepted=" << accepted_this_log << " acc=" << acc;
                if (opt.objective == SAObjective::kPrefixScore) {
                    std::cerr << " nmax=" << opt.nmax_score;
                } else if (opt.objective == SAObjective::kTargetSide) {
                    const double viol = std::max(0.0, cur_s200 - opt.target_side);
                    std::cerr << " target=" << opt.target_side << " viol=" << viol;
                }
                std::cerr << " cur_cost=" << current_cost << " best_cost=" << res.best_cost << " cur_s200=" << cur_s200
                          << " best_s200=" << res.best_s200 << " cur_score200=" << cur_score200
                          << " best_score200=" << res.best_prefix_score << " nfp_deltas=" << nfp_verts_cache.size()
                          << "\n";
            }
            attempted_this_log = 0;
            feasible_this_log = 0;
            accepted_this_log = 0;
        }
    }

    if (opt.log_every > 0) {
        const std::string prefix = opt.log_prefix.empty() ? std::string("[sa]") : opt.log_prefix;
        const double cur_s200 = bounds_metrics(world_bbs).s;
        const double cur_score200 = prefix_score_from_bbs(world_bbs, 200);
        {
            std::lock_guard<std::mutex> lk(log_mutex());
            std::cerr << prefix << " done it=" << opt.iters << "/" << opt.iters << " cur_cost=" << current_cost
                      << " best_cost=" << res.best_cost << " cur_s200=" << cur_s200 << " best_s200=" << res.best_s200
                      << " cur_score200=" << cur_score200 << " best_score200=" << res.best_prefix_score << "\n";
        }
    }

    return res;
}

}  // namespace santa2025
