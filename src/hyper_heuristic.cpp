#include "santa2025/hyper_heuristic.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "santa2025/collision_index.hpp"
#include "santa2025/constraints.hpp"
#include "santa2025/logging.hpp"

namespace santa2025 {
namespace {

struct Candidate {
    std::vector<Pose> poses;
    double s = 0.0;
    double score = 0.0;
    double obj = 0.0;
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

std::uint64_t pack_xy_key(std::int32_t qx, std::int32_t qy) {
    const std::uint64_t ux = static_cast<std::uint32_t>(qx);
    const std::uint64_t uy = static_cast<std::uint32_t>(qy);
    return (ux << 32) | uy;
}

constexpr double kQuant = 1e6;

std::int64_t quant_deg(double deg) {
    return static_cast<std::int64_t>(std::llround(deg * kQuant));
}

struct RotatedBBoxCache {
    std::unordered_map<std::int64_t, BoundingBox> bboxes;
};

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

double prefix_score_from_bbs(const std::vector<BoundingBox>& world_bbs, int nmax) {
    if (nmax <= 0 || world_bbs.empty()) {
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

double prefix_score_replace_one(
    const std::vector<BoundingBox>& world_bbs,
    int id,
    const BoundingBox& bb_new,
    int nmax
) {
    if (nmax <= 0 || world_bbs.empty()) {
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

bool cycle_applies(const std::vector<double>& cycle_deg, int cycle_prefix, int i) {
    if (cycle_deg.empty()) {
        return false;
    }
    if (cycle_prefix <= 0) {
        return true;
    }
    return i < cycle_prefix;
}

double pick_angle(
    const SAOptions& opt,
    int i,
    double current_deg,
    std::mt19937_64& rng,
    std::uniform_real_distribution<double>& unif01
) {
    if (opt.orientation_mode == OrientationMode::kCycle && cycle_applies(opt.cycle_deg, opt.cycle_prefix, i)) {
        return opt.cycle_deg[static_cast<size_t>(i) % opt.cycle_deg.size()];
    }

    const bool bias_cycle =
        (opt.orientation_mode == OrientationMode::kCycleThenAll && cycle_applies(opt.cycle_deg, opt.cycle_prefix, i) &&
         !opt.cycle_deg.empty());
    if (bias_cycle && unif01(rng) < 0.5) {
        return opt.cycle_deg[static_cast<size_t>(i) % opt.cycle_deg.size()];
    }

    if (opt.angles_deg.empty()) {
        return current_deg;
    }
    std::uniform_int_distribution<int> pick(0, static_cast<int>(opt.angles_deg.size()) - 1);
    return opt.angles_deg[static_cast<size_t>(pick(rng))];
}

std::vector<double> angle_candidates(
    const std::vector<double>& angles_deg,
    const std::vector<double>& cycle_deg,
    OrientationMode orientation_mode,
    int cycle_prefix,
    int i
) {
    if (orientation_mode == OrientationMode::kTryAll) {
        return angles_deg;
    }
    if (!cycle_applies(cycle_deg, cycle_prefix, i)) {
        return angles_deg;
    }
    const double cyc = cycle_deg[static_cast<size_t>(i) % cycle_deg.size()];
    if (orientation_mode == OrientationMode::kCycle) {
        return {cyc};
    }

    std::vector<double> out;
    out.reserve(angles_deg.size() + 1);
    out.push_back(cyc);
    for (const double a : angles_deg) {
        if (std::abs(a - cyc) <= 1e-12) {
            continue;
        }
        out.push_back(a);
    }
    return out;
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

bool normalize_to_quadrant(const Polygon& tree_poly, std::vector<Pose>& poses, CollisionIndex* index, double eps) {
    if (poses.empty()) {
        return true;
    }
    RotatedBBoxCache bbox_cache;
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

int default_lahc_length(int n) {
    const int v = 10 * n;
    return std::clamp(v, 50, 500);
}

double objective_value(HHObjective obj, double s, double score) {
    switch (obj) {
    case HHObjective::kS:
        return s;
    case HHObjective::kPrefixScore:
        return score;
    }
    return s;
}

struct Operator {
    std::string name;
    // Apply returns a feasible candidate (operators are expected to keep feasibility).
    // Empty poses means "no candidate" (treated as infeasible).
    Candidate (*apply)(const Polygon&, const std::vector<Pose>&, const HHOptions&, std::mt19937_64&, double eps);
};

Candidate op_move_single_translate(
    const Polygon& poly,
    const std::vector<Pose>& cur,
    const HHOptions& opt,
    std::mt19937_64& rng,
    double eps
) {
    Candidate out;
    const int n = opt.n;
    if (n <= 0) {
        return out;
    }

    RotatedBBoxCache bbox_cache;
    std::vector<BoundingBox> local_bbs(static_cast<size_t>(n));
    std::vector<BoundingBox> world_bbs(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const Pose& p = cur[static_cast<size_t>(i)];
        local_bbs[static_cast<size_t>(i)] = rotated_bbox_cached(poly, p.deg, bbox_cache);
        world_bbs[static_cast<size_t>(i)] = bbox_for_pose(local_bbs[static_cast<size_t>(i)], p);
    }
    const BoundsMetrics m = bounds_metrics(world_bbs);
    const double tol = std::max(1e-9, 1e-6 * m.s);
    const auto bnd = boundary_ids(world_bbs, m, tol);

    CollisionIndex index(poly, eps);
    index.resize(n);
    for (int i = 0; i < n; ++i) {
        index.set_pose(i, cur[static_cast<size_t>(i)]);
    }

    std::uniform_real_distribution<double> unif01(0.0, 1.0);
    std::uniform_int_distribution<int> pick_any(0, n - 1);
    std::uniform_int_distribution<int> pick_bnd(0, bnd.empty() ? 0 : (static_cast<int>(bnd.size()) - 1));

    const int id =
        (!bnd.empty() && unif01(rng) < opt.sa_base.boundary_prob) ? bnd[static_cast<size_t>(pick_bnd(rng))] : pick_any(rng);

    std::normal_distribution<double> norm01(0.0, 1.0);
    const double sigma = std::max(1e-9, opt.sa_base.trans_sigma0);
    const double dx = norm01(rng) * sigma;
    const double dy = norm01(rng) * sigma;

    std::vector<Pose> poses = cur;
    Pose cand = poses[static_cast<size_t>(id)];
    cand.x += dx;
    cand.y += dy;
    cand = clamp_pose_xy(cand, kCoordMin, kCoordMax);

    const double extra_eps = std::max(opt.sa_base.safety_eps, opt.lns_base.safety_eps);
    if (index.collides_with_any(cand, id, extra_eps)) {
        return out;
    }

    poses[static_cast<size_t>(id)] = cand;

    const BoundingBox cand_world_bb = bbox_for_pose(local_bbs[static_cast<size_t>(id)], cand);
    const BoundsMetrics new_bounds = bounds_metrics_replace_one(world_bbs, id, cand_world_bb);

    out.poses = std::move(poses);
    out.s = new_bounds.s;
    out.score = prefix_score_replace_one(
        world_bbs,
        id,
        cand_world_bb,
        (opt.objective == HHObjective::kPrefixScore) ? opt.nmax_score : 200
    );
    out.obj = objective_value(opt.objective, out.s, out.score);
    return out;
}

Candidate op_move_single_rotate(
    const Polygon& poly,
    const std::vector<Pose>& cur,
    const HHOptions& opt,
    std::mt19937_64& rng,
    double eps
) {
    Candidate out;
    const int n = opt.n;
    if (n <= 0) {
        return out;
    }

    RotatedBBoxCache bbox_cache;
    std::vector<BoundingBox> local_bbs(static_cast<size_t>(n));
    std::vector<BoundingBox> world_bbs(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const Pose& p = cur[static_cast<size_t>(i)];
        local_bbs[static_cast<size_t>(i)] = rotated_bbox_cached(poly, p.deg, bbox_cache);
        world_bbs[static_cast<size_t>(i)] = bbox_for_pose(local_bbs[static_cast<size_t>(i)], p);
    }
    const BoundsMetrics m = bounds_metrics(world_bbs);
    const double tol = std::max(1e-9, 1e-6 * m.s);
    const auto bnd = boundary_ids(world_bbs, m, tol);

    CollisionIndex index(poly, eps);
    index.resize(n);
    for (int i = 0; i < n; ++i) {
        index.set_pose(i, cur[static_cast<size_t>(i)]);
    }

    std::uniform_real_distribution<double> unif01(0.0, 1.0);
    std::uniform_int_distribution<int> pick_any(0, n - 1);
    std::uniform_int_distribution<int> pick_bnd(0, bnd.empty() ? 0 : (static_cast<int>(bnd.size()) - 1));

    const int id =
        (!bnd.empty() && unif01(rng) < opt.sa_base.boundary_prob) ? bnd[static_cast<size_t>(pick_bnd(rng))] : pick_any(rng);

    std::vector<Pose> poses = cur;
    Pose cand = poses[static_cast<size_t>(id)];
    cand.deg = pick_angle(opt.sa_base, id, cand.deg, rng, unif01);

    const BoundingBox cand_local_bb = rotated_bbox_cached(poly, cand.deg, bbox_cache);
    const BoundingBox cand_world_bb = bbox_for_pose(cand_local_bb, cand);

    const double extra_eps = std::max(opt.sa_base.safety_eps, opt.lns_base.safety_eps);
    if (index.collides_with_any(cand, id, extra_eps)) {
        return out;
    }

    poses[static_cast<size_t>(id)] = cand;

    const BoundsMetrics new_bounds = bounds_metrics_replace_one(world_bbs, id, cand_world_bb);

    out.poses = std::move(poses);
    out.s = new_bounds.s;
    out.score = prefix_score_replace_one(
        world_bbs,
        id,
        cand_world_bb,
        (opt.objective == HHObjective::kPrefixScore) ? opt.nmax_score : 200
    );
    out.obj = objective_value(opt.objective, out.s, out.score);
    return out;
}

Candidate op_nudge_boundary(
    const Polygon& poly,
    const std::vector<Pose>& cur,
    const HHOptions& opt,
    std::mt19937_64& rng,
    double eps
) {
    Candidate out;
    const int n = opt.n;
    if (n <= 0) {
        return out;
    }

    RotatedBBoxCache bbox_cache;
    std::vector<BoundingBox> local_bbs(static_cast<size_t>(n));
    std::vector<BoundingBox> world_bbs(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const Pose& p = cur[static_cast<size_t>(i)];
        local_bbs[static_cast<size_t>(i)] = rotated_bbox_cached(poly, p.deg, bbox_cache);
        world_bbs[static_cast<size_t>(i)] = bbox_for_pose(local_bbs[static_cast<size_t>(i)], p);
    }
    const BoundsMetrics m = bounds_metrics(world_bbs);
    const double tol = std::max(1e-9, 1e-6 * m.s);
    const auto bnd = boundary_ids(world_bbs, m, tol);

    CollisionIndex index(poly, eps);
    index.resize(n);
    for (int i = 0; i < n; ++i) {
        index.set_pose(i, cur[static_cast<size_t>(i)]);
    }

    std::uniform_int_distribution<int> pick_any(0, n - 1);
    std::uniform_int_distribution<int> pick_bnd(0, bnd.empty() ? 0 : (static_cast<int>(bnd.size()) - 1));
    const int id = bnd.empty() ? pick_any(rng) : bnd[static_cast<size_t>(pick_bnd(rng))];

    const Pose old = cur[static_cast<size_t>(id)];
    const BoundingBox old_world_bb = world_bbs[static_cast<size_t>(id)];

    const double step = std::max(1e-6, 0.5 * std::max(1e-9, opt.sa_base.trans_sigma0));
    double dx = 0.0;
    double dy = 0.0;
    if (old_world_bb.min_x <= m.min_x + tol) {
        dx += step;
    }
    if (old_world_bb.max_x >= m.max_x - tol) {
        dx -= step;
    }
    if (old_world_bb.min_y <= m.min_y + tol) {
        dy += step;
    }
    if (old_world_bb.max_y >= m.max_y - tol) {
        dy -= step;
    }

    std::vector<std::pair<double, double>> offsets;
    if (dx != 0.0) {
        offsets.emplace_back(dx, 0.0);
    }
    if (dy != 0.0) {
        offsets.emplace_back(0.0, dy);
    }
    if (dx != 0.0 || dy != 0.0) {
        offsets.emplace_back(dx, dy);
    }
    if (offsets.empty()) {
        return out;
    }

    const double extra_eps = std::max(opt.sa_base.safety_eps, opt.lns_base.safety_eps);
    bool found = false;
    Pose best_pose;
    BoundingBox best_world_bb;
    BoundsMetrics best_bounds;
    double best_score = 0.0;
    double best_obj = 0.0;

    for (const auto& [ox, oy] : offsets) {
        Pose cand = old;
        cand.x += ox;
        cand.y += oy;
        cand = clamp_pose_xy(cand, kCoordMin, kCoordMax);
        if (index.collides_with_any(cand, id, extra_eps)) {
            continue;
        }
        const BoundingBox cand_world_bb = bbox_for_pose(local_bbs[static_cast<size_t>(id)], cand);
        const BoundsMetrics new_bounds = bounds_metrics_replace_one(world_bbs, id, cand_world_bb);
        const double new_s = new_bounds.s;
        const double new_score = prefix_score_replace_one(
            world_bbs,
            id,
            cand_world_bb,
            (opt.objective == HHObjective::kPrefixScore) ? opt.nmax_score : 200
        );
        const double new_obj = objective_value(opt.objective, new_s, new_score);
        if (!found || new_obj < best_obj - 1e-15) {
            found = true;
            best_pose = cand;
            best_world_bb = cand_world_bb;
            best_bounds = new_bounds;
            best_score = new_score;
            best_obj = new_obj;
        }
    }

    if (!found) {
        return out;
    }

    std::vector<Pose> poses = cur;
    poses[static_cast<size_t>(id)] = best_pose;
    out.poses = std::move(poses);
    out.s = best_bounds.s;
    out.score = best_score;
    out.obj = best_obj;
    return out;
}

Candidate op_cluster_move(
    const Polygon& poly,
    const std::vector<Pose>& cur,
    const HHOptions& opt,
    std::mt19937_64& rng,
    double eps
) {
    Candidate out;
    const int n = opt.n;
    if (n <= 0) {
        return out;
    }
    if (n < opt.sa_base.cluster_min || opt.sa_base.cluster_min < 2) {
        return out;
    }

    RotatedBBoxCache bbox_cache;
    std::vector<BoundingBox> local_bbs(static_cast<size_t>(n));
    std::vector<BoundingBox> world_bbs(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const Pose& p = cur[static_cast<size_t>(i)];
        local_bbs[static_cast<size_t>(i)] = rotated_bbox_cached(poly, p.deg, bbox_cache);
        world_bbs[static_cast<size_t>(i)] = bbox_for_pose(local_bbs[static_cast<size_t>(i)], p);
    }
    const BoundsMetrics m = bounds_metrics(world_bbs);
    const double tol = std::max(1e-9, 1e-6 * m.s);
    const auto bnd = boundary_ids(world_bbs, m, tol);

    CollisionIndex index(poly, eps);
    index.resize(n);
    for (int i = 0; i < n; ++i) {
        index.set_pose(i, cur[static_cast<size_t>(i)]);
    }

    std::uniform_real_distribution<double> unif01(0.0, 1.0);
    std::uniform_int_distribution<int> pick_any(0, n - 1);
    std::uniform_int_distribution<int> pick_bnd(0, bnd.empty() ? 0 : (static_cast<int>(bnd.size()) - 1));

    const int seed_id =
        (!bnd.empty() && unif01(rng) < opt.sa_base.boundary_prob) ? bnd[static_cast<size_t>(pick_bnd(rng))] : pick_any(rng);

    const double r = opt.sa_base.cluster_radius_mult * index.tree_radius();
    const double r2 = r * r;

    std::vector<int> cluster;
    cluster.reserve(static_cast<size_t>(opt.sa_base.cluster_max));
    cluster.push_back(seed_id);
    const Pose& seed_pose = cur[static_cast<size_t>(seed_id)];
    for (int j = 0; j < n; ++j) {
        if (j == seed_id) {
            continue;
        }
        const Pose& pj = cur[static_cast<size_t>(j)];
        const double ddx = pj.x - seed_pose.x;
        const double ddy = pj.y - seed_pose.y;
        if (ddx * ddx + ddy * ddy <= r2) {
            cluster.push_back(j);
        }
    }

    if (static_cast<int>(cluster.size()) < opt.sa_base.cluster_min) {
        return out;
    }
    if (static_cast<int>(cluster.size()) > opt.sa_base.cluster_max) {
        std::shuffle(cluster.begin() + 1, cluster.end(), rng);
        cluster.resize(static_cast<size_t>(opt.sa_base.cluster_max));
    }

    for (const int id : cluster) {
        index.remove(id);
    }

    std::normal_distribution<double> norm01(0.0, 1.0);
    const double sigma = std::max(1e-9, opt.sa_base.trans_sigma0);
    const double s_cluster = sigma * opt.sa_base.cluster_sigma_mult;
    const double dx = norm01(rng) * s_cluster;
    const double dy = norm01(rng) * s_cluster;

    const double extra_eps = std::max(opt.sa_base.safety_eps, opt.lns_base.safety_eps);
    for (const int id : cluster) {
        Pose p = cur[static_cast<size_t>(id)];
        p.x += dx;
        p.y += dy;
        if (!within_coord_bounds(p, kCoordMin, kCoordMax, eps)) {
            return out;
        }
        if (index.collides_with_any(p, -1, extra_eps)) {
            return out;
        }
    }

    std::vector<Pose> poses = cur;
    for (const int id : cluster) {
        poses[static_cast<size_t>(id)].x += dx;
        poses[static_cast<size_t>(id)].y += dy;
        world_bbs[static_cast<size_t>(id)] = BoundingBox{
            world_bbs[static_cast<size_t>(id)].min_x + dx,
            world_bbs[static_cast<size_t>(id)].min_y + dy,
            world_bbs[static_cast<size_t>(id)].max_x + dx,
            world_bbs[static_cast<size_t>(id)].max_y + dy,
        };
    }

    const BoundsMetrics new_bounds = bounds_metrics(world_bbs);

    out.poses = std::move(poses);
    out.s = new_bounds.s;
    out.score = (opt.objective == HHObjective::kPrefixScore) ? prefix_score_from_bbs(world_bbs, opt.nmax_score)
                                                             : prefix_score_from_bbs(world_bbs, 200);
    out.obj = objective_value(opt.objective, out.s, out.score);
    return out;
}

Candidate op_touch_best_of(
    const Polygon& poly,
    const std::vector<Pose>& cur,
    const HHOptions& opt,
    std::mt19937_64& rng,
    double eps
) {
    Candidate out;
    const int n = opt.n;
    if (n <= 1) {
        return out;
    }

    RotatedBBoxCache bbox_cache;
    std::vector<BoundingBox> local_bbs(static_cast<size_t>(n));
    std::vector<BoundingBox> world_bbs(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const Pose& p = cur[static_cast<size_t>(i)];
        local_bbs[static_cast<size_t>(i)] = rotated_bbox_cached(poly, p.deg, bbox_cache);
        world_bbs[static_cast<size_t>(i)] = bbox_for_pose(local_bbs[static_cast<size_t>(i)], p);
    }

    CollisionIndex index(poly, eps);
    index.resize(n);
    for (int i = 0; i < n; ++i) {
        index.set_pose(i, cur[static_cast<size_t>(i)]);
    }

    std::uniform_real_distribution<double> unif01(0.0, 1.0);
    std::normal_distribution<double> norm01(0.0, 1.0);
    std::uniform_int_distribution<int> pick_id(0, n - 1);

    const int i = pick_id(rng);
    int j = pick_id(rng);
    if (j == i) {
        j = (j + 1) % n;
    }

    const Pose old = cur[static_cast<size_t>(i)];
    const Pose& anchor = cur[static_cast<size_t>(j)];
    const double ang = pick_angle(opt.sa_base, i, old.deg, rng, unif01);

    const double delta = ang - anchor.deg;
    const auto& nfp = index.nfp(delta);
    const auto offsets = nfp_vertices_sorted_limited(nfp, opt.sa_base.max_offsets_per_delta, eps);
    if (offsets.empty()) {
        return out;
    }

    const int k = std::max(1, opt.sa_base.touch_best_of);
    std::uniform_int_distribution<int> pick_off(0, static_cast<int>(offsets.size()) - 1);

    const double extra_eps = std::max(opt.sa_base.safety_eps, opt.lns_base.safety_eps);
    const double sigma = std::max(1e-9, opt.sa_base.trans_sigma0);
    const double j_sigma = 0.25 * sigma;

    bool found = false;
    Pose best_pose;
    BoundingBox best_world_bb;
    BoundsMetrics best_bounds;
    double best_score = 0.0;
    double best_obj = 0.0;

    for (int t = 0; t < k; ++t) {
        const Point v = offsets[static_cast<size_t>(pick_off(rng))];
        const double nrm = std::hypot(v.x, v.y);
        if (!(nrm > 0.0)) {
            continue;
        }
        const Point u{v.x / nrm, v.y / nrm};
        const Point v_out{v.x + opt.sa_base.gap * u.x, v.y + opt.sa_base.gap * u.y};
        const Point t_world = rotate_point(v_out, anchor.deg);

        Pose cand;
        cand.deg = ang;
        cand.x = anchor.x + t_world.x + norm01(rng) * j_sigma;
        cand.y = anchor.y + t_world.y + norm01(rng) * j_sigma;
        cand = clamp_pose_xy(cand, kCoordMin, kCoordMax);

        if (index.collides_with_any(cand, i, extra_eps)) {
            continue;
        }

        const BoundingBox cand_local_bb = rotated_bbox_cached(poly, cand.deg, bbox_cache);
        const BoundingBox cand_world_bb = bbox_for_pose(cand_local_bb, cand);
        const BoundsMetrics new_bounds = bounds_metrics_replace_one(world_bbs, i, cand_world_bb);
        const double new_s = new_bounds.s;
        const double new_score = prefix_score_replace_one(
            world_bbs,
            i,
            cand_world_bb,
            (opt.objective == HHObjective::kPrefixScore) ? opt.nmax_score : 200
        );
        const double new_obj = objective_value(opt.objective, new_s, new_score);
        if (!found || new_obj < best_obj - 1e-15) {
            found = true;
            best_pose = cand;
            best_world_bb = cand_world_bb;
            best_bounds = new_bounds;
            best_score = new_score;
            best_obj = new_obj;
        }
    }

    if (!found) {
        return out;
    }

    std::vector<Pose> poses = cur;
    poses[static_cast<size_t>(i)] = best_pose;
    out.poses = std::move(poses);
    out.s = best_bounds.s;
    out.score = best_score;
    out.obj = best_obj;
    return out;
}

Candidate op_reinsert_one(
    const Polygon& poly,
    const std::vector<Pose>& cur,
    const HHOptions& opt,
    std::mt19937_64& rng,
    double eps
) {
    Candidate out;
    const int n = opt.n;
    if (n <= 1) {
        return out;
    }

    std::vector<Pose> poses = cur;
    CollisionIndex index(poly, eps);
    index.resize(n);
    for (int i = 0; i < n; ++i) {
        index.set_pose(i, poses[static_cast<size_t>(i)]);
    }
    if (!normalize_to_quadrant(poly, poses, &index, eps)) {
        return out;
    }

    RotatedBBoxCache bbox_cache;
    std::vector<BoundingBox> local_bbs(static_cast<size_t>(n));
    std::vector<BoundingBox> world_bbs(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const Pose& p = poses[static_cast<size_t>(i)];
        local_bbs[static_cast<size_t>(i)] = rotated_bbox_cached(poly, p.deg, bbox_cache);
        world_bbs[static_cast<size_t>(i)] = bbox_for_pose(local_bbs[static_cast<size_t>(i)], p);
    }
    const BoundsMetrics m = bounds_metrics(world_bbs);
    const double tol = std::max(1e-9, 1e-6 * m.s);
    const auto bnd = boundary_ids(world_bbs, m, tol);

    std::uniform_int_distribution<int> pick_any(0, n - 1);
    std::uniform_int_distribution<int> pick_bnd(0, bnd.empty() ? 0 : (static_cast<int>(bnd.size()) - 1));
    std::uniform_real_distribution<double> unif01(0.0, 1.0);

    const int id =
        (!bnd.empty() && unif01(rng) < opt.lns_base.boundary_prob) ? bnd[static_cast<size_t>(pick_bnd(rng))] : pick_any(rng);

    const Pose original = poses[static_cast<size_t>(id)];
    index.remove(id);

    bool bounds_init = false;
    double cur_min_x = 0.0;
    double cur_min_y = 0.0;
    double cur_max_x = 0.0;
    double cur_max_y = 0.0;
    for (int j = 0; j < n; ++j) {
        if (j == id) {
            continue;
        }
        const auto& bb = world_bbs[static_cast<size_t>(j)];
        if (!bounds_init) {
            bounds_init = true;
            cur_min_x = bb.min_x;
            cur_min_y = bb.min_y;
            cur_max_x = bb.max_x;
            cur_max_y = bb.max_y;
        } else {
            cur_min_x = std::min(cur_min_x, bb.min_x);
            cur_min_y = std::min(cur_min_y, bb.min_y);
            cur_max_x = std::max(cur_max_x, bb.max_x);
            cur_max_y = std::max(cur_max_y, bb.max_y);
        }
    }

    std::unordered_map<std::int64_t, std::vector<Point>> verts_cache;
    auto offsets_for = [&](double delta_deg) -> const std::vector<Point>& {
        const std::int64_t key = quant_deg(delta_deg);
        auto it = verts_cache.find(key);
        if (it != verts_cache.end()) {
            return it->second;
        }
        const auto& nfp = index.nfp(delta_deg);
        auto verts = nfp_vertices_sorted_limited(nfp, opt.lns_base.max_offsets_per_delta, eps);
        auto [ins, _ok] = verts_cache.emplace(key, std::move(verts));
        return ins->second;
    };

    const double side = m.s;

    Pose best = original;
    BoundingBox best_world_bb = world_bbs[static_cast<size_t>(id)];
    double best_s = std::numeric_limits<double>::infinity();
    bool found_any = false;

    auto consider = [&](const BoundingBox& cand_local_bb, Pose cand_start) {
        cand_start = slide_down_left(cand_start, cand_local_bb, index, side, opt.lns_base.safety_eps, opt.lns_base.slide_iters, eps);
        if (!std::isfinite(cand_start.x) || !std::isfinite(cand_start.y)) {
            return;
        }

        const BoundingBox cand_world_bb = bbox_for_pose(cand_local_bb, cand_start);
        double new_min_x = cand_world_bb.min_x;
        double new_min_y = cand_world_bb.min_y;
        double new_max_x = cand_world_bb.max_x;
        double new_max_y = cand_world_bb.max_y;
        if (bounds_init) {
            new_min_x = std::min(new_min_x, cur_min_x);
            new_min_y = std::min(new_min_y, cur_min_y);
            new_max_x = std::max(new_max_x, cur_max_x);
            new_max_y = std::max(new_max_y, cur_max_y);
        }
        const double new_s = std::max(new_max_x - new_min_x, new_max_y - new_min_y);

        found_any = true;
        if (new_s < best_s - 1e-12 ||
            (std::abs(new_s - best_s) <= 1e-12 &&
             (cand_start.y < best.y - 1e-12 ||
              (std::abs(cand_start.y - best.y) <= 1e-12 && cand_start.x < best.x)))) {
            best = cand_start;
            best_world_bb = cand_world_bb;
            best_s = new_s;
        }
    };

    const auto angs = angle_candidates(opt.lns_base.angles_deg, opt.lns_base.cycle_deg, opt.lns_base.orientation_mode, opt.lns_base.cycle_prefix, id);
    for (const double ang : angs) {
        const BoundingBox cand_local_bb = rotated_bbox_cached(poly, ang, bbox_cache);
        if (cand_local_bb.max_x - cand_local_bb.min_x > side + eps || cand_local_bb.max_y - cand_local_bb.min_y > side + eps) {
            continue;
        }

        Pose start;
        start.deg = ang;

        const double x_low = -cand_local_bb.min_x;
        const double y_low = -cand_local_bb.min_y;

        // Bottom-left corner.
        start.x = x_low;
        start.y = y_low;
        consider(cand_local_bb, start);

        for (int j = 0; j < n; ++j) {
            if (j == id) {
                continue;
            }
            const Pose& anchor = poses[static_cast<size_t>(j)];
            const BoundingBox anchor_local_bb = rotated_bbox_cached(poly, anchor.deg, bbox_cache);
            const BoundingBox anchor_world_bb = bbox_for_pose(anchor_local_bb, anchor);
            const double delta_deg = ang - anchor.deg;

            // Bounding box corners.
            consider(cand_local_bb, Pose{anchor_world_bb.max_x - cand_local_bb.min_x + opt.lns_base.gap, y_low, ang});
            consider(cand_local_bb, Pose{x_low, anchor_world_bb.max_y - cand_local_bb.min_y + opt.lns_base.gap, ang});
            consider(cand_local_bb,
                     Pose{anchor_world_bb.max_x - cand_local_bb.min_x + opt.lns_base.gap,
                          anchor_world_bb.max_y - cand_local_bb.min_y + opt.lns_base.gap,
                          ang});

            const auto& offsets = offsets_for(delta_deg);
            for (const auto& v : offsets) {
                const double nrm = std::hypot(v.x, v.y);
                if (!(nrm > 0.0)) {
                    continue;
                }
                const Point u{v.x / nrm, v.y / nrm};
                const Point v_out{v.x + opt.lns_base.gap * u.x, v.y + opt.lns_base.gap * u.y};
                const Point t_world = rotate_point(v_out, anchor.deg);
                consider(cand_local_bb, Pose{anchor.x + t_world.x, anchor.y + t_world.y, ang});
            }
        }
    }

    if (!found_any) {
        // Keep original (after normalization) so operator is always feasible/neutral.
        best = original;
        best_world_bb = bbox_for_pose(local_bbs[static_cast<size_t>(id)], best);
        best_s = m.s;
    }

    poses[static_cast<size_t>(id)] = best;

    // Recompute objective from updated bboxes (single replace is fine).
    world_bbs[static_cast<size_t>(id)] = best_world_bb;
    const BoundsMetrics new_bounds = bounds_metrics(world_bbs);

    out.poses = std::move(poses);
    out.s = new_bounds.s;
    out.score = (opt.objective == HHObjective::kPrefixScore) ? prefix_score_from_bbs(world_bbs, opt.nmax_score)
                                                             : prefix_score_from_bbs(world_bbs, 200);
    out.obj = objective_value(opt.objective, out.s, out.score);
    return out;
}

Candidate op_swap_poses_prefix(
    const Polygon& poly,
    const std::vector<Pose>& cur,
    const HHOptions& opt,
    std::mt19937_64& rng,
    double /*eps*/
) {
    Candidate out;
    if (opt.objective != HHObjective::kPrefixScore) {
        return out;
    }
    const int n = opt.n;
    if (n <= 1) {
        return out;
    }

    std::uniform_int_distribution<int> pick(0, n - 1);
    int i = pick(rng);
    int j = pick(rng);
    if (j == i) {
        j = (j + 1) % n;
    }

    std::vector<Pose> poses = cur;
    std::swap(poses[static_cast<size_t>(i)], poses[static_cast<size_t>(j)]);

    RotatedBBoxCache bbox_cache;
    std::vector<BoundingBox> world_bbs;
    world_bbs.reserve(poses.size());
    for (const auto& p : poses) {
        const BoundingBox local = rotated_bbox_cached(poly, p.deg, bbox_cache);
        world_bbs.push_back(bbox_for_pose(local, p));
    }

    const BoundsMetrics m = bounds_metrics(world_bbs);
    out.poses = std::move(poses);
    out.s = m.s;
    out.score = prefix_score_from_bbs(world_bbs, opt.nmax_score);
    out.obj = objective_value(opt.objective, out.s, out.score);
    return out;
}

Candidate op_sa_intensify(
    const Polygon& poly,
    const std::vector<Pose>& cur,
    const HHOptions& opt,
    std::mt19937_64& rng,
    double eps
) {
    SAOptions sa = opt.sa_base;
    sa.n = opt.n;
    sa.objective = (opt.objective == HHObjective::kPrefixScore) ? SAObjective::kPrefixScore : SAObjective::kS200;
    sa.nmax_score = opt.nmax_score;

    sa.iters = std::max(50, sa.iters);
    if (opt.sa_burst_iters > 0) {
        sa.iters = std::max(50, opt.sa_burst_iters);
    }
    sa.t0 = std::min(sa.t0, 0.10);
    sa.t1 = std::min(sa.t1, 1e-4);
    sa.trans_sigma0 = std::min(sa.trans_sigma0, 0.08);
    sa.trans_sigma1 = std::min(sa.trans_sigma1, 0.01);
    sa.boundary_prob = std::max(sa.boundary_prob, 0.6);
    sa.touch_prob = std::max(sa.touch_prob, 0.30);
    sa.cluster_prob = std::min(sa.cluster_prob, 0.15);

    sa.seed = rng();
    sa.log_every = 0;

    const auto res = simulated_annealing(poly, cur, sa, eps);
    Candidate out;
    out.poses = res.best_poses;
    out.s = packing_s200(poly, out.poses);
    out.score = packing_prefix_score(poly, out.poses, opt.nmax_score);
    out.obj = objective_value(opt.objective, out.s, out.score);
    return out;
}

Candidate op_sa_diversify(
    const Polygon& poly,
    const std::vector<Pose>& cur,
    const HHOptions& opt,
    std::mt19937_64& rng,
    double eps
) {
    SAOptions sa = opt.sa_base;
    sa.n = opt.n;
    sa.objective = (opt.objective == HHObjective::kPrefixScore) ? SAObjective::kPrefixScore : SAObjective::kS200;
    sa.nmax_score = opt.nmax_score;

    sa.iters = std::max(50, sa.iters);
    if (opt.sa_burst_iters > 0) {
        sa.iters = std::max(50, opt.sa_burst_iters);
    }
    sa.t0 = std::max(sa.t0, 0.35);
    sa.t1 = std::max(sa.t1, 1e-4);
    sa.trans_sigma0 = std::max(sa.trans_sigma0, 0.25);
    sa.trans_sigma1 = std::max(sa.trans_sigma1, 0.02);
    sa.boundary_prob = std::max(sa.boundary_prob, 0.4);
    sa.touch_prob = std::max(sa.touch_prob, 0.20);
    sa.cluster_prob = std::max(sa.cluster_prob, 0.15);

    sa.seed = rng();
    sa.log_every = 0;

    const auto res = simulated_annealing(poly, cur, sa, eps);
    Candidate out;
    out.poses = res.best_poses;
    out.s = packing_s200(poly, out.poses);
    out.score = packing_prefix_score(poly, out.poses, opt.nmax_score);
    out.obj = objective_value(opt.objective, out.s, out.score);
    return out;
}

Candidate op_lns_boundary_small(
    const Polygon& poly,
    const std::vector<Pose>& cur,
    const HHOptions& opt,
    std::mt19937_64& rng,
    double eps
) {
    LNSOptions lns = opt.lns_base;
    lns.n = opt.n;
    lns.stages = std::max(1, lns.stages);
    lns.stage_attempts = std::max(1, lns.stage_attempts);
    lns.remove_frac = std::clamp(lns.remove_frac, 0.03, 0.10);
    lns.boundary_prob = 1.0;
    lns.destroy_mode = LNSDestroyMode::kBoundary;
    lns.seed = rng();
    lns.log_every = 0;

    const auto r = lns_shrink_wrap(poly, cur, lns, eps);
    Candidate out;
    out.poses = r.best_poses;
    out.s = packing_s200(poly, out.poses);
    out.score = packing_prefix_score(poly, out.poses, opt.nmax_score);
    out.obj = objective_value(opt.objective, out.s, out.score);
    return out;
}

Candidate op_lns_boundary_large(
    const Polygon& poly,
    const std::vector<Pose>& cur,
    const HHOptions& opt,
    std::mt19937_64& rng,
    double eps
) {
    LNSOptions lns = opt.lns_base;
    lns.n = opt.n;
    lns.stages = std::max(1, lns.stages);
    lns.stage_attempts = std::max(1, lns.stage_attempts);
    lns.remove_frac = std::clamp(lns.remove_frac, 0.10, 0.25);
    lns.boundary_prob = 1.0;
    lns.destroy_mode = LNSDestroyMode::kBoundary;
    lns.seed = rng();
    lns.log_every = 0;

    const auto r = lns_shrink_wrap(poly, cur, lns, eps);
    Candidate out;
    out.poses = r.best_poses;
    out.s = packing_s200(poly, out.poses);
    out.score = packing_prefix_score(poly, out.poses, opt.nmax_score);
    out.obj = objective_value(opt.objective, out.s, out.score);
    return out;
}

Candidate op_lns_random(
    const Polygon& poly,
    const std::vector<Pose>& cur,
    const HHOptions& opt,
    std::mt19937_64& rng,
    double eps
) {
    LNSOptions lns = opt.lns_base;
    lns.n = opt.n;
    lns.stages = std::max(1, lns.stages);
    lns.stage_attempts = std::max(1, lns.stage_attempts);
    lns.boundary_prob = 0.0;
    lns.destroy_mode = LNSDestroyMode::kRandom;
    lns.seed = rng();
    lns.log_every = 0;

    const auto r = lns_shrink_wrap(poly, cur, lns, eps);
    Candidate out;
    out.poses = r.best_poses;
    out.s = packing_s200(poly, out.poses);
    out.score = packing_prefix_score(poly, out.poses, opt.nmax_score);
    out.obj = objective_value(opt.objective, out.s, out.score);
    return out;
}

Candidate op_lns_cluster(
    const Polygon& poly,
    const std::vector<Pose>& cur,
    const HHOptions& opt,
    std::mt19937_64& rng,
    double eps
) {
    LNSOptions lns = opt.lns_base;
    lns.n = opt.n;
    lns.stages = std::max(1, lns.stages);
    lns.stage_attempts = std::max(1, lns.stage_attempts);
    lns.boundary_prob = std::clamp(lns.boundary_prob, 0.0, 1.0);
    lns.destroy_mode = LNSDestroyMode::kCluster;
    lns.seed = rng();
    lns.log_every = 0;

    const auto r = lns_shrink_wrap(poly, cur, lns, eps);
    Candidate out;
    out.poses = r.best_poses;
    out.s = packing_s200(poly, out.poses);
    out.score = packing_prefix_score(poly, out.poses, opt.nmax_score);
    out.obj = objective_value(opt.objective, out.s, out.score);
    return out;
}

template <class StatsT>
int pick_ucb(const std::vector<StatsT>& stats, double ucb_c, int t) {
    if (stats.empty()) {
        return -1;
    }

    for (size_t i = 0; i < stats.size(); ++i) {
        if (stats[i].selected == 0) {
            return static_cast<int>(i);
        }
    }
    const double logt = std::log(static_cast<double>(t + 2));
    int best = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < stats.size(); ++i) {
        const double mean = stats[i].mean_reward;
        const double n = static_cast<double>(std::max(1, stats[i].selected));
        const double bonus = ucb_c * std::sqrt(logt / n);
        const double score = mean + bonus;
        if (score > best_score) {
            best_score = score;
            best = static_cast<int>(i);
        }
    }
    return best;
}

int pick_operator_alns(
    const std::vector<HHOperatorStats>& stats,
    const std::vector<int>& op_ids,
    std::mt19937_64& rng
) {
    if (op_ids.empty()) {
        return -1;
    }

    std::vector<int> untried;
    untried.reserve(op_ids.size());
    for (const int id : op_ids) {
        if (id < 0 || id >= static_cast<int>(stats.size())) {
            continue;
        }
        if (stats[static_cast<size_t>(id)].selected == 0) {
            untried.push_back(id);
        }
    }
    if (!untried.empty()) {
        std::uniform_int_distribution<int> pick(0, static_cast<int>(untried.size()) - 1);
        return untried[static_cast<size_t>(pick(rng))];
    }

    double total = 0.0;
    for (const int id : op_ids) {
        if (id < 0 || id >= static_cast<int>(stats.size())) {
            continue;
        }
        total += std::max(0.0, stats[static_cast<size_t>(id)].weight);
    }
    if (!(total > 0.0)) {
        std::uniform_int_distribution<int> pick(0, static_cast<int>(op_ids.size()) - 1);
        return op_ids[static_cast<size_t>(pick(rng))];
    }

    std::uniform_real_distribution<double> pickw(0.0, total);
    double r = pickw(rng);
    for (const int id : op_ids) {
        if (id < 0 || id >= static_cast<int>(stats.size())) {
            continue;
        }
        r -= std::max(0.0, stats[static_cast<size_t>(id)].weight);
        if (r <= 0.0) {
            return id;
        }
    }

    return op_ids.back();
}

double reward_delta_per_ms(double old_obj, double new_obj, double elapsed_ms, double eps_ms) {
    const double delta = std::max(0.0, old_obj - new_obj);
    const double denom = elapsed_ms + std::max(0.0, eps_ms);
    if (!(denom > 0.0)) {
        return 0.0;
    }
    return delta / denom;
}

void update_running_mean(double& mean, int count, double value) {
    const int n = std::max(1, count);
    mean += (value - mean) / static_cast<double>(n);
}

void update_alns_weight(double& w, double reward, double rho, double min_w) {
    const double rr = std::clamp(rho, 0.0, 1.0);
    w = (1.0 - rr) * w + rr * reward;
    w = std::max(w, min_w);
}

int default_sa_burst_after(int hh_iters) {
    // Heuristic: allow at least one burst for small budgets, without spamming bursts.
    return std::clamp(hh_iters / 3, 10, 200);
}

}  // namespace

HHResult hyper_heuristic_optimize(
    const Polygon& tree_poly,
    const std::vector<Pose>& initial,
    const HHOptions& opt,
    double eps
) {
    if (opt.n <= 0) {
        return HHResult{};
    }
    if (static_cast<int>(initial.size()) != opt.n) {
        throw std::invalid_argument("hyper_heuristic_optimize: initial.size() must equal opt.n");
    }
    if (opt.hh_iters <= 0) {
        throw std::invalid_argument("hyper_heuristic_optimize: hh_iters must be > 0");
    }
    if (!(opt.ucb_c >= 0.0)) {
        throw std::invalid_argument("hyper_heuristic_optimize: ucb_c must be >= 0");
    }
    if (!(opt.alns_rho >= 0.0 && opt.alns_rho <= 1.0)) {
        throw std::invalid_argument("hyper_heuristic_optimize: alns_rho must be in [0,1]");
    }
    if (!(opt.alns_min_weight > 0.0)) {
        throw std::invalid_argument("hyper_heuristic_optimize: alns_min_weight must be > 0");
    }
    if (!(opt.reward_time_eps_ms >= 0.0)) {
        throw std::invalid_argument("hyper_heuristic_optimize: reward_time_eps_ms must be >= 0");
    }
    if (opt.sa_burst_after < 0) {
        throw std::invalid_argument("hyper_heuristic_optimize: sa_burst_after must be >= 0");
    }
    if (opt.sa_burst_cooldown < 0) {
        throw std::invalid_argument("hyper_heuristic_optimize: sa_burst_cooldown must be >= 0");
    }
    if (opt.sa_burst_iters < 0) {
        throw std::invalid_argument("hyper_heuristic_optimize: sa_burst_iters must be >= 0");
    }

    const int L = (opt.lahc_length > 0) ? opt.lahc_length : default_lahc_length(opt.n);

    std::vector<Pose> cur = initial;
    const double init_s = packing_s200(tree_poly, cur);
    const double init_score = packing_prefix_score(tree_poly, cur, opt.nmax_score);
    double cur_s = init_s;
    double cur_score = init_score;
    double cur_obj = objective_value(opt.objective, cur_s, cur_score);

    std::vector<Pose> best = cur;
    double best_s = cur_s;
    double best_score = cur_score;
    double best_obj = cur_obj;

    std::vector<double> hist(static_cast<size_t>(L), cur_obj);

    std::mt19937_64 rng(opt.seed);

    struct MacroGroup {
        std::string name;
        std::vector<int> op_ids;
    };

    std::vector<Operator> ops;
    ops.reserve(16);

    auto add_op = [&](std::string name,
                      Candidate (*fn)(const Polygon&, const std::vector<Pose>&, const HHOptions&, std::mt19937_64&, double))
        -> int {
        ops.push_back(Operator{std::move(name), fn});
        return static_cast<int>(ops.size()) - 1;
    };

    // Small (cheap) local moves.
    const int op_move_translate_id = add_op("move_translate", &op_move_single_translate);
    const int op_move_rotate_id = add_op("move_rotate", &op_move_single_rotate);
    const int op_nudge_boundary_id = add_op("nudge_boundary", &op_nudge_boundary);

    // Medium moves.
    const int op_cluster_move_id = add_op("cluster_move", &op_cluster_move);
    const int op_touch_best_of_id = add_op("touch_best_of", &op_touch_best_of);
    const int op_reinsert_one_id = add_op("reinsert_one", &op_reinsert_one);
    const int op_swap_poses_id =
        (opt.objective == HHObjective::kPrefixScore) ? add_op("swap_poses", &op_swap_poses_prefix) : -1;

    // Large (expensive) ruin-recreate moves.
    const int op_lns_boundary_small_id = add_op("lns_boundary_small", &op_lns_boundary_small);
    const int op_lns_boundary_large_id = add_op("lns_boundary_large", &op_lns_boundary_large);
    const int op_lns_cluster_id = add_op("lns_cluster", &op_lns_cluster);
    const int op_lns_random_id = add_op("lns_random", &op_lns_random);

    // SA bursts (not part of the normal operator selection; triggered on stagnation).
    const int op_sa_intensify_id = add_op("sa_intensify", &op_sa_intensify);
    const int op_sa_diversify_id = add_op("sa_diversify", &op_sa_diversify);

    std::vector<MacroGroup> macros;
    macros.push_back(MacroGroup{"small", {op_move_translate_id, op_move_rotate_id, op_nudge_boundary_id}});
    macros.push_back(MacroGroup{"medium", {op_cluster_move_id, op_touch_best_of_id, op_reinsert_one_id}});
    if (op_swap_poses_id >= 0) {
        macros.back().op_ids.push_back(op_swap_poses_id);
    }
    macros.push_back(MacroGroup{
        "large",
        {op_lns_boundary_small_id, op_lns_boundary_large_id, op_lns_cluster_id, op_lns_random_id},
    });

    HHResult out;
    out.best_poses = best;
    out.init_s = init_s;
    out.best_s = best_s;
    out.init_score = init_score;
    out.best_score = best_score;
    out.init_obj = cur_obj;
    out.best_obj = best_obj;
    out.ops.resize(ops.size());
    for (size_t i = 0; i < ops.size(); ++i) {
        out.ops[i].name = ops[i].name;
        out.ops[i].weight = 1.0;
    }
    out.macros.resize(macros.size());
    for (size_t i = 0; i < macros.size(); ++i) {
        out.macros[i].name = macros[i].name;
    }

    int accepted_log = 0;
    int feasible_log = 0;

    const std::string prefix = opt.log_prefix.empty() ? std::string("[hh]") : opt.log_prefix;
    if (opt.log_every > 0) {
        std::lock_guard<std::mutex> lk(log_mutex());
        std::cerr << prefix << " start n=" << opt.n << " hh_iters=" << opt.hh_iters << " L=" << L
                  << " objective=" << ((opt.objective == HHObjective::kS) ? "s" : "score") << " init_obj=" << cur_obj
                  << " init_s=" << init_s << " init_score=" << init_score << " alns_rho=" << opt.alns_rho
                  << " sa_burst_after=" << ((opt.sa_burst_after > 0) ? opt.sa_burst_after : default_sa_burst_after(opt.hh_iters))
                  << " sa_burst_iters=" << opt.sa_burst_iters << "\n";
    }

    const int sa_burst_after = (opt.sa_burst_after > 0) ? opt.sa_burst_after : default_sa_burst_after(opt.hh_iters);
    const int sa_burst_cooldown = (opt.sa_burst_cooldown > 0) ? opt.sa_burst_cooldown : sa_burst_after;

    int last_best_t = 0;
    int last_sa_burst_t = std::numeric_limits<int>::min() / 2;

    for (int t = 0; t < opt.hh_iters; ++t) {
        const bool do_burst =
            ((t - last_best_t) >= sa_burst_after) && ((t - last_sa_burst_t) >= sa_burst_cooldown);

        int macro_id = -1;
        int op_id = -1;
        std::string macro_name;

        if (do_burst) {
            last_sa_burst_t = t;
            const bool diversify = ((t - last_best_t) >= 2 * sa_burst_after);
            op_id = diversify ? op_sa_diversify_id : op_sa_intensify_id;
            macro_name = "sa_burst";
        } else {
            macro_id = pick_ucb(out.macros, opt.ucb_c, t);
            if (macro_id < 0 || macro_id >= static_cast<int>(macros.size())) {
                break;
            }
            auto& mst = out.macros[static_cast<size_t>(macro_id)];
            mst.selected++;
            macro_name = mst.name;

            op_id = pick_operator_alns(out.ops, macros[static_cast<size_t>(macro_id)].op_ids, rng);
        }

        if (op_id < 0 || op_id >= static_cast<int>(ops.size())) {
            continue;
        }

        auto& st = out.ops[static_cast<size_t>(op_id)];
        st.selected++;
        out.attempted++;

        const double old_obj = cur_obj;
        const int pos = t % L;

        const auto t0 = std::chrono::steady_clock::now();
        Candidate cand = ops[static_cast<size_t>(op_id)].apply(tree_poly, cur, opt, rng, eps);
        const auto t1 = std::chrono::steady_clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        const bool feasible = (static_cast<int>(cand.poses.size()) == opt.n);

        double reward = 0.0;
        if (feasible) {
            reward = reward_delta_per_ms(old_obj, cand.obj, elapsed_ms, opt.reward_time_eps_ms);
        }

        if (macro_id >= 0 && macro_id < static_cast<int>(out.macros.size())) {
            update_running_mean(out.macros[static_cast<size_t>(macro_id)].mean_reward,
                                out.macros[static_cast<size_t>(macro_id)].selected,
                                reward);
        }

        if (!feasible) {
            update_running_mean(st.mean_reward, st.selected, 0.0);
            update_alns_weight(st.weight, 0.0, opt.alns_rho, opt.alns_min_weight);
            continue;
        }

        out.feasible++;
        st.feasible++;
        feasible_log++;

        bool accept = false;
        if (cand.obj <= old_obj + 1e-15) {
            accept = true;
        } else if (cand.obj <= hist[static_cast<size_t>(pos)] + 1e-15) {
            accept = true;
        }

        hist[static_cast<size_t>(pos)] = old_obj;

        if (!accept) {
            update_running_mean(st.mean_reward, st.selected, 0.0);
            update_alns_weight(st.weight, 0.0, opt.alns_rho, opt.alns_min_weight);
            continue;
        }

        out.accepted++;
        st.accepted++;
        accepted_log++;

        // Credit assignment: improvement per ms (0 for non-improvements).
        update_running_mean(st.mean_reward, st.selected, reward);
        update_alns_weight(st.weight, reward, opt.alns_rho, opt.alns_min_weight);

        cur = std::move(cand.poses);
        cur_s = cand.s;
        cur_score = cand.score;
        cur_obj = cand.obj;

        if (cur_obj < best_obj - 1e-15) {
            last_best_t = t;
            best = cur;
            best_s = cur_s;
            best_score = cur_score;
            best_obj = cur_obj;
            out.best_poses = best;
            out.best_s = best_s;
            out.best_score = best_score;
            out.best_obj = best_obj;
        }

        if (opt.log_every > 0 && (t % opt.log_every) == 0) {
            const double acc =
                static_cast<double>(accepted_log) / static_cast<double>(std::max(1, feasible_log));
            std::lock_guard<std::mutex> lk(log_mutex());
            std::cerr << prefix << " t=" << t << "/" << opt.hh_iters << " macro=" << macro_name
                      << " op=" << ops[static_cast<size_t>(op_id)].name << " cur_obj=" << cur_obj
                      << " best_obj=" << best_obj << " cur_s=" << cur_s << " best_s=" << best_s << " acc=" << acc
                      << " feasible=" << feasible_log << " accepted=" << accepted_log << "\n";
            accepted_log = 0;
            feasible_log = 0;
        }
    }

    if (opt.log_every > 0) {
        std::lock_guard<std::mutex> lk(log_mutex());
        std::cerr << prefix << " done best_obj=" << best_obj << " best_s=" << best_s << " best_score=" << best_score
                  << " attempted=" << out.attempted << " feasible=" << out.feasible << " accepted=" << out.accepted
                  << "\n";
    }

    return out;
}

}  // namespace santa2025
