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

struct HoleInfo {
    bool ok = false;
    double cx = 0.0;
    double cy = 0.0;
    int cells = 0;
};

double sq_dist_point_bb(double x, double y, const BoundingBox& bb) {
    const double dx = (x < bb.min_x) ? (bb.min_x - x) : ((x > bb.max_x) ? (x - bb.max_x) : 0.0);
    const double dy = (y < bb.min_y) ? (bb.min_y - y) : ((y > bb.max_y) ? (y - bb.max_y) : 0.0);
    return dx * dx + dy * dy;
}

HoleInfo find_largest_empty_hole_grid(
    const std::vector<BoundingBox>& world_bbs,
    const BoundsMetrics& m,
    int grid_n,
    double eps
) {
    HoleInfo out;
    if (grid_n < 4) {
        return out;
    }
    if (!(std::isfinite(m.min_x) && std::isfinite(m.max_x) && std::isfinite(m.min_y) && std::isfinite(m.max_y))) {
        return out;
    }

    const double x0 = m.min_x;
    const double x1 = m.max_x;
    const double y0 = m.min_y;
    const double y1 = m.max_y;
    if (!(x1 > x0 + eps) || !(y1 > y0 + eps)) {
        return out;
    }

    const double dx = (x1 - x0) / static_cast<double>(grid_n);
    const double dy = (y1 - y0) / static_cast<double>(grid_n);
    if (!(dx > eps) || !(dy > eps)) {
        return out;
    }

    const int n = grid_n;
    std::vector<char> occ(static_cast<size_t>(n * n), 0);
    std::vector<char> seen(static_cast<size_t>(n * n), 0);

    auto clamp_i = [&](int v) { return std::clamp(v, 0, n - 1); };
    auto idx = [&](int ix, int iy) { return static_cast<size_t>(iy * n + ix); };

    for (const auto& bb : world_bbs) {
        const int ix0 = clamp_i(static_cast<int>(std::floor((bb.min_x - x0) / dx)));
        const int ix1 = clamp_i(static_cast<int>(std::floor((bb.max_x - x0) / dx)));
        const int iy0 = clamp_i(static_cast<int>(std::floor((bb.min_y - y0) / dy)));
        const int iy1 = clamp_i(static_cast<int>(std::floor((bb.max_y - y0) / dy)));
        for (int iy = iy0; iy <= iy1; ++iy) {
            for (int ix = ix0; ix <= ix1; ++ix) {
                occ[idx(ix, iy)] = 1;
            }
        }
    }

    int best_cells = 0;
    double best_sum_x = 0.0;
    double best_sum_y = 0.0;

    std::vector<int> stack;
    stack.reserve(static_cast<size_t>(n * n));

    for (int iy = 0; iy < n; ++iy) {
        for (int ix = 0; ix < n; ++ix) {
            const size_t id = idx(ix, iy);
            if (occ[id] || seen[id]) {
                continue;
            }

            stack.clear();
            stack.push_back(iy * n + ix);
            seen[id] = 1;

            int cells = 0;
            double sum_x = 0.0;
            double sum_y = 0.0;

            while (!stack.empty()) {
                const int cur = stack.back();
                stack.pop_back();
                const int cx_i = cur % n;
                const int cy_i = cur / n;

                ++cells;
                sum_x += x0 + (static_cast<double>(cx_i) + 0.5) * dx;
                sum_y += y0 + (static_cast<double>(cy_i) + 0.5) * dy;

                const int nx[4] = {cx_i - 1, cx_i + 1, cx_i, cx_i};
                const int ny[4] = {cy_i, cy_i, cy_i - 1, cy_i + 1};
                for (int k = 0; k < 4; ++k) {
                    const int x2 = nx[k];
                    const int y2 = ny[k];
                    if (x2 < 0 || x2 >= n || y2 < 0 || y2 >= n) {
                        continue;
                    }
                    const size_t id2 = idx(x2, y2);
                    if (occ[id2] || seen[id2]) {
                        continue;
                    }
                    seen[id2] = 1;
                    stack.push_back(y2 * n + x2);
                }
            }

            if (cells > best_cells) {
                best_cells = cells;
                best_sum_x = sum_x;
                best_sum_y = sum_y;
            }
        }
    }

    if (best_cells <= 0) {
        return out;
    }

    out.ok = true;
    out.cells = best_cells;
    out.cx = best_sum_x / static_cast<double>(best_cells);
    out.cy = best_sum_y / static_cast<double>(best_cells);
    return out;
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
    double remove_frac,
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

    const HoleInfo hole = (opt.destroy_mode == LNSDestroyMode::kGap || opt.gap_try_hole_center)
                              ? find_largest_empty_hole_grid(world, m, opt.gap_grid, eps)
                              : HoleInfo{};

    int k = static_cast<int>(std::ceil(remove_frac * static_cast<double>(n)));
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
    case LNSDestroyMode::kGap: {
        if (hole.ok) {
            std::vector<std::pair<double, int>> near;
            near.reserve(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) {
                near.emplace_back(sq_dist_point_bb(hole.cx, hole.cy, world[static_cast<size_t>(i)]), i);
            }
            std::sort(near.begin(), near.end(), [](const auto& a, const auto& b) {
                if (a.first != b.first) {
                    return a.first < b.first;
                }
                return a.second < b.second;
            });

            size_t pos = 0;
            while (static_cast<int>(removed.size()) < k && guard++ < 20 * n) {
                if (!boundary.empty() && unif01(rng) < opt.boundary_prob) {
                    take(boundary[static_cast<size_t>(pick_bnd(rng))]);
                    continue;
                }
                while (pos < near.size() && picked[static_cast<size_t>(near[pos].second)]) {
                    ++pos;
                }
                if (pos < near.size()) {
                    take(near[pos].second);
                    ++pos;
                    continue;
                }
                take(pick_any(rng));
            }
            break;
        }

        // Fallback: behave like mix random/boundary if hole detection fails.
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

    bool bounds_init = false;
    double cur_min_x = 0.0;
    double cur_min_y = 0.0;
    double cur_max_x = 0.0;
    double cur_max_y = 0.0;
    for (int j = 0; j < n; ++j) {
        if (picked[static_cast<size_t>(j)]) {
            continue;
        }
        const BoundingBox local_bb = rotated_bbox_cached(tree_poly, poses[static_cast<size_t>(j)].deg, bbox_cache);
        const BoundingBox bb = bbox_for_pose(local_bb, poses[static_cast<size_t>(j)]);
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
        BoundingBox best_world_bb{};
        double best_s = std::numeric_limits<double>::infinity();

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
                const BoundingBox cand_world_bb = bbox_for_pose(local_bb, cand);

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

                any_angle = true;
                if (new_s < best_s - 1e-12 ||
                    (std::abs(new_s - best_s) <= 1e-12 &&
                     (cand.y < best.y - 1e-12 || (std::abs(cand.y - best.y) <= 1e-12 && cand.x < best.x)))) {
                    best = cand;
                    best_world_bb = cand_world_bb;
                    best_s = new_s;
                }
            };

            // Bottom-left corner.
            consider(Pose{x_low, y_low, ang});

            if (hole.ok && opt.gap_try_hole_center) {
                const double bb_cx = 0.5 * (local_bb.min_x + local_bb.max_x);
                const double bb_cy = 0.5 * (local_bb.min_y + local_bb.max_y);
                consider(Pose{hole.cx - bb_cx, hole.cy - bb_cy, ang});
            }

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

        if (!bounds_init) {
            bounds_init = true;
            cur_min_x = best_world_bb.min_x;
            cur_min_y = best_world_bb.min_y;
            cur_max_x = best_world_bb.max_x;
            cur_max_y = best_world_bb.max_y;
        } else {
            cur_min_x = std::min(cur_min_x, best_world_bb.min_x);
            cur_min_y = std::min(cur_min_y, best_world_bb.min_y);
            cur_max_x = std::max(cur_max_x, best_world_bb.max_x);
            cur_max_y = std::max(cur_max_y, best_world_bb.max_y);
        }
    }

    // Optional local search on the removed subset only (others fixed).
    if (opt.repair_sa_iters > 0 && !removed.empty()) {
        std::vector<BoundingBox> world_bbs(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            const Pose& p = poses[static_cast<size_t>(i)];
            const BoundingBox local = rotated_bbox_cached(tree_poly, p.deg, bbox_cache);
            world_bbs[static_cast<size_t>(i)] = bbox_for_pose(local, p);
        }
        double cur_s = bounds_metrics(world_bbs).s;

        auto s_replace_one = [&](int id, const BoundingBox& bb_new) -> double {
            BoundsMetrics mm;
            mm.min_x = std::numeric_limits<double>::infinity();
            mm.min_y = std::numeric_limits<double>::infinity();
            mm.max_x = -std::numeric_limits<double>::infinity();
            mm.max_y = -std::numeric_limits<double>::infinity();
            for (int i = 0; i < n; ++i) {
                const auto& bb = (i == id) ? bb_new : world_bbs[static_cast<size_t>(i)];
                mm.min_x = std::min(mm.min_x, bb.min_x);
                mm.min_y = std::min(mm.min_y, bb.min_y);
                mm.max_x = std::max(mm.max_x, bb.max_x);
                mm.max_y = std::max(mm.max_y, bb.max_y);
            }
            mm.width = mm.max_x - mm.min_x;
            mm.height = mm.max_y - mm.min_y;
            mm.s = std::max(mm.width, mm.height);
            return mm.s;
        };

        auto temp_at = [&](int iter) -> double {
            if (opt.repair_sa_iters <= 1) {
                return opt.repair_sa_t1;
            }
            const double t = static_cast<double>(iter) / static_cast<double>(opt.repair_sa_iters - 1);
            const double ratio = opt.repair_sa_t1 / opt.repair_sa_t0;
            return opt.repair_sa_t0 * std::pow(ratio, t);
        };

        std::uniform_int_distribution<int> pick_removed(0, static_cast<int>(removed.size()) - 1);
        std::uniform_int_distribution<int> pick_any_id(0, n - 1);

        const int best_of = std::max(1, opt.repair_sa_best_of);
        const int anchor_samples = std::max(1, opt.repair_sa_anchor_samples);

        for (int it = 0; it < opt.repair_sa_iters; ++it) {
            const int id = removed[static_cast<size_t>(pick_removed(rng))];
            const Pose old = poses[static_cast<size_t>(id)];
            const BoundingBox old_world = world_bbs[static_cast<size_t>(id)];

            index.remove(id);

            Pose best_pose = old;
            BoundingBox best_world = old_world;
            double best_s = cur_s;
            bool any = false;

            const double T = temp_at(it);

            for (int rep = 0; rep < best_of; ++rep) {
                const auto angs = angle_candidates(opt, id);
                if (angs.empty()) {
                    continue;
                }
                std::uniform_int_distribution<int> pick_ang(0, static_cast<int>(angs.size()) - 1);
                const double ang = angs[static_cast<size_t>(pick_ang(rng))];

                const BoundingBox local_bb = rotated_bbox_cached(tree_poly, ang, bbox_cache);
                if (local_bb.max_x - local_bb.min_x > side + eps || local_bb.max_y - local_bb.min_y > side + eps) {
                    continue;
                }

                auto propose = [&](const Pose& start) -> void {
                    Pose cand = start;
                    cand.deg = ang;
                    cand = slide_down_left(cand, local_bb, index, side, opt.safety_eps, opt.slide_iters, eps);
                    if (!std::isfinite(cand.x) || !std::isfinite(cand.y)) {
                        return;
                    }
                    const BoundingBox cand_world = bbox_for_pose(local_bb, cand);
                    const double s_new = s_replace_one(id, cand_world);

                    if (!any || s_new < best_s - 1e-12 ||
                        (std::abs(s_new - best_s) <= 1e-12 &&
                         (cand.y < best_pose.y - 1e-12 ||
                          (std::abs(cand.y - best_pose.y) <= 1e-12 && cand.x < best_pose.x)))) {
                        any = true;
                        best_pose = cand;
                        best_world = cand_world;
                        best_s = s_new;
                    }
                };

                // Occasionally bias proposals to the detected hole center (if available).
                if (hole.ok && opt.gap_try_hole_center && unif01(rng) < 0.25) {
                    const double bb_cx = 0.5 * (local_bb.min_x + local_bb.max_x);
                    const double bb_cy = 0.5 * (local_bb.min_y + local_bb.max_y);
                    propose(Pose{hole.cx - bb_cx, hole.cy - bb_cy, ang});
                }

                // Touch moves: sample a few random anchors.
                for (int a = 0; a < anchor_samples; ++a) {
                    int anchor_id = pick_any_id(rng);
                    if (anchor_id == id) {
                        anchor_id = (anchor_id + 1) % n;
                    }
                    const Pose& anchor = poses[static_cast<size_t>(anchor_id)];
                    const double delta = ang - anchor.deg;
                    const auto& offsets = offsets_for(delta);
                    if (offsets.empty()) {
                        continue;
                    }
                    std::uniform_int_distribution<int> pick_off(0, static_cast<int>(offsets.size()) - 1);
                    const auto& v = offsets[static_cast<size_t>(pick_off(rng))];
                    const double nrm = std::hypot(v.x, v.y);
                    if (!(nrm > 0.0)) {
                        continue;
                    }
                    const Point u{v.x / nrm, v.y / nrm};
                    const Point v_out{v.x + opt.gap * u.x, v.y + opt.gap * u.y};
                    const Point t_world = rotate_point(v_out, anchor.deg);
                    propose(Pose{anchor.x + t_world.x, anchor.y + t_world.y, ang});
                }
            }

            bool accept = false;
            if (any) {
                const double delta = best_s - cur_s;
                if (delta <= -1e-12) {
                    accept = true;
                } else if (T > 0.0) {
                    const double p = std::exp(-delta / T);
                    accept = (unif01(rng) < p);
                }
            }

            if (accept) {
                poses[static_cast<size_t>(id)] = best_pose;
                world_bbs[static_cast<size_t>(id)] = best_world;
                index.set_pose(id, best_pose);
                cur_s = best_s;
            } else {
                index.set_pose(id, old);
            }
        }
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
    if (!(opt.remove_frac_max > 0.0 && opt.remove_frac_max <= 1.0)) {
        throw std::invalid_argument("lns_shrink_wrap: remove_frac_max must be in (0,1]");
    }
    if (!(opt.remove_frac_max + 1e-12 >= opt.remove_frac)) {
        throw std::invalid_argument("lns_shrink_wrap: remove_frac_max must be >= remove_frac");
    }
    if (!(opt.remove_frac_growth >= 1.0)) {
        throw std::invalid_argument("lns_shrink_wrap: remove_frac_growth must be >= 1");
    }
    if (opt.remove_frac_growth_every <= 0) {
        throw std::invalid_argument("lns_shrink_wrap: remove_frac_growth_every must be > 0");
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
    if (opt.gap_grid < 4) {
        throw std::invalid_argument("lns_shrink_wrap: gap_grid must be >= 4");
    }
    if (opt.repair_sa_iters < 0) {
        throw std::invalid_argument("lns_shrink_wrap: repair_sa_iters must be >= 0");
    }
    if (opt.repair_sa_best_of <= 0) {
        throw std::invalid_argument("lns_shrink_wrap: repair_sa_best_of must be >= 1");
    }
    if (opt.repair_sa_anchor_samples <= 0) {
        throw std::invalid_argument("lns_shrink_wrap: repair_sa_anchor_samples must be >= 1");
    }
    if (opt.repair_sa_iters > 0 && (!(opt.repair_sa_t0 > 0.0) || !(opt.repair_sa_t1 > 0.0))) {
        throw std::invalid_argument("lns_shrink_wrap: repair_sa_t0/repair_sa_t1 must be > 0 when repair_sa_iters>0");
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
                  << " remove_frac_max=" << opt.remove_frac_max << " remove_frac_growth=" << opt.remove_frac_growth
                  << " remove_frac_growth_every=" << opt.remove_frac_growth_every
                  << " boundary_prob=" << opt.boundary_prob << " destroy_mode=" << static_cast<int>(opt.destroy_mode)
                  << " gap_grid=" << opt.gap_grid << " repair_sa_iters=" << opt.repair_sa_iters
                  << " shrink_factor=" << opt.shrink_factor
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

            double attempt_remove_frac = opt.remove_frac;
            if (opt.remove_frac_growth > 1.0 && opt.remove_frac_max > opt.remove_frac + 1e-12) {
                const int steps = attempt / opt.remove_frac_growth_every;
                attempt_remove_frac *= std::pow(opt.remove_frac_growth, static_cast<double>(steps));
                attempt_remove_frac = std::min(attempt_remove_frac, opt.remove_frac_max);
            }

            if (!destroy_and_repair(tree_poly,
                                    cand,
                                    index,
                                    bbox_cache,
                                    verts_cache,
                                    opt,
                                    attempt_remove_frac,
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
