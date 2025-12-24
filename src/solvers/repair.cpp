#include "solvers/repair.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <utility>

#include "geometry/collision.hpp"
#include "utils/submission_io.hpp"
#include "utils/wrap_utils.hpp"

namespace repair {
namespace {

double orient(const Point& a, const Point& b, const Point& c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

bool point_on_segment(const Point& a, const Point& b, const Point& p, double eps) {
    if (std::abs(orient(a, b, p)) > eps) {
        return false;
    }
    return (std::min(a.x, b.x) - eps <= p.x && p.x <= std::max(a.x, b.x) + eps) &&
           (std::min(a.y, b.y) - eps <= p.y && p.y <= std::max(a.y, b.y) + eps);
}

bool point_in_polygon_strict(const Point& pt, const Polygon& poly, double eps) {
    const double x = pt.x;
    const double y = pt.y;
    bool inside = false;

    size_t n = poly.size();
    if (n < 3) {
        return false;
    }

    for (size_t i = 0; i < n; ++i) {
        const Point& a = poly[i];
        const Point& b = poly[(i + 1) % n];
        if (point_on_segment(a, b, pt, eps)) {
            return false;
        }
    }

    size_t j = n - 1;
    for (size_t i = 0; i < n; ++i) {
        const auto& pi = poly[i];
        const auto& pj = poly[j];
        bool intersect =
            ((pi.y > y) != (pj.y > y)) &&
            (x < (pj.x - pi.x) * (y - pi.y) / ((pj.y - pi.y) + 1e-18) + pi.x);
        if (intersect) {
            inside = !inside;
        }
        j = i;
    }
    return inside;
}

Point normalize_dir(const Point& p) {
    const double norm = std::hypot(p.x, p.y);
    if (!(norm > 1e-12)) {
        return Point{0.0, 0.0};
    }
    return Point{p.x / norm, p.y / norm};
}

void apply_global_rotation(std::vector<TreePose>& poses, double ang_deg) {
    if (!(std::abs(ang_deg) > 1e-15)) {
        return;
    }
    for (auto& pose : poses) {
        Point p = rotate_point(Point{pose.x, pose.y}, ang_deg);
        pose.x = p.x;
        pose.y = p.y;
        pose.deg = wrap_deg(pose.deg + ang_deg);
    }
}

double side_for_quantized(const Polygon& base_poly,
                          const std::vector<TreePose>& poses,
                          int decimals) {
    std::vector<TreePose> q = quantize_poses_wrap_deg(poses, decimals);
    std::vector<BoundingBox> bbs = bounding_boxes_for_poses(base_poly, q);
    return side_from_extents(compute_extents(bbs));
}

bool push_along_dir(const Polygon& base_poly,
                    const std::vector<TreePose>& poses,
                    const std::vector<Polygon>& polys,
                    const std::vector<BoundingBox>& bbs,
                    int idx,
                    const TreePose& base_pose,
                    const Point& dir_in,
                    double max_step,
                    int bisect_iters,
                    TreePose& pose_out,
                    Polygon& poly_out,
                    BoundingBox& bb_out) {
    Point dir = normalize_dir(dir_in);
    if (!(max_step > 1e-12) || (std::abs(dir.x) < 1e-12 && std::abs(dir.y) < 1e-12)) {
        return false;
    }

    auto valid_at = [&](double delta,
                        TreePose& pose,
                        Polygon& poly,
                        BoundingBox& bb) -> bool {
        pose = base_pose;
        pose.x += dir.x * delta;
        pose.y += dir.y * delta;
        if (pose.x < -100.0 || pose.x > 100.0 || pose.y < -100.0 || pose.y > 100.0) {
            return false;
        }
        poly = transform_polygon(base_poly, pose);
        bb = bounding_box(poly);

        for (size_t j = 0; j < poses.size(); ++j) {
            if (static_cast<int>(j) == idx) {
                continue;
            }
            if (!aabb_overlap(bb, bbs[j])) {
                continue;
            }
            if (polygons_intersect(poly, polys[j])) {
                return false;
            }
        }
        return true;
    };

    TreePose best_pose;
    Polygon best_poly;
    BoundingBox best_bb;
    if (valid_at(max_step, best_pose, best_poly, best_bb)) {
        pose_out = best_pose;
        poly_out = std::move(best_poly);
        bb_out = best_bb;
        return true;
    }

    double lo = 0.0;
    double hi = max_step;
    bool found = false;
    for (int it = 0; it < bisect_iters; ++it) {
        double mid = 0.5 * (lo + hi);
        TreePose cand_pose;
        Polygon cand_poly;
        BoundingBox cand_bb;
        if (valid_at(mid, cand_pose, cand_poly, cand_bb)) {
            found = true;
            lo = mid;
            best_pose = cand_pose;
            best_poly = std::move(cand_poly);
            best_bb = cand_bb;
        } else {
            hi = mid;
        }
    }

    if (!found) {
        return false;
    }

    pose_out = best_pose;
    poly_out = std::move(best_poly);
    bb_out = best_bb;
    return true;
}

bool try_interlock_pair(const Polygon& base_poly,
                        std::vector<TreePose>& poses,
                        std::vector<Polygon>& polys,
                        std::vector<BoundingBox>& bbs,
                        int i,
                        int j,
                        const InterlockOptions& opt,
                        double& curr_side) {
    const int n = static_cast<int>(poses.size());
    if (n < 2 || i < 0 || j < 0 || i == j) {
        return false;
    }

    std::vector<double> rot_deltas;
    rot_deltas.push_back(0.0);
    if (opt.rot_steps > 0 && opt.rot_deg > 0.0) {
        for (int s = 1; s <= opt.rot_steps; ++s) {
            double ang = opt.rot_deg * (static_cast<double>(s) /
                                        static_cast<double>(opt.rot_steps));
            rot_deltas.push_back(ang);
            rot_deltas.push_back(-ang);
        }
    }

    auto try_one = [&](int mover, int target) -> bool {
        Point dir{poses[static_cast<size_t>(target)].x - poses[static_cast<size_t>(mover)].x,
                  poses[static_cast<size_t>(target)].y - poses[static_cast<size_t>(mover)].y};
        const double dist = std::hypot(dir.x, dir.y);
        if (!(dist > 1e-9)) {
            return false;
        }
        double max_step = opt.max_step_frac * std::max(1e-9, curr_side);
        max_step = std::min(max_step, dist);
        if (!(max_step > 1e-12)) {
            return false;
        }

        Extents ext_wo = compute_extents_excluding(bbs, mover);
        double best_side = curr_side;
        TreePose best_pose;
        Polygon best_poly;
        BoundingBox best_bb;
        bool improved = false;

        for (double ddeg : rot_deltas) {
            TreePose base_pose = poses[static_cast<size_t>(mover)];
            base_pose.deg = wrap_deg(base_pose.deg + ddeg);

            TreePose cand_pose;
            Polygon cand_poly;
            BoundingBox cand_bb;
            if (!push_along_dir(base_poly,
                                poses,
                                polys,
                                bbs,
                                mover,
                                base_pose,
                                dir,
                                max_step,
                                opt.bisect_iters,
                                cand_pose,
                                cand_poly,
                                cand_bb)) {
                continue;
            }

            Extents merged = merge_extents_bb(ext_wo, cand_bb);
            double side = side_from_extents(merged);
            if (side + 1e-15 < best_side) {
                best_side = side;
                best_pose = cand_pose;
                best_poly = std::move(cand_poly);
                best_bb = cand_bb;
                improved = true;
            }
        }

        if (improved) {
            poses[static_cast<size_t>(mover)] = best_pose;
            polys[static_cast<size_t>(mover)] = std::move(best_poly);
            bbs[static_cast<size_t>(mover)] = best_bb;
            curr_side = best_side;
            return true;
        }
        return false;
    };

    bool moved = false;
    moved |= try_one(i, j);
    moved |= try_one(j, i);
    return moved;
}

bool find_pocket_center(const std::vector<Point>& centers,
                        const std::vector<Polygon>& polys,
                        const std::vector<BoundingBox>& bbs,
                        int grid,
                        Point& out_center) {
    if (polys.empty() || grid < 2) {
        return false;
    }
    Extents e = compute_extents(bbs);
    if (!(e.max_x > e.min_x) || !(e.max_y > e.min_y)) {
        return false;
    }

    const double step_x = (e.max_x - e.min_x) / static_cast<double>(grid - 1);
    const double step_y = (e.max_y - e.min_y) / static_cast<double>(grid - 1);

    int best_occ = std::numeric_limits<int>::max();
    double best_dist2 = -1.0;
    Point best{0.0, 0.0};
    const double eps = 1e-12;

    for (int ix = 0; ix < grid; ++ix) {
        const double x = e.min_x + step_x * static_cast<double>(ix);
        for (int iy = 0; iy < grid; ++iy) {
            const double y = e.min_y + step_y * static_cast<double>(iy);
            Point pt{x, y};

            int occ = 0;
            bool inside = false;
            for (size_t i = 0; i < polys.size(); ++i) {
                const auto& bb = bbs[i];
                if (pt.x < bb.min_x || pt.x > bb.max_x || pt.y < bb.min_y || pt.y > bb.max_y) {
                    continue;
                }
                occ += 1;
                if (point_in_polygon_strict(pt, polys[i], eps)) {
                    inside = true;
                    break;
                }
            }
            if (inside) {
                continue;
            }

            double min_d2 = std::numeric_limits<double>::infinity();
            for (const auto& c : centers) {
                const double dx = pt.x - c.x;
                const double dy = pt.y - c.y;
                min_d2 = std::min(min_d2, dx * dx + dy * dy);
            }

            if (occ < best_occ || (occ == best_occ && min_d2 > best_dist2)) {
                best_occ = occ;
                best_dist2 = min_d2;
                best = pt;
            }
        }
    }

    if (best_occ == std::numeric_limits<int>::max()) {
        return false;
    }
    out_center = best;
    return true;
}

}  // namespace

Extents compute_extents(const std::vector<BoundingBox>& bbs) {
    Extents e;
    e.min_x = std::numeric_limits<double>::infinity();
    e.max_x = -std::numeric_limits<double>::infinity();
    e.min_y = std::numeric_limits<double>::infinity();
    e.max_y = -std::numeric_limits<double>::infinity();
    for (const auto& bb : bbs) {
        e.min_x = std::min(e.min_x, bb.min_x);
        e.max_x = std::max(e.max_x, bb.max_x);
        e.min_y = std::min(e.min_y, bb.min_y);
        e.max_y = std::max(e.max_y, bb.max_y);
    }
    return e;
}

Extents compute_extents_excluding(const std::vector<BoundingBox>& bbs, int skip) {
    Extents e;
    e.min_x = std::numeric_limits<double>::infinity();
    e.max_x = -std::numeric_limits<double>::infinity();
    e.min_y = std::numeric_limits<double>::infinity();
    e.max_y = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < bbs.size(); ++i) {
        if (static_cast<int>(i) == skip) {
            continue;
        }
        const auto& bb = bbs[i];
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

Extents merge_extents_bb(const Extents& e, const BoundingBox& bb) {
    Extents out = e;
    out.min_x = std::min(out.min_x, bb.min_x);
    out.max_x = std::max(out.max_x, bb.max_x);
    out.min_y = std::min(out.min_y, bb.min_y);
    out.max_y = std::max(out.max_y, bb.max_y);
    return out;
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

bool apply_interlock_passes(const Polygon& base_poly,
                            std::vector<TreePose>& poses,
                            const InterlockOptions& opt,
                            uint64_t seed) {
    if (opt.passes <= 0 || poses.size() < 2) {
        return false;
    }

    std::mt19937_64 rng(seed);
    std::vector<Polygon> polys = transformed_polygons(base_poly, poses);
    std::vector<BoundingBox> bbs;
    bbs.reserve(poses.size());
    for (const auto& poly : polys) {
        bbs.push_back(bounding_box(poly));
    }
    double curr_side = side_from_extents(compute_extents(bbs));
    bool improved = false;

    for (int pass = 0; pass < opt.passes; ++pass) {
        std::vector<int> pool = build_extreme_pool(bbs, std::max(1, opt.boundary_topk));
        if (pool.empty()) {
            break;
        }
        std::uniform_int_distribution<int> pick_pool(0, static_cast<int>(pool.size()) - 1);

        for (int attempt = 0; attempt < opt.attempts; ++attempt) {
            int i = pool[static_cast<size_t>(pick_pool(rng))];
            int j = -1;
            int k = -1;
            double dj = std::numeric_limits<double>::infinity();
            double dk = std::numeric_limits<double>::infinity();
            const TreePose& pi = poses[static_cast<size_t>(i)];
            for (size_t t = 0; t < poses.size(); ++t) {
                if (static_cast<int>(t) == i) {
                    continue;
                }
                const TreePose& pt = poses[t];
                double dist = std::hypot(pt.x - pi.x, pt.y - pi.y);
                if (dist < dj) {
                    dk = dj;
                    k = j;
                    dj = dist;
                    j = static_cast<int>(t);
                } else if (dist < dk) {
                    dk = dist;
                    k = static_cast<int>(t);
                }
            }
            if (j < 0) {
                continue;
            }
            bool moved = try_interlock_pair(base_poly, poses, polys, bbs, i, j, opt, curr_side);
            if (opt.group >= 3 && k >= 0) {
                moved |= try_interlock_pair(base_poly, poses, polys, bbs, i, k, opt, curr_side);
            }
            if (moved) {
                improved = true;
            }
        }
    }

    return improved;
}

bool pocket_repack(const Polygon& base_poly,
                   double radius,
                   std::vector<TreePose>& poses,
                   const PocketOptions& opt,
                   std::mt19937_64& rng) {
    const int n = static_cast<int>(poses.size());
    if (n <= 1 || opt.take <= 0) {
        return false;
    }

    poses = quantize_poses_wrap_deg(poses, opt.output_decimals);

    std::vector<Point> centers;
    centers.reserve(poses.size());
    std::vector<Polygon> polys;
    polys.reserve(poses.size());
    std::vector<BoundingBox> bbs;
    bbs.reserve(poses.size());
    for (const auto& p : poses) {
        centers.push_back(Point{p.x, p.y});
        Polygon poly = transform_polygon(base_poly, p);
        polys.push_back(std::move(poly));
        bbs.push_back(bounding_box(polys.back()));
    }

    Point pocket_center;
    if (!find_pocket_center(centers, polys, bbs, opt.grid, pocket_center)) {
        return false;
    }

    const double curr_side = side_from_extents(compute_extents(bbs));
    const double pocket_radius =
        std::max(opt.radius_frac * curr_side, 2.0 * radius);
    const double pocket_radius_sq = pocket_radius * pocket_radius;

    std::vector<std::pair<double, int>> ranked;
    ranked.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const auto& bb = bbs[static_cast<size_t>(i)];
        double dx = 0.0;
        if (pocket_center.x < bb.min_x) {
            dx = bb.min_x - pocket_center.x;
        } else if (pocket_center.x > bb.max_x) {
            dx = pocket_center.x - bb.max_x;
        }
        double dy = 0.0;
        if (pocket_center.y < bb.min_y) {
            dy = bb.min_y - pocket_center.y;
        } else if (pocket_center.y > bb.max_y) {
            dy = pocket_center.y - bb.max_y;
        }
        ranked.push_back({dx * dx + dy * dy, i});
    }
    std::sort(ranked.begin(),
              ranked.end(),
              [](const auto& a, const auto& b) {
                  if (a.first != b.first) {
                      return a.first < b.first;
                  }
                  return a.second < b.second;
              });

    int take = std::min(opt.take, n);
    std::vector<int> moved;
    moved.reserve(static_cast<size_t>(take));
    for (int k = 0; k < take; ++k) {
        moved.push_back(ranked[static_cast<size_t>(k)].second);
    }
    if (moved.empty()) {
        return false;
    }

    std::vector<char> is_moved(static_cast<size_t>(n), 0);
    for (int idx : moved) {
        is_moved[static_cast<size_t>(idx)] = 1;
    }

    std::vector<char> placed(static_cast<size_t>(n), 0);
    std::vector<Point> place_centers(static_cast<size_t>(n));
    std::vector<Polygon> place_polys(static_cast<size_t>(n));
    std::vector<BoundingBox> place_bbs(static_cast<size_t>(n));
    Extents e;
    e.min_x = std::numeric_limits<double>::infinity();
    e.max_x = -std::numeric_limits<double>::infinity();
    e.min_y = std::numeric_limits<double>::infinity();
    e.max_y = -std::numeric_limits<double>::infinity();
    int placed_count = 0;

    for (int i = 0; i < n; ++i) {
        if (is_moved[static_cast<size_t>(i)]) {
            continue;
        }
        const auto& pose = poses[static_cast<size_t>(i)];
        Polygon poly = transform_polygon(base_poly, pose);
        BoundingBox bb = bounding_box(poly);
        place_centers[static_cast<size_t>(i)] = Point{pose.x, pose.y};
        place_polys[static_cast<size_t>(i)] = std::move(poly);
        place_bbs[static_cast<size_t>(i)] = bb;
        placed[static_cast<size_t>(i)] = 1;
        if (placed_count == 0) {
            e.min_x = bb.min_x;
            e.max_x = bb.max_x;
            e.min_y = bb.min_y;
            e.max_y = bb.max_y;
        } else {
            e = merge_extents_bb(e, bb);
        }
        placed_count += 1;
    }

    std::vector<int> order = moved;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        const auto& pa = poses[static_cast<size_t>(a)];
        const auto& pb = poses[static_cast<size_t>(b)];
        const double da =
            (pa.x - pocket_center.x) * (pa.x - pocket_center.x) +
            (pa.y - pocket_center.y) * (pa.y - pocket_center.y);
        const double db =
            (pb.x - pocket_center.x) * (pb.x - pocket_center.x) +
            (pb.y - pocket_center.y) * (pb.y - pocket_center.y);
        return da < db;
    });

    const double min_x = std::max(-100.0, pocket_center.x - pocket_radius);
    const double max_x = std::min(100.0, pocket_center.x + pocket_radius);
    const double min_y = std::max(-100.0, pocket_center.y - pocket_radius);
    const double max_y = std::min(100.0, pocket_center.y + pocket_radius);
    if (!(max_x > min_x) || !(max_y > min_y)) {
        return false;
    }

    std::uniform_real_distribution<double> ux(min_x, max_x);
    std::uniform_real_distribution<double> uy(min_y, max_y);
    std::uniform_real_distribution<double> udeg(-opt.rot_deg, opt.rot_deg);

    const double thr = 2.0 * radius + 1e-9;
    const double limit_sq = thr * thr;

    for (int idx : order) {
        const TreePose base = poses[static_cast<size_t>(idx)];
        bool found = false;
        double best_side = std::numeric_limits<double>::infinity();
        TreePose best_pose = base;
        Polygon best_poly;
        BoundingBox best_bb{};

        for (int attempt = 0; attempt < opt.attempts; ++attempt) {
            TreePose cand = base;
            if (attempt > 0) {
                cand.x = ux(rng);
                cand.y = uy(rng);
                cand.deg = wrap_deg(cand.deg + udeg(rng));
            }
            cand = quantize_pose_wrap_deg(cand, opt.output_decimals);

            if (cand.x < -100.0 || cand.x > 100.0 || cand.y < -100.0 || cand.y > 100.0) {
                continue;
            }
            double dxp = cand.x - pocket_center.x;
            double dyp = cand.y - pocket_center.y;
            if (dxp * dxp + dyp * dyp > pocket_radius_sq) {
                continue;
            }

            Polygon cand_poly = transform_polygon(base_poly, cand);
            BoundingBox cand_bb = bounding_box(cand_poly);

            bool collide = false;
            for (int j = 0; j < n; ++j) {
                if (!placed[static_cast<size_t>(j)]) {
                    continue;
                }
                const double dx = cand.x - place_centers[static_cast<size_t>(j)].x;
                const double dy = cand.y - place_centers[static_cast<size_t>(j)].y;
                if (dx * dx + dy * dy > limit_sq) {
                    continue;
                }
                if (!aabb_overlap(cand_bb, place_bbs[static_cast<size_t>(j)])) {
                    continue;
                }
                if (polygons_intersect(cand_poly, place_polys[static_cast<size_t>(j)])) {
                    collide = true;
                    break;
                }
            }
            if (collide) {
                continue;
            }

            Extents e_new;
            if (placed_count == 0) {
                e_new.min_x = cand_bb.min_x;
                e_new.max_x = cand_bb.max_x;
                e_new.min_y = cand_bb.min_y;
                e_new.max_y = cand_bb.max_y;
            } else {
                e_new = merge_extents_bb(e, cand_bb);
            }
            const double side_new = side_from_extents(e_new);
            if (!found || side_new + 1e-15 < best_side) {
                found = true;
                best_side = side_new;
                best_pose = cand;
                best_poly = std::move(cand_poly);
                best_bb = cand_bb;
            }
        }

        if (!found) {
            return false;
        }

        poses[static_cast<size_t>(idx)] = best_pose;
        place_centers[static_cast<size_t>(idx)] = Point{best_pose.x, best_pose.y};
        place_polys[static_cast<size_t>(idx)] = std::move(best_poly);
        place_bbs[static_cast<size_t>(idx)] = best_bb;
        placed[static_cast<size_t>(idx)] = 1;
        if (placed_count == 0) {
            e.min_x = best_bb.min_x;
            e.max_x = best_bb.max_x;
            e.min_y = best_bb.min_y;
            e.max_y = best_bb.max_y;
        } else {
            e = merge_extents_bb(e, best_bb);
        }
        placed_count += 1;
    }

    return true;
}

bool micro_refine_rigid_rotation(const Polygon& base_poly,
                                 double radius,
                                 std::vector<TreePose>& poses,
                                 int micro_rigid_steps,
                                 double micro_rigid_step_deg,
                                 int output_decimals) {
    if (micro_rigid_steps <= 0 || !(micro_rigid_step_deg > 0.0)) {
        return false;
    }

    double best_side = side_for_quantized(base_poly, poses, output_decimals);
    double best_ang = 0.0;

    for (int step = -micro_rigid_steps; step <= micro_rigid_steps; ++step) {
        if (step == 0) {
            continue;
        }
        const double ang = static_cast<double>(step) * micro_rigid_step_deg;
        std::vector<TreePose> cand = poses;
        apply_global_rotation(cand, ang);
        std::vector<TreePose> cand_q = quantize_poses_wrap_deg(cand, output_decimals);
        if (any_overlap(base_poly, cand_q, radius)) {
            continue;
        }
        std::vector<BoundingBox> bbs = bounding_boxes_for_poses(base_poly, cand_q);
        double side = side_from_extents(compute_extents(bbs));
        if (side + 1e-15 < best_side) {
            best_side = side;
            best_ang = ang;
        }
    }

    if (std::abs(best_ang) > 1e-15) {
        apply_global_rotation(poses, best_ang);
        return true;
    }
    return false;
}

}  // namespace repair
