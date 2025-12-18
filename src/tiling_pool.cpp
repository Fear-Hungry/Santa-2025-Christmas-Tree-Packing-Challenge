#include "tiling_pool.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "collision.hpp"
#include "prefix_prune.hpp"
#include "submission_io.hpp"
#include "wrap_utils.hpp"

namespace {

constexpr double kSqrt3 = 1.732050807568877293527446341505872366942805254;

struct Candidate {
    TreePose pose;
    double key1;
    double key2;
};

std::pair<Point, Point> hex_basis(double spacing, double angle_deg) {
    Point u{spacing, 0.0};
    Point v{0.5 * spacing, 0.5 * spacing * kSqrt3};
    return {rotate_point(u, angle_deg), rotate_point(v, angle_deg)};
}

std::pair<Point, Point> bravais_basis(double u_len,
                                      double v_ratio,
                                      double theta_deg,
                                      double angle_deg) {
    const double v_len = u_len * v_ratio;
    const double th = theta_deg * 3.14159265358979323846 / 180.0;
    Point u0{u_len, 0.0};
    Point v0{v_len * std::cos(th), v_len * std::sin(th)};
    return {rotate_point(u0, angle_deg), rotate_point(v0, angle_deg)};
}

std::pair<Point, Point> lattice_basis(double spacing,
                                      double angle_deg,
                                      const Pattern& pattern) {
    if (std::abs(pattern.lattice_v_ratio - 1.0) <= 1e-12 &&
        std::abs(pattern.lattice_theta_deg - 60.0) <= 1e-12) {
        return hex_basis(spacing, angle_deg);
    }
    return bravais_basis(spacing,
                         pattern.lattice_v_ratio,
                         pattern.lattice_theta_deg,
                         angle_deg);
}

TreePose make_pose(int i,
                   int j,
                   const MotifPoint& mp,
                   const Point& u,
                   const Point& v,
                   double angle_deg,
                   double shift_a,
                   double shift_b) {
    const double cu = static_cast<double>(i) + mp.a - shift_a;
    const double cv = static_cast<double>(j) + mp.b - shift_b;
    const double x = cu * u.x + cv * v.x;
    const double y = cu * u.y + cv * v.y;
    const double deg = wrap_deg(mp.deg + angle_deg);
    return TreePose{x, y, deg};
}

bool periodic_safe(const Polygon& base_poly,
                   const Pattern& pattern,
                   double radius,
                   double spacing,
                   double eps) {
    if (!(spacing > 0.0)) {
        return false;
    }
    const int k = static_cast<int>(pattern.motif.size());
    if (k <= 0) {
        return false;
    }

    auto [u, v] = lattice_basis(spacing, 0.0, pattern);

    const double limit = 2.0 * radius + eps;
    const double limit_sq = limit * limit;

    const double u_len = std::hypot(u.x, u.y);
    const double v_len = std::hypot(v.x, v.y);
    const double search = limit + u_len + v_len + 1e-9;

    auto neighbor_offsets = [&](double max_dist) -> std::vector<std::pair<int, int>> {
        const double det = u.x * v.y - u.y * v.x;
        if (!(std::abs(det) > 1e-12)) {
            return {};
        }
        const double inv00 = v.y / det;
        const double inv01 = -v.x / det;
        const double inv10 = -u.y / det;
        const double inv11 = u.x / det;

        int bi = static_cast<int>(std::ceil((std::abs(inv00) + std::abs(inv01)) * max_dist)) + 2;
        int bj = static_cast<int>(std::ceil((std::abs(inv10) + std::abs(inv11)) * max_dist)) + 2;
        bi = std::max(1, std::min(200, bi));
        bj = std::max(1, std::min(200, bj));

        std::vector<std::pair<int, int>> out;
        out.reserve(static_cast<size_t>((2 * bi + 1) * (2 * bj + 1) - 1));
        for (int i = -bi; i <= bi; ++i) {
            for (int j = -bj; j <= bj; ++j) {
                if (i == 0 && j == 0) {
                    continue;
                }
                const double tx = static_cast<double>(i) * u.x + static_cast<double>(j) * v.x;
                const double ty = static_cast<double>(i) * u.y + static_cast<double>(j) * v.y;
                if (std::hypot(tx, ty) > max_dist + 1e-12) {
                    continue;
                }
                out.push_back({i, j});
            }
        }
        std::sort(out.begin(), out.end());
        return out;
    };

    std::vector<std::pair<int, int>> neigh = neighbor_offsets(search);
    if (neigh.empty()) {
        return false;
    }

    std::vector<TreePose> origin_poses;
    origin_poses.reserve(static_cast<size_t>(k));
    std::vector<Polygon> origin_polys;
    origin_polys.reserve(static_cast<size_t>(k));
    for (const auto& mp : pattern.motif) {
        TreePose p =
            make_pose(0, 0, mp, u, v, 0.0, pattern.shift_a, pattern.shift_b);
        origin_poses.push_back(p);
        origin_polys.push_back(transform_polygon(base_poly, p));
    }

    for (int a = 0; a < k; ++a) {
        for (int b = a + 1; b < k; ++b) {
            if (polygons_intersect(origin_polys[a], origin_polys[b])) {
                return false;
            }
        }
    }

    for (const auto& ij : neigh) {
        const int i = ij.first;
        const int j = ij.second;
        for (int qi = 0; qi < k; ++qi) {
            const auto& mp = pattern.motif[static_cast<size_t>(qi)];
            TreePose q =
                make_pose(i, j, mp, u, v, 0.0, pattern.shift_a, pattern.shift_b);
            Polygon q_poly = transform_polygon(base_poly, q);
            for (int pi = 0; pi < k; ++pi) {
                double dx = q.x - origin_poses[static_cast<size_t>(pi)].x;
                double dy = q.y - origin_poses[static_cast<size_t>(pi)].y;
                if (dx * dx + dy * dy > limit_sq) {
                    continue;
                }
                if (polygons_intersect(origin_polys[static_cast<size_t>(pi)], q_poly)) {
                    return false;
                }
            }
        }
    }

    return true;
}

}  // namespace

std::vector<TreePose> generate_ordered_tiling(int n,
                                              double spacing,
                                              double angle_deg,
                                              const Pattern& pattern) {
    if (n <= 0) {
        return {};
    }
    const int k = static_cast<int>(pattern.motif.size());
    if (k <= 0) {
        throw std::runtime_error("Pattern.motif vazio.");
    }

    auto [u, v] = lattice_basis(spacing, angle_deg, pattern);

    int m = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(n) /
                                                 static_cast<double>(k)))) +
            12;
    while (static_cast<int>((2LL * m + 1) * (2LL * m + 1) * k) < n) {
        ++m;
    }

    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<size_t>((2LL * m + 1) * (2LL * m + 1) * k));

    for (int i = -m; i <= m; ++i) {
        for (int j = -m; j <= m; ++j) {
            for (const auto& mp : pattern.motif) {
                TreePose pose = make_pose(i,
                                          j,
                                          mp,
                                          u,
                                          v,
                                          angle_deg,
                                          pattern.shift_a,
                                          pattern.shift_b);
                double key1 = std::max(std::abs(pose.x), std::abs(pose.y));
                double key2 = std::hypot(pose.x, pose.y);
                candidates.push_back({pose, key1, key2});
            }
        }
    }

    std::sort(candidates.begin(),
              candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  if (a.key1 != b.key1) {
                      return a.key1 < b.key1;
                  }
                  if (a.key2 != b.key2) {
                      return a.key2 < b.key2;
                  }
                  if (a.pose.x != b.pose.x) {
                      return a.pose.x < b.pose.x;
                  }
                  if (a.pose.y != b.pose.y) {
                      return a.pose.y < b.pose.y;
                  }
                  return a.pose.deg < b.pose.deg;
              });

    std::vector<TreePose> out;
    out.reserve(static_cast<size_t>(n));
    for (const auto& c : candidates) {
        out.push_back(c.pose);
        if (static_cast<int>(out.size()) >= n) {
            break;
        }
    }
    return out;
}

std::vector<TreePose> generate_windowed_tiling(int n,
                                               double spacing,
                                               double angle_deg,
                                               const Pattern& pattern,
                                               const Polygon& base_poly,
                                               int window_radius,
                                               int eval_n) {
    if (n <= 0) {
        return {};
    }
    const int k = static_cast<int>(pattern.motif.size());
    if (k <= 0) {
        throw std::runtime_error("Pattern.motif vazio.");
    }
    if (window_radius <= 0) {
        return generate_ordered_tiling(n, spacing, angle_deg, pattern);
    }

    const int eval_n_clamped = std::max(1, std::min(n, eval_n));
    const int cells_needed = static_cast<int>(
        std::ceil(static_cast<double>(n) / static_cast<double>(k)));
    int w = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(cells_needed))));
    int h = static_cast<int>(std::ceil(static_cast<double>(cells_needed) / w));
    w = std::max(1, w);
    h = std::max(1, h);

    auto [u, v] = lattice_basis(spacing, angle_deg, pattern);

    auto build_pool = [&](int center_i, int center_j) {
        const int start_i = center_i - (w / 2);
        const int start_j = center_j - (h / 2);
        std::vector<Candidate> candidates;
        candidates.reserve(static_cast<size_t>(w * h * k));
        for (int i = start_i; i < start_i + w; ++i) {
            for (int j = start_j; j < start_j + h; ++j) {
                for (const auto& mp : pattern.motif) {
                    TreePose pose = make_pose(i,
                                              j,
                                              mp,
                                              u,
                                              v,
                                              angle_deg,
                                              pattern.shift_a,
                                              pattern.shift_b);
                    double key1 = std::max(std::abs(pose.x), std::abs(pose.y));
                    double key2 = std::hypot(pose.x, pose.y);
                    candidates.push_back({pose, key1, key2});
                }
            }
        }
        std::sort(candidates.begin(),
                  candidates.end(),
                  [](const Candidate& a, const Candidate& b) {
                      if (a.key1 != b.key1) {
                          return a.key1 < b.key1;
                      }
                      if (a.key2 != b.key2) {
                          return a.key2 < b.key2;
                      }
                      if (a.pose.x != b.pose.x) {
                          return a.pose.x < b.pose.x;
                      }
                      if (a.pose.y != b.pose.y) {
                          return a.pose.y < b.pose.y;
                      }
                      return a.pose.deg < b.pose.deg;
                  });

        std::vector<TreePose> pool;
        pool.reserve(static_cast<size_t>(n));
        for (const auto& c : candidates) {
            pool.push_back(c.pose);
            if (static_cast<int>(pool.size()) >= n) {
                break;
            }
        }
        return pool;
    };

    double best_total = std::numeric_limits<double>::infinity();
    std::vector<TreePose> best_pool;

    for (int ci = -window_radius; ci <= window_radius; ++ci) {
        for (int cj = -window_radius; cj <= window_radius; ++cj) {
            std::vector<TreePose> pool = build_pool(ci, cj);
            if (static_cast<int>(pool.size()) < eval_n_clamped) {
                continue;
            }
            std::vector<TreePose> prefix(pool.begin(), pool.begin() + eval_n_clamped);
            std::vector<double> prefix_side_by_n =
                prefix_sides_from_bbs(bounding_boxes_for_poses(base_poly, prefix));
            double total = total_score_from_sides(prefix_side_by_n, eval_n_clamped);
            if (total < best_total) {
                best_total = total;
                best_pool = std::move(pool);
            }
        }
    }

    if (best_pool.empty()) {
        return generate_ordered_tiling(n, spacing, angle_deg, pattern);
    }
    return best_pool;
}

double find_min_safe_spacing(const Polygon& base_poly,
                             const Pattern& pattern,
                             double radius,
                             double eps,
                             double spacing_hint) {
    double hi = std::max(spacing_hint, 1e-6);
    for (int it = 0; it < 80 && !periodic_safe(base_poly, pattern, radius, hi, eps); ++it) {
        hi *= 1.25;
    }
    if (!periodic_safe(base_poly, pattern, radius, hi, eps)) {
        return std::numeric_limits<double>::infinity();
    }

    double lo = 0.0;
    // Primeiro, tenta encolher para chegar perto do limite.
    for (int it = 0; it < 30; ++it) {
        double cand = hi * 0.95;
        if (!(cand > 0.0)) {
            break;
        }
        if (!periodic_safe(base_poly, pattern, radius, cand, eps)) {
            lo = cand;
            break;
        }
        hi = cand;
    }

    for (int it = 0; it < 70; ++it) {
        double mid = 0.5 * (lo + hi);
        if (periodic_safe(base_poly, pattern, radius, mid, eps)) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    return hi;
}

struct TileScoreEval {
    double total = std::numeric_limits<double>::infinity();
    double best_angle = 0.0;
    double spacing = 0.0;
    bool ok = false;
};

double total_score_min_sides(const std::vector<double>& prefix_side_by_n,
                             const std::vector<double>* prune_side_by_n,
                             int n_max,
                             double best_cap) {
    double total = 0.0;
    for (int n = 1; n <= n_max; ++n) {
        double s = prefix_side_by_n[static_cast<size_t>(n)];
        if (prune_side_by_n) {
            s = std::min(s, (*prune_side_by_n)[static_cast<size_t>(n)]);
        }
        total += (s * s) / static_cast<double>(n);
        if (total >= best_cap) {
            break;
        }
    }
    return total;
}

TileScoreEval eval_tile_score_fast(const Polygon& base_poly,
                                   const Pattern& pattern,
                                   double radius,
                                   double spacing,
                                   double angle_deg,
                                   int pool_size,
                                   int n_max) {
    TileScoreEval out;
    out.best_angle = angle_deg;
    out.spacing = spacing;

    std::vector<TreePose> pool = generate_ordered_tiling(pool_size, spacing, angle_deg, pattern);
    if (static_cast<int>(pool.size()) < n_max) {
        return out;
    }
    std::vector<TreePose> prefix(pool.begin(), pool.begin() + n_max);
    std::vector<double> prefix_side_by_n =
        prefix_sides_from_bbs(bounding_boxes_for_poses(base_poly, prefix));
    out.total = total_score_from_sides(prefix_side_by_n, n_max);
    out.ok = std::isfinite(out.total);
    return out;
}

bool quantized_pool_is_valid(const Polygon& base_poly,
                             const std::vector<TreePose>& pool,
                             double radius) {
    auto pool_q = quantize_poses(pool);
    return !any_overlap(base_poly, pool_q, radius);
}

double find_min_feasible_spacing_quantized(const Polygon& base_poly,
                                           const Pattern& pattern,
                                           double radius,
                                           double angle_deg,
                                           int pool_size,
                                           double min_spacing_hint,
                                           double spacing_hi_hint) {
    auto safe = [&](double spacing) -> bool {
        if (spacing < min_spacing_hint) {
            return false;
        }
        std::vector<TreePose> pool = generate_ordered_tiling(pool_size, spacing, angle_deg, pattern);
        return quantized_pool_is_valid(base_poly, pool, radius);
    };

    double lo = std::max(min_spacing_hint, 1e-6);
    if (safe(lo)) {
        return lo;
    }

    double hi = lo;
    if (spacing_hi_hint > hi) {
        hi = spacing_hi_hint;
    }
    for (int it = 0; it < 60 && !safe(hi); ++it) {
        hi *= 1.01;
    }
    if (!safe(hi)) {
        return std::numeric_limits<double>::infinity();
    }

    for (int it = 0; it < 50; ++it) {
        double mid = 0.5 * (lo + hi);
        if (safe(mid)) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    return hi;
}

TileScoreEval eval_tile_score_full(const Polygon& base_poly,
                                   const Pattern& pattern,
                                   double radius,
                                   double min_spacing,
                                   const Options& opt) {
    TileScoreEval best;
    const int n_max = opt.tile_score_nmax;
    int pool_size = opt.tile_score_pool_size;
    if (pool_size <= 0) {
        pool_size = std::max(opt.n_max, n_max);
    }
    pool_size = std::max(pool_size, n_max);

    for (double ang : opt.angle_candidates) {
        double spacing = min_spacing * opt.spacing_safety;
        if (opt.tile_obj == TileObjective::kScore) {
            double s_q = find_min_feasible_spacing_quantized(
                base_poly, pattern, radius, ang, pool_size, min_spacing, spacing);
            if (std::isfinite(s_q)) {
                spacing = s_q * opt.spacing_safety;
            }
        }

        std::vector<TreePose> pool = generate_ordered_tiling(pool_size, spacing, ang, pattern);
        if (any_overlap(base_poly, pool, radius)) {
            continue;
        }
        auto pool_q = quantize_poses(pool);
        if (any_overlap(base_poly, pool_q, radius)) {
            continue;
        }

        std::vector<TreePose> prefix_nmax;
        if (opt.prefix_order == "greedy") {
            prefix_nmax = greedy_prefix_min_side(base_poly, pool_q, n_max);
        } else {
            prefix_nmax = std::vector<TreePose>(pool_q.begin(), pool_q.begin() + n_max);
        }

        std::vector<double> prefix_side_by_n =
            prefix_sides_from_bbs(bounding_boxes_for_poses(base_poly, prefix_nmax));

        double total = 0.0;
        if (!opt.final_rigid) {
            if (opt.prune) {
                std::vector<double> prune_side_by_n =
                    greedy_pruned_sides(bounding_boxes_for_poses(base_poly, pool_q), n_max, 1e-12);
                total = total_score_min_sides(prefix_side_by_n, &prune_side_by_n, n_max, best.total);
            } else {
                total = total_score_min_sides(prefix_side_by_n, nullptr, n_max, best.total);
            }
        } else {
            PruneResult pr;
            if (opt.prune) {
                pr = build_greedy_pruned_solutions(base_poly, pool_q, n_max);
            }

            for (int n = 1; n <= n_max; ++n) {
                double best_side = prefix_side_by_n[static_cast<size_t>(n)];
                std::vector<TreePose> best_sol(prefix_nmax.begin(), prefix_nmax.begin() + n);

                if (opt.prune) {
                    double s_prune = pr.side_by_n[static_cast<size_t>(n)];
                    if (s_prune + 1e-15 < best_side) {
                        best_side = s_prune;
                        best_sol = pr.solutions_by_n[static_cast<size_t>(n)];
                    }
                }

                std::vector<TreePose> rigid_sol = best_sol;
                optimize_rigid_rotation(base_poly, rigid_sol);
                auto rigid_q = quantize_poses(rigid_sol);
                double rigid_side =
                    bounding_square_side(transformed_polygons(base_poly, rigid_q));
                if (rigid_side + 1e-15 < best_side) {
                    best_side = rigid_side;
                }

                total += (best_side * best_side) / static_cast<double>(n);
                if (total >= best.total) {
                    break;
                }
            }
        }

        if (total < best.total) {
            best.total = total;
            best.best_angle = ang;
            best.spacing = spacing;
            best.ok = true;
        }
    }
    return best;
}

Pattern make_initial_pattern(int k) {
    Pattern p;
    p.shift_a = 0.0;
    p.shift_b = 0.0;
    if (k <= 0) {
        return p;
    }

    p.motif.reserve(static_cast<size_t>(k));
    p.motif.push_back(MotifPoint{0.0, 0.0, 0.0});
    if (k == 1) {
        return p;
    }

    int g = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(k - 1))));
    for (int idx = 0; idx < k - 1; ++idx) {
        int gx = idx % g;
        int gy = idx / g;
        double a = (static_cast<double>(gx) + 0.5) / static_cast<double>(g);
        double b = (static_cast<double>(gy) + 0.5) / static_cast<double>(g);
        double deg = (idx % 2 == 0) ? 0.0 : 180.0;
        p.motif.push_back(MotifPoint{wrap01(a), wrap01(b), deg});
    }
    return p;
}

Pattern optimize_tile_by_spacing(const Polygon& base_poly,
                                 Pattern pattern,
                                 double radius,
                                 const Options& opt) {
    if (opt.tile_iters <= 0) {
        return pattern;
    }
    if (pattern.motif.size() <= 1 && !opt.tile_opt_lattice) {
        return pattern;
    }

    std::mt19937_64 rng(opt.seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> uni01(0.0, 1.0);

    double best_spacing =
        find_min_safe_spacing(base_poly, pattern, radius, 1e-9, 2.0 * radius);
    if (!std::isfinite(best_spacing)) {
        throw std::runtime_error("Tile inicial não é viável (nunca fica seguro).");
    }

    auto area_per_tree = [&](double spacing, const Pattern& p) -> double {
        const int k = static_cast<int>(p.motif.size());
        if (k <= 0) {
            return std::numeric_limits<double>::infinity();
        }
        const double th = p.lattice_theta_deg * 3.14159265358979323846 / 180.0;
        const double cell_area =
            (spacing * spacing) * p.lattice_v_ratio * std::abs(std::sin(th));
        return cell_area / static_cast<double>(k);
    };

    Pattern best = pattern;
    double best_obj =
        (opt.tile_obj == TileObjective::kDensity)
            ? area_per_tree(best_spacing, best)
            : std::numeric_limits<double>::infinity();
    TileScoreEval best_score_eval;
    if (opt.tile_obj == TileObjective::kScore) {
        best_score_eval = eval_tile_score_full(base_poly, best, radius, best_spacing, opt);
        if (!best_score_eval.ok) {
            throw std::runtime_error("Tile inicial falhou na avaliação de score (nenhum ângulo válido).");
        }
        best_obj = best_score_eval.total;
    }

    // Para k pequeno (especialmente k=2), o greedy é bem estável e costuma achar
    // patterns densos rapidamente. Mantemos esse comportamento como padrão no
    // objetivo de densidade.
    if (opt.tile_obj == TileObjective::kDensity && pattern.motif.size() <= 2) {
        const bool can_move_motif = (pattern.motif.size() >= 2);
        for (int it = 0; it < opt.tile_iters; ++it) {
            Pattern cand = best;

            double t = static_cast<double>(it) / std::max(1, opt.tile_iters - 1);
            double sigma_pos = 0.10 * (1.0 - t) + 0.01 * t;
            double sigma_deg = 35.0 * (1.0 - t) + 3.0 * t;
            double sigma_ratio = 0.15 * (1.0 - t) + 0.02 * t;
            double sigma_theta = 12.0 * (1.0 - t) + 2.0 * t;

            const bool move_lattice =
                opt.tile_opt_lattice && (!can_move_motif || (uni01(rng) < 0.30));
            if (move_lattice) {
                cand.lattice_v_ratio =
                    std::max(0.50,
                             std::min(2.00,
                                      cand.lattice_v_ratio *
                                          std::exp(sigma_ratio * normal(rng))));
                cand.lattice_theta_deg =
                    std::max(20.0,
                             std::min(160.0,
                                      cand.lattice_theta_deg +
                                          sigma_theta * normal(rng)));
            } else {
                std::uniform_int_distribution<int> pick_idx(1, static_cast<int>(cand.motif.size()) - 1);
                const int idx = pick_idx(rng);
                cand.motif[static_cast<size_t>(idx)].a =
                    wrap01(cand.motif[static_cast<size_t>(idx)].a +
                           sigma_pos * normal(rng));
                cand.motif[static_cast<size_t>(idx)].b =
                    wrap01(cand.motif[static_cast<size_t>(idx)].b +
                           sigma_pos * normal(rng));
                cand.motif[static_cast<size_t>(idx)].deg =
                    wrap_deg(cand.motif[static_cast<size_t>(idx)].deg +
                             sigma_deg * normal(rng));
            }

            double spacing =
                find_min_safe_spacing(base_poly, cand, radius, 1e-9, best_spacing);
            if (!std::isfinite(spacing)) {
                continue;
            }
            const double obj = area_per_tree(spacing, cand);
            if (obj + 1e-12 < best_obj) {
                best_obj = obj;
                best_spacing = spacing;
                best = std::move(cand);
            }
        }
        return best;
    }

    // Para k>=3 (ou quando o objetivo é score), usa SA no pattern para escapar
    // de ótimos locais.
    Pattern curr = std::move(pattern);
    double curr_spacing = best_spacing;
    double curr_obj =
        (opt.tile_obj == TileObjective::kDensity)
            ? area_per_tree(curr_spacing, curr)
            : eval_tile_score_fast(base_poly,
                                   curr,
                                   radius,
                                   curr_spacing * opt.spacing_safety,
                                   0.0,
                                   std::max(opt.tile_score_pool_size, opt.tile_score_nmax),
                                   opt.tile_score_nmax)
                  .total;

    for (int it = 0; it < opt.tile_iters; ++it) {
        Pattern cand = curr;

        double t = static_cast<double>(it) / std::max(1, opt.tile_iters - 1);
        double sigma_pos = 0.10 * (1.0 - t) + 0.01 * t;
        double sigma_deg = 35.0 * (1.0 - t) + 3.0 * t;
        double sigma_ratio = 0.15 * (1.0 - t) + 0.02 * t;
        double sigma_theta = 12.0 * (1.0 - t) + 2.0 * t;

        const bool do_reset = (uni01(rng) < 0.10);
        const bool can_move_motif = (cand.motif.size() >= 2);
        const bool move_lattice =
            opt.tile_opt_lattice &&
            (!can_move_motif || (uni01(rng) < 0.20));
        if (move_lattice) {
            if (do_reset) {
                cand.lattice_v_ratio = 0.50 + uni01(rng) * 1.50;
                cand.lattice_theta_deg = 25.0 + uni01(rng) * 130.0;
            } else {
                cand.lattice_v_ratio =
                    std::max(0.50,
                             std::min(2.00,
                                      cand.lattice_v_ratio *
                                          std::exp(sigma_ratio * normal(rng))));
                cand.lattice_theta_deg =
                    std::max(20.0,
                             std::min(160.0,
                                      cand.lattice_theta_deg +
                                          sigma_theta * normal(rng)));
            }
        } else {
            std::uniform_int_distribution<int> pick_idx(
                1, static_cast<int>(cand.motif.size()) - 1);
            const int idx = pick_idx(rng);
            if (do_reset) {
                cand.motif[static_cast<size_t>(idx)].a = uni01(rng);
                cand.motif[static_cast<size_t>(idx)].b = uni01(rng);
                cand.motif[static_cast<size_t>(idx)].deg =
                    wrap_deg(360.0 * (uni01(rng) - 0.5));
            } else {
                cand.motif[static_cast<size_t>(idx)].a =
                    wrap01(cand.motif[static_cast<size_t>(idx)].a +
                           sigma_pos * normal(rng));
                cand.motif[static_cast<size_t>(idx)].b =
                    wrap01(cand.motif[static_cast<size_t>(idx)].b +
                           sigma_pos * normal(rng));
                cand.motif[static_cast<size_t>(idx)].deg =
                    wrap_deg(cand.motif[static_cast<size_t>(idx)].deg +
                             sigma_deg * normal(rng));
            }
        }

        double spacing = find_min_safe_spacing(base_poly, cand, radius, 1e-9, curr_spacing);
        if (!std::isfinite(spacing)) {
            continue;
        }

        double obj = 0.0;
        if (opt.tile_obj == TileObjective::kDensity) {
            obj = area_per_tree(spacing, cand);
        } else {
            const int pool_size = std::max(opt.tile_score_pool_size, opt.tile_score_nmax);
            TileScoreEval fast = eval_tile_score_fast(
                base_poly, cand, radius, spacing * opt.spacing_safety, 0.0, pool_size, opt.tile_score_nmax);
            if (!fast.ok) {
                continue;
            }
            obj = fast.total;
        }

        double T = 0.10 * (1.0 - t) + 0.01 * t;
        bool accept = false;
        if (obj + 1e-12 < curr_obj) {
            accept = true;
        } else if (T > 0.0) {
            double rel = (obj - curr_obj) / std::max(1e-12, curr_obj);
            double prob = std::exp(-rel / std::max(1e-12, T));
            accept = (uni01(rng) < prob);
        }

        if (accept) {
            curr = std::move(cand);
            curr_spacing = spacing;
            curr_obj = obj;
            if (opt.tile_obj == TileObjective::kDensity) {
                if (obj + 1e-12 < best_obj) {
                    best_obj = obj;
                    best_spacing = spacing;
                    best = curr;
                }
            }
        }

        if (opt.tile_obj == TileObjective::kScore &&
            ((it % opt.tile_score_full_every) == 0 || it + 1 == opt.tile_iters)) {
            TileScoreEval full = eval_tile_score_full(base_poly, curr, radius, curr_spacing, opt);
            if (full.ok && full.total + 1e-12 < best_obj) {
                best_obj = full.total;
                best_spacing = curr_spacing;
                best = curr;
                best_score_eval = full;
            }
        }
    }

    return best;
}
