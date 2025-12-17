#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "collision.hpp"
#include "geom.hpp"
#include "sa.hpp"

namespace {

constexpr double kSqrt3 = 1.732050807568877293527446341505872366942805254;
constexpr int kOutputDecimals = 9;

struct MotifPoint {
    double a;    // coord em u (fração do tile)
    double b;    // coord em v (fração do tile)
    double deg;  // rotação relativa (graus)
};

struct Pattern {
    std::vector<MotifPoint> motif;
    double shift_a = 0.0;
    double shift_b = 0.0;
};

struct Options {
    int n_max = 200;
    int k = 1;
    int tile_iters = 0;
    int refine_iters = 0;
    bool prune = true;
    int sa_restarts = 0;
    int sa_base_iters = 0;
    int sa_iters_per_n = 0;
    double sa_w_micro = 1.0;
    double sa_w_swap_rot = 0.25;
    double sa_w_relocate = 0.15;
    double sa_w_block_translate = 0.05;
    double sa_w_block_rotate = 0.02;
    double sa_w_lns = 0.001;
    int sa_block_size = 6;
    int sa_lns_remove = 6;
    int sa_hh_segment = 50;
    double sa_hh_reaction = 0.20;
    bool final_rigid = true;
    uint64_t seed = 123456789ULL;
    double spacing_safety = 1.001;
    double shift_a = 0.0;
    double shift_b = 0.0;
    std::vector<double> angle_candidates;
    std::string output_path = "submission_tile_cpp.csv";
};

struct Candidate {
    TreePose pose;
    double key1;
    double key2;
};

double wrap01(double x) {
    x -= std::floor(x);
    if (x < 0.0) {
        x += 1.0;
    }
    if (x >= 1.0) {
        x -= 1.0;
    }
    return x;
}

double wrap_deg(double deg) {
    deg = std::fmod(deg, 360.0);
    if (deg <= -180.0) {
        deg += 360.0;
    } else if (deg > 180.0) {
        deg -= 360.0;
    }
    return deg;
}

double quantize_value(double x) {
    const std::string s = fmt_submission_value(x, kOutputDecimals);
    return std::stod(s.substr(1));
}

TreePose quantize_pose(const TreePose& pose) {
    return TreePose{quantize_value(pose.x),
                    quantize_value(pose.y),
                    quantize_value(pose.deg)};
}

std::vector<TreePose> quantize_poses(const std::vector<TreePose>& poses) {
    std::vector<TreePose> out;
    out.reserve(poses.size());
    for (const auto& p : poses) {
        out.push_back(quantize_pose(p));
    }
    return out;
}

std::pair<Point, Point> hex_basis(double spacing, double angle_deg) {
    Point u{spacing, 0.0};
    Point v{0.5 * spacing, 0.5 * spacing * kSqrt3};
    return {rotate_point(u, angle_deg), rotate_point(v, angle_deg)};
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

    auto [u, v] = hex_basis(spacing, angle_deg);

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

double total_score_prefixes(const Polygon& base_poly,
                            const std::vector<TreePose>& poses) {
    if (poses.empty()) {
        return 0.0;
    }
    auto polys = transformed_polygons(base_poly, poses);
    std::vector<BoundingBox> bbs;
    bbs.reserve(polys.size());
    for (const auto& poly : polys) {
        bbs.push_back(bounding_box(poly));
    }

    double total = 0.0;
    double min_x = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < bbs.size(); ++i) {
        min_x = std::min(min_x, bbs[i].min_x);
        max_x = std::max(max_x, bbs[i].max_x);
        min_y = std::min(min_y, bbs[i].min_y);
        max_y = std::max(max_y, bbs[i].max_y);
        double width = max_x - min_x;
        double height = max_y - min_y;
        double s = std::max(width, height);
        total += (s * s) / static_cast<double>(i + 1);
    }

    return total;
}

double side_for_all(const std::vector<BoundingBox>& bbs) {
    if (bbs.empty()) {
        return 0.0;
    }
    double min_x = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    for (const auto& bb : bbs) {
        min_x = std::min(min_x, bb.min_x);
        max_x = std::max(max_x, bb.max_x);
        min_y = std::min(min_y, bb.min_y);
        max_y = std::max(max_y, bb.max_y);
    }
    return std::max(max_x - min_x, max_y - min_y);
}

struct Extents {
    double min_x;
    double max_x;
    double min_y;
    double max_y;
};

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

double side_from_extents(const Extents& e) {
    return std::max(e.max_x - e.min_x, e.max_y - e.min_y);
}

std::vector<int> boundary_indices(const std::vector<BoundingBox>& bbs,
                                  const Extents& e,
                                  double tol) {
    std::vector<int> out;
    out.reserve(bbs.size());
    for (size_t i = 0; i < bbs.size(); ++i) {
        const auto& bb = bbs[i];
        if (bb.min_x <= e.min_x + tol || bb.max_x >= e.max_x - tol ||
            bb.min_y <= e.min_y + tol || bb.max_y >= e.max_y - tol) {
            out.push_back(static_cast<int>(i));
        }
    }
    return out;
}

Extents extents_without_index(const std::vector<BoundingBox>& bbs, int skip) {
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

struct PruneResult {
    std::vector<std::vector<TreePose>> solutions_by_n;  // index = n
    std::vector<double> side_by_n;  // index = n (lado do bounding square)
    double total_score = 0.0;
};

std::vector<BoundingBox> bounding_boxes_for_poses(const Polygon& base_poly,
                                                  const std::vector<TreePose>& poses) {
    std::vector<BoundingBox> bbs;
    bbs.reserve(poses.size());
    for (const auto& pose : poses) {
        bbs.push_back(bounding_box(transform_polygon(base_poly, pose)));
    }
    return bbs;
}

std::vector<double> prefix_sides_from_bbs(const std::vector<BoundingBox>& bbs) {
    std::vector<double> side;
    side.resize(bbs.size() + 1, 0.0);
    Extents e;
    e.min_x = std::numeric_limits<double>::infinity();
    e.max_x = -std::numeric_limits<double>::infinity();
    e.min_y = std::numeric_limits<double>::infinity();
    e.max_y = -std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < bbs.size(); ++i) {
        e.min_x = std::min(e.min_x, bbs[i].min_x);
        e.max_x = std::max(e.max_x, bbs[i].max_x);
        e.min_y = std::min(e.min_y, bbs[i].min_y);
        e.max_y = std::max(e.max_y, bbs[i].max_y);
        side[i + 1] = side_from_extents(e);
    }
    return side;
}

PruneResult build_greedy_pruned_solutions(const Polygon& base_poly,
                                         const std::vector<TreePose>& poses_nmax,
                                         int n_max,
                                         double tol = 1e-12) {
    if (static_cast<int>(poses_nmax.size()) != n_max) {
        throw std::runtime_error("build_greedy_pruned_solutions: tamanho != n_max.");
    }

    std::vector<TreePose> poses = poses_nmax;
    std::vector<BoundingBox> bbs = bounding_boxes_for_poses(base_poly, poses);

    PruneResult res;
    res.solutions_by_n.resize(static_cast<size_t>(n_max + 1));
    res.side_by_n.resize(static_cast<size_t>(n_max + 1), 0.0);

    for (int m = n_max; m >= 1; --m) {
        Extents e = compute_extents(bbs);
        double s = side_from_extents(e);
        res.total_score += (s * s) / static_cast<double>(m);
        res.side_by_n[static_cast<size_t>(m)] = s;
        res.solutions_by_n[static_cast<size_t>(m)] = poses;

        if (m == 1) {
            break;
        }

        std::vector<int> candidates = boundary_indices(bbs, e, tol);
        if (candidates.empty()) {
            candidates.resize(bbs.size());
            for (size_t i = 0; i < bbs.size(); ++i) {
                candidates[i] = static_cast<int>(i);
            }
        }

        int best_idx = candidates.front();
        double best_side = std::numeric_limits<double>::infinity();

        for (int idx : candidates) {
            Extents e2 = extents_without_index(bbs, idx);
            double s2 = side_from_extents(e2);
            if (s2 < best_side - 1e-15 ||
                (std::abs(s2 - best_side) <= 1e-15 && idx < best_idx)) {
                best_side = s2;
                best_idx = idx;
            }
        }

        poses.erase(poses.begin() + best_idx);
        bbs.erase(bbs.begin() + best_idx);
    }

    return res;
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

    auto [u, v] = hex_basis(spacing, 0.0);

    int m = static_cast<int>(std::ceil((2.0 * radius) / spacing)) + 8;
    m = std::max(2, m);

    const double limit = 2.0 * radius + eps;
    const double limit_sq = limit * limit;

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

    for (int i = -m; i <= m; ++i) {
        for (int j = -m; j <= m; ++j) {
            if (i == 0 && j == 0) {
                continue;
            }
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
    }

    return true;
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
    if (opt.tile_iters <= 0 || pattern.motif.size() <= 1) {
        return pattern;
    }

    std::mt19937_64 rng(opt.seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> uni01(0.0, 1.0);
    std::uniform_int_distribution<int> pick_idx(
        0, std::max(0, static_cast<int>(pattern.motif.size()) - 1));

    double best_spacing =
        find_min_safe_spacing(base_poly, pattern, radius, 1e-9, 2.0 * radius);
    if (!std::isfinite(best_spacing)) {
        throw std::runtime_error("Tile inicial não é viável (nunca fica seguro).");
    }

    Pattern best = pattern;

    // Para k pequeno (especialmente k=2), o greedy é bem estável e costuma achar
    // patterns densos rapidamente. Mantemos esse comportamento como padrão.
    if (pattern.motif.size() <= 2) {
        for (int it = 0; it < opt.tile_iters; ++it) {
            Pattern cand = best;
            int idx = pick_idx(rng);
            if (idx == 0) {
                // Mantém (0,0) fixo para preservar uma árvore no centro.
                idx = 1 + (it % std::max(1, static_cast<int>(cand.motif.size()) - 1));
            }
            if (idx >= static_cast<int>(cand.motif.size())) {
                idx = 0;
            }

            double t = static_cast<double>(it) / std::max(1, opt.tile_iters - 1);
            double sigma_pos = 0.10 * (1.0 - t) + 0.01 * t;
            double sigma_deg = 35.0 * (1.0 - t) + 3.0 * t;

            cand.motif[static_cast<size_t>(idx)].a =
                wrap01(cand.motif[static_cast<size_t>(idx)].a +
                       sigma_pos * normal(rng));
            cand.motif[static_cast<size_t>(idx)].b =
                wrap01(cand.motif[static_cast<size_t>(idx)].b +
                       sigma_pos * normal(rng));
            cand.motif[static_cast<size_t>(idx)].deg =
                wrap_deg(cand.motif[static_cast<size_t>(idx)].deg +
                         sigma_deg * normal(rng));

            double spacing =
                find_min_safe_spacing(base_poly, cand, radius, 1e-9, best_spacing);
            if (!std::isfinite(spacing)) {
                continue;
            }
            if (spacing + 1e-12 < best_spacing) {
                best_spacing = spacing;
                best = std::move(cand);
            }
        }
        return best;
    }

    // Para k>=3, usa SA no pattern para escapar de ótimos locais.
    Pattern curr = std::move(pattern);
    double curr_spacing = best_spacing;

    for (int it = 0; it < opt.tile_iters; ++it) {
        Pattern cand = curr;
        int idx = pick_idx(rng);
        if (idx == 0) {
            // Mantém (0,0) fixo para preservar uma árvore no centro.
            idx = 1 + (it % std::max(1, static_cast<int>(cand.motif.size()) - 1));
        }
        if (idx >= static_cast<int>(cand.motif.size())) {
            idx = 0;
        }

        double t = static_cast<double>(it) / std::max(1, opt.tile_iters - 1);
        double sigma_pos = 0.10 * (1.0 - t) + 0.01 * t;
        double sigma_deg = 35.0 * (1.0 - t) + 3.0 * t;

        const bool do_reset = (cand.motif.size() >= 3) && (uni01(rng) < 0.10);
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

        double spacing =
            find_min_safe_spacing(base_poly, cand, radius, 1e-9, curr_spacing);
        if (!std::isfinite(spacing)) {
            continue;
        }

        double T = 0.10 * (1.0 - t) + 0.01 * t;
        bool accept = false;
        if (spacing + 1e-12 < curr_spacing) {
            accept = true;
        } else if (T > 0.0) {
            double rel = (spacing - curr_spacing) / std::max(1e-12, curr_spacing);
            double prob = std::exp(-rel / std::max(1e-12, T));
            accept = (uni01(rng) < prob);
        }

        if (accept) {
            curr = std::move(cand);
            curr_spacing = spacing;
            if (spacing + 1e-12 < best_spacing) {
                best_spacing = spacing;
                best = curr;
            }
        }
    }

    return best;
}

struct Eval {
    double best_total = std::numeric_limits<double>::infinity();
    double best_angle = 0.0;
    std::vector<TreePose> best_poses;
};

Eval choose_best_angle_for_prefix_score(const Polygon& base_poly,
                                        const Pattern& pattern,
                                        double radius,
                                        double spacing,
                                        const std::vector<double>& angle_candidates) {
    Eval best;
    for (double ang : angle_candidates) {
        auto poses = generate_ordered_tiling(200, spacing, ang, pattern);
        if (any_overlap(base_poly, poses, radius)) {
            continue;
        }
        double total = total_score_prefixes(base_poly, poses);
        if (total < best.best_total) {
            best.best_total = total;
            best.best_angle = ang;
            best.best_poses = std::move(poses);
        }
    }
    if (!std::isfinite(best.best_total)) {
        throw std::runtime_error("Nenhum ângulo candidato gerou configuração válida.");
    }
    return best;
}

void refine_boundary(const Polygon& base_poly,
                     double radius,
                     std::vector<TreePose>& poses,
                     int iters,
                     uint64_t seed,
                     double step_hint) {
    if (iters <= 0 || poses.empty()) {
        return;
    }

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::normal_distribution<double> normal(0.0, 1.0);

    const double eps = 1e-9;
    const double limit_sq = (2.0 * radius + eps) * (2.0 * radius + eps);

    auto polys = transformed_polygons(base_poly, poses);
    std::vector<BoundingBox> bbs;
    bbs.reserve(polys.size());
    for (const auto& poly : polys) {
        bbs.push_back(bounding_box(poly));
    }

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

        TreePose cand = poses[static_cast<size_t>(idx)];

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

        cand.x += dir_x * step + normal(rng) * step * 0.15;
        cand.y += dir_y * step + normal(rng) * step * 0.15;
        cand.deg = wrap_deg(cand.deg + normal(rng) * 5.0);

        if (cand.x < -100.0 || cand.x > 100.0 || cand.y < -100.0 || cand.y > 100.0) {
            continue;
        }

        Polygon cand_poly = transform_polygon(base_poly, cand);
        BoundingBox cand_bb = bounding_box(cand_poly);

        bool collide = false;
        for (size_t j = 0; j < poses.size(); ++j) {
            if (static_cast<int>(j) == idx) {
                continue;
            }
            double dx = cand.x - poses[j].x;
            double dy = cand.y - poses[j].y;
            if (dx * dx + dy * dy > limit_sq) {
                continue;
            }
            if (polygons_intersect(cand_poly, polys[j])) {
                collide = true;
                break;
            }
        }
        if (collide) {
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
        } else if (T > 0.0) {
            double prob = std::exp((current_s - new_s) / T);
            if (uni(rng) < prob) {
                accept = true;
            }
        }

        if (!accept) {
            continue;
        }

        poses[static_cast<size_t>(idx)] = cand;
        polys[static_cast<size_t>(idx)] = std::move(cand_poly);
        bbs[static_cast<size_t>(idx)] = cand_bb;
        current_s = recompute_side_and_extents(min_x, max_x, min_y, max_y);
    }
}

Options parse_args(int argc, char** argv) {
    Options opt;
    opt.angle_candidates = {0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0};

    auto parse_int = [](const std::string& s) -> int {
        size_t pos = 0;
        int v = std::stoi(s, &pos);
        if (pos != s.size()) {
            throw std::runtime_error("Inteiro inválido: " + s);
        }
        return v;
    };
    auto parse_u64 = [](const std::string& s) -> uint64_t {
        size_t pos = 0;
        uint64_t v = std::stoull(s, &pos);
        if (pos != s.size()) {
            throw std::runtime_error("uint64 inválido: " + s);
        }
        return v;
    };
    auto parse_double = [](const std::string& s) -> double {
        size_t pos = 0;
        double v = std::stod(s, &pos);
        if (pos != s.size()) {
            throw std::runtime_error("Double inválido: " + s);
        }
        return v;
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Faltou valor para " + name);
            }
            return argv[++i];
        };

        if (arg == "--k") {
            opt.k = parse_int(need(arg));
        } else if (arg == "--tile-iters") {
            opt.tile_iters = parse_int(need(arg));
        } else if (arg == "--refine-iters") {
            opt.refine_iters = parse_int(need(arg));
        } else if (arg == "--sa-restarts") {
            opt.sa_restarts = parse_int(need(arg));
        } else if (arg == "--sa-base-iters") {
            opt.sa_base_iters = parse_int(need(arg));
        } else if (arg == "--sa-iters-per-n") {
            opt.sa_iters_per_n = parse_int(need(arg));
        } else if (arg == "--sa-w-micro") {
            opt.sa_w_micro = parse_double(need(arg));
        } else if (arg == "--sa-w-swap-rot") {
            opt.sa_w_swap_rot = parse_double(need(arg));
        } else if (arg == "--sa-w-relocate") {
            opt.sa_w_relocate = parse_double(need(arg));
        } else if (arg == "--sa-w-block-translate") {
            opt.sa_w_block_translate = parse_double(need(arg));
        } else if (arg == "--sa-w-block-rotate") {
            opt.sa_w_block_rotate = parse_double(need(arg));
        } else if (arg == "--sa-w-lns") {
            opt.sa_w_lns = parse_double(need(arg));
        } else if (arg == "--sa-block-size") {
            opt.sa_block_size = parse_int(need(arg));
        } else if (arg == "--sa-lns-remove") {
            opt.sa_lns_remove = parse_int(need(arg));
        } else if (arg == "--sa-hh-segment") {
            opt.sa_hh_segment = parse_int(need(arg));
        } else if (arg == "--sa-hh-reaction") {
            opt.sa_hh_reaction = parse_double(need(arg));
        } else if (arg == "--no-final-rigid" || arg == "--no-sa-rigid") {
            opt.final_rigid = false;
        } else if (arg == "--seed") {
            opt.seed = parse_u64(need(arg));
        } else if (arg == "--spacing-safety") {
            opt.spacing_safety = parse_double(need(arg));
        } else if (arg == "--shift-a") {
            opt.shift_a = parse_double(need(arg));
        } else if (arg == "--shift-b") {
            opt.shift_b = parse_double(need(arg));
        } else if (arg == "--shift") {
            std::string s = need(arg);
            std::stringstream ss(s);
            std::string a, b;
            if (!std::getline(ss, a, ',') || !std::getline(ss, b, ',')) {
                throw std::runtime_error("--shift precisa ser 'a,b'.");
            }
            opt.shift_a = parse_double(a);
            opt.shift_b = parse_double(b);
        } else if (arg == "--angles") {
            std::string s = need(arg);
            opt.angle_candidates.clear();
            std::stringstream ss(s);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (item.empty()) {
                    continue;
                }
                opt.angle_candidates.push_back(parse_double(item));
            }
            if (opt.angle_candidates.empty()) {
                throw std::runtime_error("--angles vazio.");
            }
        } else if (arg == "--output") {
            opt.output_path = need(arg);
        } else if (arg == "--no-prune") {
            opt.prune = false;
        } else {
            throw std::runtime_error("Argumento desconhecido: " + arg);
        }
    }

    if (opt.k <= 0) {
        throw std::runtime_error("--k precisa ser > 0.");
    }
    if (!(opt.spacing_safety >= 1.0)) {
        throw std::runtime_error("--spacing-safety precisa ser >= 1.0.");
    }
    if (opt.sa_w_micro < 0.0 || opt.sa_w_swap_rot < 0.0 || opt.sa_w_relocate < 0.0 ||
        opt.sa_w_block_translate < 0.0 || opt.sa_w_block_rotate < 0.0 || opt.sa_w_lns < 0.0) {
        throw std::runtime_error("Pesos de SA precisam ser >= 0.");
    }
    if (opt.sa_block_size <= 0) {
        throw std::runtime_error("--sa-block-size precisa ser > 0.");
    }
    if (opt.sa_lns_remove < 0) {
        throw std::runtime_error("--sa-lns-remove precisa ser >= 0.");
    }
    if (opt.sa_hh_segment < 0) {
        throw std::runtime_error("--sa-hh-segment precisa ser >= 0.");
    }
    if (opt.sa_hh_reaction < 0.0 || opt.sa_hh_reaction > 1.0) {
        throw std::runtime_error("--sa-hh-reaction precisa estar em [0, 1].");
    }
    return opt;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);

        Polygon base_poly = get_tree_polygon();
        const double radius = enclosing_circle_radius(base_poly);
        const double eps = 1e-9;

        Pattern pattern = make_initial_pattern(opt.k);
        pattern = optimize_tile_by_spacing(base_poly, pattern, radius, opt);
        pattern.shift_a = wrap01(opt.shift_a);
        pattern.shift_b = wrap01(opt.shift_b);

        double min_spacing =
            find_min_safe_spacing(base_poly, pattern, radius, eps, 2.0 * radius);
        if (!std::isfinite(min_spacing)) {
            throw std::runtime_error("Não foi possível encontrar spacing seguro.");
        }
        const double spacing = min_spacing * opt.spacing_safety;

        Eval chosen = choose_best_angle_for_prefix_score(
            base_poly, pattern, radius, spacing, opt.angle_candidates);

        std::vector<TreePose> poses200 = std::move(chosen.best_poses);

        refine_boundary(base_poly, radius, poses200, opt.refine_iters, opt.seed + 999, spacing);

        // Reordena por "centralidade" após o refino (melhora recortes n pequenos).
        {
            std::vector<Candidate> tmp;
            tmp.reserve(poses200.size());
            for (const auto& pose : poses200) {
                tmp.push_back({pose,
                               std::max(std::abs(pose.x), std::abs(pose.y)),
                               std::hypot(pose.x, pose.y)});
            }
            std::sort(tmp.begin(),
                      tmp.end(),
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
            for (size_t i = 0; i < poses200.size(); ++i) {
                poses200[i] = tmp[i].pose;
            }
        }

        if (any_overlap(base_poly, poses200, radius)) {
            throw std::runtime_error("Overlap detectado após refino.");
        }

        auto poses200_q = quantize_poses(poses200);
        if (any_overlap(base_poly, poses200_q, radius)) {
            throw std::runtime_error(
                "Overlap detectado após arredondamento para o submission.");
        }

        std::vector<std::vector<TreePose>> prefix_by_n;
        prefix_by_n.resize(static_cast<size_t>(opt.n_max + 1));
        for (int n = 1; n <= opt.n_max; ++n) {
            prefix_by_n[static_cast<size_t>(n)] =
                std::vector<TreePose>(poses200_q.begin(), poses200_q.begin() + n);
        }

        std::vector<double> prefix_side_by_n =
            prefix_sides_from_bbs(bounding_boxes_for_poses(base_poly, poses200_q));

        std::vector<std::vector<TreePose>> solutions_by_n;
        solutions_by_n.resize(static_cast<size_t>(opt.n_max + 1));

        double total_score = 0.0;
        if (opt.prune) {
            PruneResult pr = build_greedy_pruned_solutions(base_poly, poses200_q, opt.n_max);
            for (int n = 1; n <= opt.n_max; ++n) {
                double s_prefix = prefix_side_by_n[static_cast<size_t>(n)];
                double s_prune = pr.side_by_n[static_cast<size_t>(n)];
                if (s_prune + 1e-15 < s_prefix) {
                    solutions_by_n[static_cast<size_t>(n)] = pr.solutions_by_n[static_cast<size_t>(n)];
                    total_score += (s_prune * s_prune) / static_cast<double>(n);
                } else {
                    solutions_by_n[static_cast<size_t>(n)] = prefix_by_n[static_cast<size_t>(n)];
                    total_score += (s_prefix * s_prefix) / static_cast<double>(n);
                }
            }
        } else {
            for (int n = 1; n <= opt.n_max; ++n) {
                double s = prefix_side_by_n[static_cast<size_t>(n)];
                solutions_by_n[static_cast<size_t>(n)] = std::move(prefix_by_n[static_cast<size_t>(n)]);
                total_score += (s * s) / static_cast<double>(n);
            }
        }

        if (opt.sa_restarts > 0 && opt.sa_base_iters > 0) {
            double total_score_sa = 0.0;
            SARefiner sa(base_poly, radius);

            for (int n = 1; n <= opt.n_max; ++n) {
                SARefiner::Params p;
                p.iters = opt.sa_base_iters + opt.sa_iters_per_n * n;
                p.w_micro = opt.sa_w_micro;
                p.w_swap_rot = opt.sa_w_swap_rot;
                p.w_relocate = opt.sa_w_relocate;
                p.w_block_translate = opt.sa_w_block_translate;
                p.w_block_rotate = opt.sa_w_block_rotate;
                p.w_lns = opt.sa_w_lns;
                p.block_size = opt.sa_block_size;
                p.lns_remove = opt.sa_lns_remove;
                p.hh_segment = opt.sa_hh_segment;
                p.hh_reaction = opt.sa_hh_reaction;

                std::vector<TreePose> best_sol =
                    solutions_by_n[static_cast<size_t>(n)];
                double best_side = bounding_square_side(
                    transformed_polygons(base_poly, best_sol));

                for (int r = 0; r < opt.sa_restarts; ++r) {
                    uint64_t seed =
                        opt.seed ^
                        (0x9e3779b97f4a7c15ULL +
                         static_cast<uint64_t>(n) * 0xbf58476d1ce4e5b9ULL +
                         static_cast<uint64_t>(r) * 0x94d049bb133111ebULL);
                    SARefiner::Result res =
                        sa.refine_min_side(best_sol, seed, p);

                    auto cand_q = quantize_poses(res.best_poses);
                    if (any_overlap(base_poly, cand_q, radius)) {
                        continue;
                    }
                    double cand_side = bounding_square_side(
                        transformed_polygons(base_poly, cand_q));
                    if (cand_side + 1e-15 < best_side) {
                        best_side = cand_side;
                        best_sol = std::move(cand_q);
                    }
                }

                solutions_by_n[static_cast<size_t>(n)] = best_sol;
                total_score_sa += (best_side * best_side) /
                                  static_cast<double>(n);
            }

            total_score = total_score_sa;
        }

        // Pós-processamento "final rigid": otimiza um ângulo global por n.
        if (opt.final_rigid) {
            double total_score_rigid = 0.0;
            for (int n = 1; n <= opt.n_max; ++n) {
                std::vector<TreePose> best_sol =
                    solutions_by_n[static_cast<size_t>(n)];
                double best_side = bounding_square_side(
                    transformed_polygons(base_poly, best_sol));

                std::vector<TreePose> rigid_sol = best_sol;
                optimize_rigid_rotation(base_poly, rigid_sol);
                auto rigid_q = quantize_poses(rigid_sol);
                if (!any_overlap(base_poly, rigid_q, radius)) {
                    double rigid_side = bounding_square_side(
                        transformed_polygons(base_poly, rigid_q));
                    if (rigid_side + 1e-15 < best_side) {
                        best_side = rigid_side;
                        best_sol = std::move(rigid_q);
                        solutions_by_n[static_cast<size_t>(n)] = best_sol;
                    }
                }

                total_score_rigid += (best_side * best_side) /
                                     static_cast<double>(n);
            }
            total_score = total_score_rigid;
        }

        std::ofstream out(opt.output_path);
        if (!out) {
            throw std::runtime_error("Erro ao abrir arquivo de saída: " + opt.output_path);
        }
        out << "id,x,y,deg\n";
        for (int n = 1; n <= opt.n_max; ++n) {
            const auto& sol = solutions_by_n[static_cast<size_t>(n)];
            for (int i = 0; i < n; ++i) {
                const auto& pose = sol[static_cast<size_t>(i)];
                out << std::setw(3) << std::setfill('0') << n << "_" << i << ","
                    << fmt_submission_value(pose.x) << ","
                    << fmt_submission_value(pose.y) << ","
                    << fmt_submission_value(pose.deg) << "\n";
            }
        }

        std::cout << "Submission gerada em " << opt.output_path << "\n";
        std::cout << "k (tile): " << opt.k << "\n";
        std::cout << "Tile iters: " << opt.tile_iters << "\n";
        std::cout << "Refine iters: " << opt.refine_iters << "\n";
        std::cout << "Min spacing: " << std::fixed << std::setprecision(9) << min_spacing << "\n";
        std::cout << "Spacing (safety): " << std::fixed << std::setprecision(9) << spacing << "\n";
        std::cout << "Best angle: " << std::fixed << std::setprecision(3) << chosen.best_angle << "\n";
        std::cout << "Score (local): " << std::fixed << std::setprecision(9) << total_score << "\n";
        std::cout << "Prune: " << (opt.prune ? "on" : "off") << "\n";
        std::cout << "Final rigid: " << (opt.final_rigid ? "on" : "off") << "\n";

    } catch (const std::exception& ex) {
        std::cerr << "Erro: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
