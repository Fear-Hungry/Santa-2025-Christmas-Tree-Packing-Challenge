#include "ga.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "collision.hpp"
#include "geom.hpp"
#include "spatial_grid.hpp"
#include "wrap_utils.hpp"

namespace {

constexpr double kSqrt3 = 1.732050807568877293527446341505872366942805254;

struct CandidatePoint {
    Point p;
    double key1;
    double key2;
};

std::pair<Point, Point> hex_basis(double spacing, double angle_deg) {
    Point u{spacing, 0.0};
    Point v{0.5 * spacing, 0.5 * spacing * kSqrt3};
    return {rotate_point(u, angle_deg), rotate_point(v, angle_deg)};
}

double rotation_pattern_deg(int i, int j) {
    (void)i;
    (void)j;
    return 0.0;
}

bool safe_hex_spacing(const Polygon& base_poly,
                      double radius,
                      double spacing,
                      double eps) {
    if (!(spacing > 0.0)) {
        return false;
    }

    auto [u, v] = hex_basis(spacing, 0.0);

    const double limit = 2.0 * radius + eps;
    const double limit_sq = limit * limit;
    int m = static_cast<int>(std::ceil(limit / spacing)) + 2;
    m = std::max(2, m);

    const TreePose origin{0.0, 0.0, rotation_pattern_deg(0, 0)};

    for (int i = -m; i <= m; ++i) {
        for (int j = -m; j <= m; ++j) {
            if (i == 0 && j == 0) {
                continue;
            }
            double dx = i * u.x + j * v.x;
            double dy = i * u.y + j * v.y;
            double d2 = dx * dx + dy * dy;
            if (d2 > limit_sq) {
                continue;
            }

            const TreePose other{dx, dy, rotation_pattern_deg(i, j)};
            const std::vector<TreePose> poses{origin, other};
            if (any_overlap(base_poly, poses, radius, eps)) {
                return false;
            }
        }
    }
    return true;
}

double find_min_safe_hex_spacing(const Polygon& base_poly,
                                double radius,
                                double eps) {
    double lo = 0.0;
    double hi = 2.0 * radius;
    if (!safe_hex_spacing(base_poly, radius, hi, eps)) {
        for (int it = 0; it < 30 && !safe_hex_spacing(base_poly, radius, hi, eps);
             ++it) {
            hi *= 1.5;
        }
    }

    for (int it = 0; it < 70; ++it) {
        double mid = 0.5 * (lo + hi);
        if (safe_hex_spacing(base_poly, radius, mid, eps)) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    return hi;
}

std::vector<CandidatePoint> generate_candidate_points(int n,
                                                      double spacing,
                                                      double angle_deg,
                                                      double shift_a,
                                                      double shift_b,
                                                      int candidate_target) {
    if (n <= 0) {
        return {};
    }

    const int target = std::max(candidate_target, n);
    int m = static_cast<int>(std::ceil((std::sqrt(static_cast<double>(target)) - 1.0) / 2.0));
    m = std::max(m, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(n)))) + 8);

    auto [u, v] = hex_basis(spacing, angle_deg);

    std::vector<CandidatePoint> candidates;
    candidates.reserve(static_cast<size_t>((2 * m + 1) * (2 * m + 1)));
    for (int i = -m; i <= m; ++i) {
        for (int j = -m; j <= m; ++j) {
            double ci = static_cast<double>(i) - shift_a;
            double cj = static_cast<double>(j) - shift_b;
            double x = ci * u.x + cj * v.x;
            double y = ci * u.y + cj * v.y;
            double key1 = std::max(std::abs(x), std::abs(y));
            double key2 = std::hypot(x, y);
            candidates.push_back(CandidatePoint{Point{x, y}, key1, key2});
        }
    }

    std::sort(candidates.begin(),
              candidates.end(),
              [](const CandidatePoint& a, const CandidatePoint& b) {
                  if (a.key1 != b.key1) {
                      return a.key1 < b.key1;
                  }
                  if (a.key2 != b.key2) {
                      return a.key2 < b.key2;
                  }
                  if (a.p.x != b.p.x) {
                      return a.p.x < b.p.x;
                  }
                  return a.p.y < b.p.y;
              });
    return candidates;
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

bool aabb_overlap(const BoundingBox& a, const BoundingBox& b) {
    if (a.max_x < b.min_x || b.max_x < a.min_x) {
        return false;
    }
    if (a.max_y < b.min_y || b.max_y < a.min_y) {
        return false;
    }
    return true;
}

struct DecodeResult {
    std::vector<TreePose> poses;
    std::vector<BoundingBox> bbs;
    double side = std::numeric_limits<double>::infinity();
    bool ok = false;
};

DecodeResult decode_constructive(const Polygon& base_poly,
                                double radius,
                                int n,
                                const std::vector<int>& perm,
                                const std::vector<int>& rot_idx,
                                const std::vector<double>& rot_candidates,
                                double spacing,
                                double angle_deg,
                                double shift_a,
                                double shift_b,
                                int candidate_target) {
    DecodeResult res;
    if (n <= 0) {
        res.ok = true;
        res.side = 0.0;
        return res;
    }

    const double eps = 1e-12;
    const double thr = 2.0 * radius + 1e-9;
    const double limit_sq = thr * thr;

    std::vector<CandidatePoint> candidates =
        generate_candidate_points(n, spacing, angle_deg, shift_a, shift_b, candidate_target);
    if (static_cast<int>(candidates.size()) < n) {
        return res;
    }

    std::vector<char> used(candidates.size(), 0);
    std::vector<TreePose> out(static_cast<size_t>(n));
    std::vector<Point> centers;
    centers.reserve(static_cast<size_t>(n));
    std::vector<Polygon> polys;
    polys.reserve(static_cast<size_t>(n));
    std::vector<BoundingBox> bbs;
    bbs.reserve(static_cast<size_t>(n));

    UniformGridIndex grid(n, thr);
    std::vector<int> neigh;
    neigh.reserve(64);

    Extents e;
    e.min_x = std::numeric_limits<double>::infinity();
    e.max_x = -std::numeric_limits<double>::infinity();
    e.min_y = std::numeric_limits<double>::infinity();
    e.max_y = -std::numeric_limits<double>::infinity();

    for (int place = 0; place < n; ++place) {
        const int tree = perm[static_cast<size_t>(place)];
        const int ridx = rot_idx[static_cast<size_t>(tree)];
        const double deg = rot_candidates[static_cast<size_t>(ridx)];

        bool placed = false;
        for (size_t attempt = 0; attempt < candidates.size(); ++attempt) {
            if (used[attempt]) {
                continue;
            }
            const double x = candidates[attempt].p.x;
            const double y = candidates[attempt].p.y;
            if (x < -100.0 || x > 100.0 || y < -100.0 || y > 100.0) {
                continue;
            }

            TreePose cand{ x, y, wrap_deg(deg) };

            Polygon cand_poly = transform_polygon(base_poly, cand);
            BoundingBox cand_bb = bounding_box(cand_poly);

            bool collide = false;
            grid.gather(x, y, neigh);
            for (int j : neigh) {
                double dx = x - centers[static_cast<size_t>(j)].x;
                double dy = y - centers[static_cast<size_t>(j)].y;
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
                continue;
            }

            used[attempt] = 1;
            out[static_cast<size_t>(tree)] = cand;

            const int new_idx = static_cast<int>(centers.size());
            centers.push_back(Point{x, y});
            polys.push_back(std::move(cand_poly));
            bbs.push_back(cand_bb);
            grid.insert(new_idx, x, y);

            if (centers.size() == 1) {
                e.min_x = cand_bb.min_x;
                e.max_x = cand_bb.max_x;
                e.min_y = cand_bb.min_y;
                e.max_y = cand_bb.max_y;
            } else {
                e.min_x = std::min(e.min_x, cand_bb.min_x);
                e.max_x = std::max(e.max_x, cand_bb.max_x);
                e.min_y = std::min(e.min_y, cand_bb.min_y);
                e.max_y = std::max(e.max_y, cand_bb.max_y);
            }

            placed = true;
            break;
        }

        if (!placed) {
            return res;
        }
    }

    if (any_overlap(base_poly, out, radius, eps)) {
        return res;
    }

    res.poses = std::move(out);
    res.bbs = std::move(bbs);
    res.side = side_from_extents(e);
    res.ok = true;
    return res;
}

struct Individual {
    std::vector<int> perm;
    std::vector<int> rot_idx;
    double angle_deg = 0.0;
    double shift_a = 0.0;
    double shift_b = 0.0;
    double spacing_safety = 1.001;
    double fitness = std::numeric_limits<double>::infinity();
    std::vector<TreePose> cached_poses;
};

Individual make_random_individual(int n, std::mt19937_64& rng, const GAParams& p) {
    Individual ind;
    ind.perm.resize(static_cast<size_t>(n));
    ind.rot_idx.resize(static_cast<size_t>(n));

    for (int i = 0; i < n; ++i) {
        ind.perm[static_cast<size_t>(i)] = i;
    }
    std::shuffle(ind.perm.begin(), ind.perm.end(), rng);

    std::uniform_int_distribution<int> pick_rot(0, static_cast<int>(p.rotation_candidates.size()) - 1);
    for (int i = 0; i < n; ++i) {
        ind.rot_idx[static_cast<size_t>(i)] = pick_rot(rng);
    }

    std::uniform_real_distribution<double> uni01(0.0, 1.0);
    ind.shift_a = uni01(rng);
    ind.shift_b = uni01(rng);

    if (!p.lattice_angle_candidates.empty()) {
        std::uniform_int_distribution<int> pick_ang(0, static_cast<int>(p.lattice_angle_candidates.size()) - 1);
        ind.angle_deg = p.lattice_angle_candidates[static_cast<size_t>(pick_ang(rng))];
    } else {
        ind.angle_deg = 0.0;
    }

    std::uniform_real_distribution<double> uni_sp(p.spacing_safety_min, p.spacing_safety_max);
    ind.spacing_safety = uni_sp(rng);

    return ind;
}

void mutate_individual(Individual& ind, std::mt19937_64& rng, const GAParams& p) {
    std::uniform_real_distribution<double> uni01(0.0, 1.0);

    if (uni01(rng) < p.p_mut_swap && ind.perm.size() >= 2) {
        std::uniform_int_distribution<int> pick(0, static_cast<int>(ind.perm.size()) - 1);
        int a = pick(rng);
        int b = pick(rng);
        std::swap(ind.perm[static_cast<size_t>(a)], ind.perm[static_cast<size_t>(b)]);
    }

    if (uni01(rng) < p.p_mut_scramble && ind.perm.size() >= 3) {
        std::uniform_int_distribution<int> pick(0, static_cast<int>(ind.perm.size()) - 1);
        int a = pick(rng);
        int b = pick(rng);
        if (a > b) {
            std::swap(a, b);
        }
        std::shuffle(ind.perm.begin() + a, ind.perm.begin() + b + 1, rng);
    }

    if (uni01(rng) < p.p_mut_rot && !ind.rot_idx.empty()) {
        std::uniform_int_distribution<int> pick_gene(0, static_cast<int>(ind.rot_idx.size()) - 1);
        std::uniform_int_distribution<int> pick_rot(0, static_cast<int>(p.rotation_candidates.size()) - 1);
        int idx = pick_gene(rng);
        ind.rot_idx[static_cast<size_t>(idx)] = pick_rot(rng);
    }

    if (uni01(rng) < p.p_mut_angle && !p.lattice_angle_candidates.empty()) {
        std::uniform_int_distribution<int> pick_ang(0, static_cast<int>(p.lattice_angle_candidates.size()) - 1);
        ind.angle_deg = p.lattice_angle_candidates[static_cast<size_t>(pick_ang(rng))];
    }

    if (uni01(rng) < p.p_mut_shift) {
        std::normal_distribution<double> normal(0.0, p.shift_sigma);
        ind.shift_a = wrap01(ind.shift_a + normal(rng));
        ind.shift_b = wrap01(ind.shift_b + normal(rng));
    }

    if (uni01(rng) < p.p_mut_spacing) {
        std::normal_distribution<double> normal(0.0, 0.002);
        ind.spacing_safety = std::clamp(ind.spacing_safety + normal(rng),
                                       p.spacing_safety_min,
                                       p.spacing_safety_max);
    }
}

Individual crossover_ox(const Individual& a, const Individual& b, std::mt19937_64& rng, const GAParams& p) {
    const int n = static_cast<int>(a.perm.size());
    Individual child = a;

    std::uniform_real_distribution<double> uni01(0.0, 1.0);
    if (uni01(rng) >= p.p_crossover || n <= 2) {
        return child;
    }

    std::uniform_int_distribution<int> pick(0, n - 1);
    int l = pick(rng);
    int r = pick(rng);
    if (l > r) {
        std::swap(l, r);
    }

    std::vector<int> out(static_cast<size_t>(n), -1);
    std::vector<char> used(static_cast<size_t>(n), 0);
    for (int i = l; i <= r; ++i) {
        int v = a.perm[static_cast<size_t>(i)];
        out[static_cast<size_t>(i)] = v;
        used[static_cast<size_t>(v)] = 1;
    }

    int write = (r + 1) % n;
    for (int k = 0; k < n; ++k) {
        int idx = (r + 1 + k) % n;
        int v = b.perm[static_cast<size_t>(idx)];
        if (used[static_cast<size_t>(v)]) {
            continue;
        }
        out[static_cast<size_t>(write)] = v;
        used[static_cast<size_t>(v)] = 1;
        write = (write + 1) % n;
    }

    child.perm = std::move(out);

    // Uniform crossover para rotações
    if (!child.rot_idx.empty()) {
        for (size_t i = 0; i < child.rot_idx.size(); ++i) {
            if (uni01(rng) < 0.5) {
                child.rot_idx[i] = b.rot_idx[i];
            }
        }
    }

    // Mistura simples de parâmetros globais
    if (uni01(rng) < 0.5) {
        child.angle_deg = b.angle_deg;
    }
    if (uni01(rng) < 0.5) {
        child.shift_a = b.shift_a;
        child.shift_b = b.shift_b;
    }
    if (uni01(rng) < 0.5) {
        child.spacing_safety = b.spacing_safety;
    }

    return child;
}

int tournament_pick(const std::vector<Individual>& pop, int k, std::mt19937_64& rng) {
    if (pop.empty()) {
        return -1;
    }
    k = std::max(1, std::min(k, static_cast<int>(pop.size())));
    std::uniform_int_distribution<int> pick(0, static_cast<int>(pop.size()) - 1);
    int best = pick(rng);
    for (int i = 1; i < k; ++i) {
        int cand = pick(rng);
        if (pop[static_cast<size_t>(cand)].fitness < pop[static_cast<size_t>(best)].fitness) {
            best = cand;
        }
    }
    return best;
}

}  // namespace

GlobalSearchGA::GlobalSearchGA(const Polygon& base_poly, double radius)
    : base_poly_(base_poly), radius_(radius) {}

GAResult GlobalSearchGA::solve(int n, uint64_t seed, const GAParams& p) const {
    if (n <= 0) {
        return GAResult{};
    }
    if (p.pop_size <= 0 || p.generations < 0) {
        throw std::runtime_error("GAParams inválidos (pop_size/generations).");
    }
    if (p.rotation_candidates.empty()) {
        throw std::runtime_error("GAParams.rotation_candidates vazio.");
    }
    if (!(p.spacing_safety_min >= 1.0) || !(p.spacing_safety_max >= p.spacing_safety_min)) {
        throw std::runtime_error("GAParams.spacing_safety_{min,max} inválidos.");
    }

    const double eps = 1e-9;
    const double min_spacing = find_min_safe_hex_spacing(base_poly_, radius_, eps);

    GAResult best;
    best.best_side = std::numeric_limits<double>::infinity();

    std::mt19937_64 rng(seed);

    std::vector<Individual> pop;
    pop.reserve(static_cast<size_t>(p.pop_size));
    for (int i = 0; i < p.pop_size; ++i) {
        pop.push_back(make_random_individual(n, rng, p));
    }

    auto evaluate = [&](Individual& ind) {
        const double spacing = min_spacing * ind.spacing_safety;
        const int candidate_target = std::max(p.candidate_min, p.candidate_factor * n);

        DecodeResult dec = decode_constructive(base_poly_,
                                              radius_,
                                              n,
                                              ind.perm,
                                              ind.rot_idx,
                                              p.rotation_candidates,
                                              spacing,
                                              ind.angle_deg,
                                              ind.shift_a,
                                              ind.shift_b,
                                              candidate_target);
        if (!dec.ok) {
            ind.fitness = std::numeric_limits<double>::infinity();
            ind.cached_poses.clear();
            return;
        }
        ind.fitness = dec.side;
        ind.cached_poses = std::move(dec.poses);

        if (ind.fitness + 1e-15 < best.best_side) {
            best.best_side = ind.fitness;
            best.best_poses = ind.cached_poses;
            best.best_spacing = spacing;
            best.best_angle_deg = ind.angle_deg;
            best.best_shift_a = ind.shift_a;
            best.best_shift_b = ind.shift_b;
        }
    };

    for (auto& ind : pop) {
        evaluate(ind);
    }

    for (int gen = 0; gen < p.generations; ++gen) {
        std::sort(pop.begin(), pop.end(), [](const Individual& a, const Individual& b) {
            return a.fitness < b.fitness;
        });

        std::vector<Individual> next;
        next.reserve(pop.size());

        int elite = std::max(0, std::min(p.elite, static_cast<int>(pop.size())));
        for (int i = 0; i < elite; ++i) {
            next.push_back(pop[static_cast<size_t>(i)]);
        }

        while (static_cast<int>(next.size()) < p.pop_size) {
            int ia = tournament_pick(pop, p.tournament_k, rng);
            int ib = tournament_pick(pop, p.tournament_k, rng);
            if (ia < 0 || ib < 0) {
                break;
            }
            if (ia == ib && pop.size() >= 2) {
                ib = (ib + 1) % static_cast<int>(pop.size());
            }

            Individual child = crossover_ox(pop[static_cast<size_t>(ia)],
                                            pop[static_cast<size_t>(ib)],
                                            rng,
                                            p);
            mutate_individual(child, rng, p);
            evaluate(child);
            next.push_back(std::move(child));
        }

        pop = std::move(next);
    }

    return best;
}
