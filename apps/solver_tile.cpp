#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "collision.hpp"
#include "boundary_refine.hpp"
#include "geom.hpp"
#include "prefix_prune.hpp"
#include "sa.hpp"
#include "solver_tile_cli.hpp"
#include "spatial_grid.hpp"
#include "submission_io.hpp"
#include "tiling_pool.hpp"
#include "wrap_utils.hpp"

namespace {

struct Candidate {
    TreePose pose;
    double key1;
    double key2;
};

struct Eval {
    double best_total = std::numeric_limits<double>::infinity();
    double best_angle = 0.0;
    std::vector<TreePose> best_poses;
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

Eval choose_best_angle_for_prefix_score(const Polygon& base_poly,
                                        const Pattern& pattern,
                                        double radius,
                                        double spacing,
                                        const Options& opt) {
    Eval best;
    for (double ang : opt.angle_candidates) {
        auto pool = opt.pool_window_scan
                        ? generate_windowed_tiling(opt.pool_size,
                                                   spacing,
                                                   ang,
                                                   pattern,
                                                   base_poly,
                                                   opt.pool_window_radius,
                                                   opt.n_max)
                        : generate_ordered_tiling(opt.pool_size, spacing, ang, pattern);
        if (any_overlap(base_poly, pool, radius)) {
            continue;
        }

        // A seleção do melhor ângulo precisa refletir exatamente o que vai pro
        // submission (valores quantizados em string). Caso contrário, o ranking
        // entre ângulos pode inverter após o arredondamento.
        auto pool_q = quantize_poses(pool);
        if (any_overlap(base_poly, pool_q, radius)) {
            continue;
        }

        std::vector<TreePose> prefix;
        if (opt.prefix_order == "greedy") {
            prefix = greedy_prefix_min_side(base_poly, pool_q, opt.n_max);
        } else if (opt.prefix_order == "central") {
            prefix = std::vector<TreePose>(pool_q.begin(), pool_q.begin() + opt.n_max);
        } else {
            throw std::runtime_error("--prefix-order inválido: " + opt.prefix_order);
        }

        std::vector<double> prefix_side_by_n =
            prefix_sides_from_bbs(bounding_boxes_for_poses(base_poly, prefix));

        double total = 0.0;
        if (opt.prune) {
            std::vector<double> prune_side_by_n = greedy_pruned_sides(
                bounding_boxes_for_poses(base_poly, pool_q), opt.n_max, 1e-12);
            for (int n = 1; n <= opt.n_max; ++n) {
                double s =
                    std::min(prefix_side_by_n[static_cast<size_t>(n)],
                             prune_side_by_n[static_cast<size_t>(n)]);
                total += (s * s) / static_cast<double>(n);
            }
        } else {
            total = total_score_from_sides(prefix_side_by_n, opt.n_max);
        }

        if (total < best.best_total) {
            best.best_total = total;
            best.best_angle = ang;
            best.best_poses = std::move(pool);
        }
    }
    if (!std::isfinite(best.best_total)) {
        throw std::runtime_error("Nenhum ângulo candidato gerou configuração válida.");
    }
    return best;
}

struct ShiftFullEval {
    double total = std::numeric_limits<double>::infinity();
    double best_angle = 0.0;
    bool ok = false;
};

double eval_shift_score_fast(const Polygon& base_poly,
                             const Pattern& pattern,
                             const Options& opt,
                             double spacing,
                             int pool_size,
                             int n_eval) {
    if (pool_size < n_eval) {
        pool_size = n_eval;
    }
    std::vector<TreePose> pool = opt.pool_window_scan
                                     ? generate_windowed_tiling(pool_size,
                                                                spacing,
                                                                0.0,
                                                                pattern,
                                                                base_poly,
                                                                opt.pool_window_radius,
                                                                n_eval)
                                     : generate_ordered_tiling(pool_size, spacing, 0.0, pattern);
    std::vector<TreePose> prefix(pool.begin(), pool.begin() + n_eval);
    std::vector<double> side_by_n =
        prefix_sides_from_bbs(bounding_boxes_for_poses(base_poly, prefix));
    return total_score_from_sides(side_by_n, n_eval);
}

ShiftFullEval eval_shift_score_full(const Polygon& base_poly,
                                    const Pattern& pattern,
                                    double radius,
                                    double spacing,
                                    const Options& opt,
                                    int pool_size,
                                    int n_eval) {
    ShiftFullEval best;
    pool_size = std::max(pool_size, n_eval);

    for (double ang : opt.angle_candidates) {
        std::vector<TreePose> pool = opt.pool_window_scan
                                         ? generate_windowed_tiling(pool_size,
                                                                    spacing,
                                                                    ang,
                                                                    pattern,
                                                                    base_poly,
                                                                    opt.pool_window_radius,
                                                                    n_eval)
                                         : generate_ordered_tiling(pool_size, spacing, ang, pattern);
        if (any_overlap(base_poly, pool, radius)) {
            continue;
        }
        auto pool_q = quantize_poses(pool);
        if (any_overlap(base_poly, pool_q, radius)) {
            continue;
        }

        std::vector<TreePose> prefix_nmax;
        if (opt.prefix_order == "greedy") {
            prefix_nmax = greedy_prefix_min_side(base_poly, pool_q, n_eval);
        } else if (opt.prefix_order == "central") {
            prefix_nmax = std::vector<TreePose>(pool_q.begin(), pool_q.begin() + n_eval);
        } else {
            throw std::runtime_error("--prefix-order inválido: " + opt.prefix_order);
        }

        std::vector<double> prefix_side_by_n =
            prefix_sides_from_bbs(bounding_boxes_for_poses(base_poly, prefix_nmax));

        double total = 0.0;
        if (opt.prune) {
            std::vector<double> prune_side_by_n =
                greedy_pruned_sides(bounding_boxes_for_poses(base_poly, pool_q), n_eval, 1e-12);
            total = total_score_min_sides(prefix_side_by_n, &prune_side_by_n, n_eval, best.total);
        } else {
            total = total_score_min_sides(prefix_side_by_n, nullptr, n_eval, best.total);
        }

        if (total < best.total) {
            best.total = total;
            best.best_angle = ang;
            best.ok = true;
        }
    }

    return best;
}

struct ShiftSearchResult {
    double shift_a = 0.0;
    double shift_b = 0.0;
    double best_angle = 0.0;
    double best_total = std::numeric_limits<double>::infinity();
    bool ok = false;
};

ShiftSearchResult shift_search_multires(const Polygon& base_poly,
                                        const Pattern& base_pattern,
                                        double radius,
                                        double spacing,
                                        const Options& opt) {
    ShiftSearchResult res;

    const int n_eval = std::min(opt.tile_score_nmax, opt.n_max);
    int pool_fast = opt.tile_score_pool_size > 0 ? opt.tile_score_pool_size : opt.pool_size;
    pool_fast = std::max(pool_fast, n_eval);

    int pool_full = opt.shift_pool_size > 0 ? opt.shift_pool_size : opt.pool_size;
    pool_full = std::max(pool_full, opt.n_max);

    const int grid = opt.shift_grid;
    const int keep = opt.shift_keep;
    const int levels = opt.shift_levels;

    struct Cand {
        double a = 0.0;
        double b = 0.0;
        double fast = std::numeric_limits<double>::infinity();
    };

    auto eval_fast = [&](double a, double b) -> double {
        Pattern p = base_pattern;
        p.shift_a = wrap01(a);
        p.shift_b = wrap01(b);
        return eval_shift_score_fast(base_poly, p, opt, spacing, pool_fast, n_eval);
    };

    auto keep_top = [&](std::vector<Cand>& cands) {
        std::sort(cands.begin(), cands.end(), [](const Cand& x, const Cand& y) {
            if (x.fast != y.fast) {
                return x.fast < y.fast;
            }
            if (x.a != y.a) {
                return x.a < y.a;
            }
            return x.b < y.b;
        });
        if (static_cast<int>(cands.size()) > keep) {
            cands.resize(static_cast<size_t>(keep));
        }
    };

    std::vector<Cand> frontier;
    frontier.reserve(static_cast<size_t>(grid * grid));
    for (int ia = 0; ia < grid; ++ia) {
        for (int ib = 0; ib < grid; ++ib) {
            double a = (static_cast<double>(ia) + 0.5) / static_cast<double>(grid);
            double b = (static_cast<double>(ib) + 0.5) / static_cast<double>(grid);
            frontier.push_back(Cand{wrap01(a), wrap01(b), eval_fast(a, b)});
        }
    }
    keep_top(frontier);

    double step = 1.0 / static_cast<double>(grid);
    for (int lvl = 1; lvl < levels; ++lvl) {
        step *= 0.5;
        std::vector<Cand> next;
        next.reserve(frontier.size() * 9);

        for (const auto& c : frontier) {
            for (int da = -1; da <= 1; ++da) {
                for (int db = -1; db <= 1; ++db) {
                    double a = wrap01(c.a + static_cast<double>(da) * step);
                    double b = wrap01(c.b + static_cast<double>(db) * step);
                    next.push_back(Cand{a, b, eval_fast(a, b)});
                }
            }
        }

        std::sort(next.begin(), next.end(), [](const Cand& x, const Cand& y) {
            if (x.a != y.a) {
                return x.a < y.a;
            }
            return x.b < y.b;
        });
        next.erase(std::unique(next.begin(),
                               next.end(),
                               [](const Cand& x, const Cand& y) {
                                   return std::abs(x.a - y.a) <= 1e-15 &&
                                          std::abs(x.b - y.b) <= 1e-15;
                               }),
                   next.end());

        frontier = std::move(next);
        keep_top(frontier);
    }

    for (const auto& c : frontier) {
        Pattern p = base_pattern;
        p.shift_a = c.a;
        p.shift_b = c.b;
        ShiftFullEval full = eval_shift_score_full(
            base_poly, p, radius, spacing, opt, pool_full, opt.n_max);
        if (!full.ok) {
            continue;
        }
        if (full.total < res.best_total) {
            res.best_total = full.total;
            res.shift_a = c.a;
            res.shift_b = c.b;
            res.best_angle = full.best_angle;
            res.ok = true;
        }
    }

    return res;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);

        Polygon base_poly = get_tree_polygon();
        const double radius = enclosing_circle_radius(base_poly);
        const double eps = 1e-9;

        Pattern pattern = make_initial_pattern(opt.k);
        pattern.lattice_v_ratio = opt.lattice_v_ratio;
        pattern.lattice_theta_deg = opt.lattice_theta_deg;
        pattern = optimize_tile_by_spacing(base_poly, pattern, radius, opt);
        pattern.shift_a = wrap01(opt.shift_a);
        pattern.shift_b = wrap01(opt.shift_b);

        double min_spacing =
            find_min_safe_spacing(base_poly, pattern, radius, eps, 2.0 * radius);
        if (!std::isfinite(min_spacing)) {
            throw std::runtime_error("Não foi possível encontrar spacing seguro.");
        }
        const double spacing = min_spacing * opt.spacing_safety;

        if (opt.shift_search == ShiftSearchMode::kMultires) {
            ShiftSearchResult ss =
                shift_search_multires(base_poly, pattern, radius, spacing, opt);
            if (!ss.ok) {
                throw std::runtime_error("Shift search falhou: nenhum candidato válido.");
            }
            pattern.shift_a = ss.shift_a;
            pattern.shift_b = ss.shift_b;
            std::cout << "Shift search: a=" << std::fixed << std::setprecision(9)
                      << pattern.shift_a << " b=" << pattern.shift_b
                      << " (score=" << std::setprecision(9) << ss.best_total
                      << ", angle=" << std::setprecision(3) << ss.best_angle << ")\n";
        }

        Eval chosen = choose_best_angle_for_prefix_score(
            base_poly, pattern, radius, spacing, opt);

        std::vector<TreePose> poses_pool = std::move(chosen.best_poses);

        refine_boundary(base_poly, radius, poses_pool, opt.refine_iters, opt.seed + 999, spacing);

        // Reordena por "centralidade" após o refino (melhora recortes n pequenos).
        {
            std::vector<Candidate> tmp;
            tmp.reserve(poses_pool.size());
            for (const auto& pose : poses_pool) {
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
            for (size_t i = 0; i < poses_pool.size(); ++i) {
                poses_pool[i] = tmp[i].pose;
            }
        }

        if (any_overlap(base_poly, poses_pool, radius)) {
            throw std::runtime_error("Overlap detectado após refino.");
        }

        auto poses_pool_q = quantize_poses(poses_pool);
        if (any_overlap(base_poly, poses_pool_q, radius)) {
            throw std::runtime_error(
                "Overlap detectado após arredondamento para o submission.");
        }

        std::vector<TreePose> prefix_nmax;
        if (opt.prefix_order == "greedy") {
            prefix_nmax = greedy_prefix_min_side(base_poly, poses_pool_q, opt.n_max);
        } else {
            prefix_nmax = std::vector<TreePose>(poses_pool_q.begin(),
                                                poses_pool_q.begin() + opt.n_max);
        }

        std::vector<std::vector<TreePose>> prefix_by_n;
        prefix_by_n.resize(static_cast<size_t>(opt.n_max + 1));
        for (int n = 1; n <= opt.n_max; ++n) {
            prefix_by_n[static_cast<size_t>(n)] =
                std::vector<TreePose>(prefix_nmax.begin(), prefix_nmax.begin() + n);
        }

        std::vector<double> prefix_side_by_n =
            prefix_sides_from_bbs(bounding_boxes_for_poses(base_poly, prefix_nmax));

        std::vector<std::vector<TreePose>> solutions_by_n;
        solutions_by_n.resize(static_cast<size_t>(opt.n_max + 1));

        double total_score = 0.0;
        if (opt.prune) {
            PruneResult pr =
                build_greedy_pruned_solutions(base_poly, poses_pool_q, opt.n_max);
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

        if (opt.mz_its_iters > 0) {
            std::vector<TreePose>& sol_nmax =
                solutions_by_n[static_cast<size_t>(opt.n_max)];
            double side_before = bounding_square_side(
                transformed_polygons(base_poly, sol_nmax));
            ILSResult mz = mz_its_soft_compact(
                base_poly,
                radius,
                sol_nmax,
                opt.seed ^
                    (0xC6BC279692B5CC83ULL +
                     static_cast<uint64_t>(opt.n_max) * 0x9E3779B97F4A7C15ULL),
                opt);
            double side_after = bounding_square_side(
                transformed_polygons(base_poly, sol_nmax));
            if (mz.improved) {
                total_score +=
                    ((side_after * side_after) - (side_before * side_before)) /
                    static_cast<double>(opt.n_max);
            }
            if (any_overlap(base_poly, sol_nmax, radius)) {
                throw std::runtime_error("Overlap detectado após MZ-ITS.");
            }
            std::cout << "MZ-ITS (n=" << opt.n_max << "): "
                      << (mz.improved ? "improved" : "no-improve")
                      << " (attempts=" << mz.attempts
                      << ", accepted=" << mz.accepted
                      << ", side=" << std::fixed << std::setprecision(9)
                      << mz.start_side << " -> " << mz.best_side
                      << ")\n";
        }

        if (opt.ils_iters > 0) {
            std::vector<TreePose>& sol_nmax =
                solutions_by_n[static_cast<size_t>(opt.n_max)];
            double side_before = bounding_square_side(
                transformed_polygons(base_poly, sol_nmax));
            ILSResult ils = ils_basin_hop_compact(
                base_poly,
                radius,
                sol_nmax,
                opt.seed ^
                    (0xA24BAED4963EE407ULL +
                     static_cast<uint64_t>(opt.n_max) * 0x9E3779B97F4A7C15ULL),
                opt);
            double side_after = bounding_square_side(
                transformed_polygons(base_poly, sol_nmax));
            if (ils.improved) {
                total_score +=
                    ((side_after * side_after) - (side_before * side_before)) /
                    static_cast<double>(opt.n_max);
            }
            std::cout << "ILS (n=" << opt.n_max << "): "
                      << (ils.improved ? "improved" : "no-improve")
                      << " (attempts=" << ils.attempts
                      << ", accepted=" << ils.accepted
                      << ", side=" << std::fixed << std::setprecision(9)
                      << ils.start_side << " -> " << ils.best_side
                      << ")\n";
        }

        const bool use_beam = opt.sa_beam && opt.sa_beam_width > 0;
        const bool use_chain = opt.sa_chain &&
                               (opt.sa_chain_base_iters > 0 ||
                                opt.sa_chain_iters_per_n > 0);
        if (use_beam || use_chain) {
            std::vector<double> base_side_by_n;
            base_side_by_n.resize(static_cast<size_t>(opt.n_max + 1), 0.0);
            for (int n = 1; n <= opt.n_max; ++n) {
                base_side_by_n[static_cast<size_t>(n)] =
                    bounding_square_side(transformed_polygons(
                        base_poly, solutions_by_n[static_cast<size_t>(n)]));
            }

            const double band_step = spacing * std::min(1.0, pattern.lattice_v_ratio);
            ChainResult cr = use_beam
                                 ? build_sa_beam_chain_solutions(base_poly,
                                                                 radius,
                                                                 solutions_by_n[static_cast<size_t>(opt.n_max)],
                                                                 opt.n_max,
                                                                 band_step,
                                                                 opt)
                                 : build_sa_chain_solutions(base_poly,
                                                            radius,
                                                            solutions_by_n[static_cast<size_t>(opt.n_max)],
                                                            opt.n_max,
                                                            band_step,
                                                            opt);

            double total_score_chain = 0.0;
            for (int n = 1; n <= opt.n_max; ++n) {
                double best_side = base_side_by_n[static_cast<size_t>(n)];
                std::vector<TreePose> best_sol = solutions_by_n[static_cast<size_t>(n)];

                if (!cr.solutions_by_n[static_cast<size_t>(n)].empty() &&
                    cr.side_by_n[static_cast<size_t>(n)] + 1e-15 < best_side) {
                    best_side = cr.side_by_n[static_cast<size_t>(n)];
                    best_sol = std::move(cr.solutions_by_n[static_cast<size_t>(n)]);
                }

                solutions_by_n[static_cast<size_t>(n)] = std::move(best_sol);
                total_score_chain += (best_side * best_side) / static_cast<double>(n);
            }
            total_score = total_score_chain;
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
                p.w_push_contact = opt.sa_w_push_contact;
                p.w_squeeze = opt.sa_w_squeeze;
                p.block_size = opt.sa_block_size;
                p.lns_remove = opt.sa_lns_remove;
                p.lns_candidates = opt.sa_lns_candidates;
                p.lns_eval_attempts_per_tree = opt.sa_lns_eval_attempts_per_tree;
                p.hh_segment = opt.sa_hh_segment;
                p.hh_reaction = opt.sa_hh_reaction;
                p.overlap_metric = opt.sa_overlap_metric;
                p.overlap_weight = opt.sa_overlap_weight;
                p.overlap_weight_start = opt.sa_overlap_weight_start;
                p.overlap_weight_end = opt.sa_overlap_weight_end;
                p.overlap_weight_power = opt.sa_overlap_weight_power;
                p.overlap_weight_geometric = opt.sa_overlap_weight_geometric;
                p.overlap_eps_area = opt.sa_overlap_eps_area;
                p.overlap_cost_cap = opt.sa_overlap_cost_cap;
                p.plateau_eps = opt.sa_plateau_eps;
                p.w_resolve_overlap = opt.sa_w_resolve_overlap;
                p.resolve_attempts = opt.sa_resolve_attempts;
                p.resolve_step_frac_max = opt.sa_resolve_step_frac_max;
                p.resolve_step_frac_min = opt.sa_resolve_step_frac_min;
                p.resolve_noise_frac = opt.sa_resolve_noise_frac;
                p.push_max_step_frac = opt.sa_push_max_step_frac;
                p.push_bisect_iters = opt.sa_push_bisect_iters;
                p.push_overshoot_frac = opt.sa_push_overshoot_frac;
                p.squeeze_pushes = opt.sa_squeeze_pushes;
                if (opt.sa_aggressive) {
                    SARefiner::apply_aggressive_preset(p);
                }

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
        std::cout << "Pool size: " << opt.pool_size << "\n";
        std::cout << "Pool window: " << (opt.pool_window_scan ? "on" : "off");
        if (opt.pool_window_scan) {
            std::cout << " (radius=" << opt.pool_window_radius << ")";
        }
        std::cout << "\n";
        std::cout << "Prefix order: " << opt.prefix_order << "\n";
        std::cout << "Tile iters: " << opt.tile_iters << "\n";
        std::cout << "Refine iters: " << opt.refine_iters << "\n";
        std::cout << "Lattice v_ratio: " << std::fixed << std::setprecision(6)
                  << pattern.lattice_v_ratio << "\n";
        std::cout << "Lattice theta: " << std::fixed << std::setprecision(3)
                  << pattern.lattice_theta_deg << "\n";
        std::cout << "Min spacing: " << std::fixed << std::setprecision(9) << min_spacing << "\n";
        std::cout << "Spacing (safety): " << std::fixed << std::setprecision(9) << spacing << "\n";
        std::cout << "Best angle: " << std::fixed << std::setprecision(3) << chosen.best_angle << "\n";
        std::cout << "Score (local): " << std::fixed << std::setprecision(9) << total_score << "\n";
        std::cout << "Prune: " << (opt.prune ? "on" : "off") << "\n";
        std::cout << "Final rigid: " << (opt.final_rigid ? "on" : "off") << "\n";
        if (opt.sa_beam) {
            std::cout << "SA beam: on (width=" << opt.sa_beam_width
                      << ", remove=" << opt.sa_beam_remove
                      << ", micro=" << opt.sa_beam_micro_iters
                      << ", init=" << opt.sa_beam_init_iters
                      << ", band_layers=" << std::fixed << std::setprecision(3)
                      << opt.sa_chain_band_layers << ")\n";
        } else {
            std::cout << "SA beam: off\n";
        }
        if (opt.sa_chain && !opt.sa_beam) {
            std::cout << "SA chain: on (base=" << opt.sa_chain_base_iters
                      << ", per_n=" << opt.sa_chain_iters_per_n
                      << ", min_n=" << opt.sa_chain_min_n
                      << ", band_layers=" << std::fixed << std::setprecision(3)
                      << opt.sa_chain_band_layers << ")\n";
        } else if (!opt.sa_beam) {
            std::cout << "SA chain: off\n";
        }
        std::cout << "SA aggressive: " << (opt.sa_aggressive ? "on" : "off") << "\n";

    } catch (const std::exception& ex) {
        std::cerr << "Erro: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
