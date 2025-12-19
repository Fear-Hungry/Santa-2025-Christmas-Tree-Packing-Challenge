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
#include "boundary_refine.hpp"
#include "geom.hpp"
#include "prefix_prune.hpp"
#include "sa.hpp"
#include "solver_tile_options.hpp"
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
    auto parse_overlap_metric = [](const std::string& s) -> SARefiner::OverlapMetric {
        if (s == "area") {
            return SARefiner::OverlapMetric::kArea;
        }
        if (s == "mtv2" || s == "mtv") {
            return SARefiner::OverlapMetric::kMtv2;
        }
        throw std::runtime_error("--sa-overlap-metric precisa ser 'area' ou 'mtv2'.");
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
        } else if (arg == "--pool-size") {
            opt.pool_size = parse_int(need(arg));
        } else if (arg == "--pool-window") {
            opt.pool_window_scan = true;
        } else if (arg == "--pool-window-radius") {
            opt.pool_window_radius = parse_int(need(arg));
        } else if (arg == "--prefix-order") {
            opt.prefix_order = need(arg);
        } else if (arg == "--tile-iters") {
            opt.tile_iters = parse_int(need(arg));
        } else if (arg == "--tile-obj") {
            std::string s = need(arg);
            if (s == "density") {
                opt.tile_obj = TileObjective::kDensity;
            } else if (s == "score") {
                opt.tile_obj = TileObjective::kScore;
            } else {
                throw std::runtime_error("--tile-obj precisa ser 'density' ou 'score'.");
            }
        } else if (arg == "--tile-score-pool-size") {
            opt.tile_score_pool_size = parse_int(need(arg));
        } else if (arg == "--tile-score-nmax") {
            opt.tile_score_nmax = parse_int(need(arg));
        } else if (arg == "--tile-score-full-every") {
            opt.tile_score_full_every = parse_int(need(arg));
        } else if (arg == "--no-tile-opt-lattice") {
            opt.tile_opt_lattice = false;
        } else if (arg == "--lattice-v-ratio") {
            opt.lattice_v_ratio = parse_double(need(arg));
        } else if (arg == "--lattice-theta") {
            opt.lattice_theta_deg = parse_double(need(arg));
        } else if (arg == "--refine-iters") {
            opt.refine_iters = parse_int(need(arg));
        } else if (arg == "--sa-restarts") {
            opt.sa_restarts = parse_int(need(arg));
        } else if (arg == "--sa-base-iters") {
            opt.sa_base_iters = parse_int(need(arg));
        } else if (arg == "--sa-iters-per-n") {
            opt.sa_iters_per_n = parse_int(need(arg));
        } else if (arg == "--sa-chain") {
            opt.sa_chain = true;
        } else if (arg == "--sa-beam") {
            opt.sa_beam = true;
            if (opt.sa_beam_width <= 0) {
                opt.sa_beam_width = 16;
            }
        } else if (arg == "--sa-beam-width") {
            opt.sa_beam_width = parse_int(need(arg));
            opt.sa_beam = (opt.sa_beam_width > 0);
        } else if (arg == "--sa-beam-remove") {
            opt.sa_beam_remove = parse_int(need(arg));
        } else if (arg == "--sa-beam-micro-iters") {
            opt.sa_beam_micro_iters = parse_int(need(arg));
        } else if (arg == "--sa-beam-init-iters") {
            opt.sa_beam_init_iters = parse_int(need(arg));
        } else if (arg == "--sa-chain-base-iters") {
            opt.sa_chain_base_iters = parse_int(need(arg));
        } else if (arg == "--sa-chain-iters-per-n") {
            opt.sa_chain_iters_per_n = parse_int(need(arg));
        } else if (arg == "--sa-chain-min-n") {
            opt.sa_chain_min_n = parse_int(need(arg));
        } else if (arg == "--sa-chain-band-layers") {
            opt.sa_chain_band_layers = parse_double(need(arg));
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
        } else if (arg == "--sa-w-push-contact") {
            opt.sa_w_push_contact = parse_double(need(arg));
        } else if (arg == "--sa-w-squeeze") {
            opt.sa_w_squeeze = parse_double(need(arg));
        } else if (arg == "--sa-block-size") {
            opt.sa_block_size = parse_int(need(arg));
        } else if (arg == "--sa-lns-remove") {
            opt.sa_lns_remove = parse_int(need(arg));
        } else if (arg == "--sa-hh-segment") {
            opt.sa_hh_segment = parse_int(need(arg));
        } else if (arg == "--sa-hh-reaction") {
            opt.sa_hh_reaction = parse_double(need(arg));
        } else if (arg == "--sa-overlap-metric") {
            opt.sa_overlap_metric = parse_overlap_metric(need(arg));
        } else if (arg == "--sa-overlap-weight") {
            opt.sa_overlap_weight = parse_double(need(arg));
        } else if (arg == "--sa-overlap-weight-start") {
            opt.sa_overlap_weight_start = parse_double(need(arg));
        } else if (arg == "--sa-overlap-weight-end") {
            opt.sa_overlap_weight_end = parse_double(need(arg));
        } else if (arg == "--sa-overlap-weight-power") {
            opt.sa_overlap_weight_power = parse_double(need(arg));
        } else if (arg == "--sa-overlap-eps-area") {
            opt.sa_overlap_eps_area = parse_double(need(arg));
        } else if (arg == "--sa-overlap-cost-cap") {
            opt.sa_overlap_cost_cap = parse_double(need(arg));
        } else if (arg == "--sa-plateau-eps") {
            opt.sa_plateau_eps = parse_double(need(arg));
        } else if (arg == "--sa-w-resolve-overlap") {
            opt.sa_w_resolve_overlap = parse_double(need(arg));
        } else if (arg == "--sa-resolve-attempts") {
            opt.sa_resolve_attempts = parse_int(need(arg));
        } else if (arg == "--sa-resolve-step-frac-max") {
            opt.sa_resolve_step_frac_max = parse_double(need(arg));
        } else if (arg == "--sa-resolve-step-frac-min") {
            opt.sa_resolve_step_frac_min = parse_double(need(arg));
        } else if (arg == "--sa-resolve-noise-frac") {
            opt.sa_resolve_noise_frac = parse_double(need(arg));
        } else if (arg == "--sa-push-max-step-frac") {
            opt.sa_push_max_step_frac = parse_double(need(arg));
        } else if (arg == "--sa-push-bisect-iters") {
            opt.sa_push_bisect_iters = parse_int(need(arg));
        } else if (arg == "--sa-push-overshoot-frac") {
            opt.sa_push_overshoot_frac = parse_double(need(arg));
        } else if (arg == "--sa-squeeze-pushes") {
            opt.sa_squeeze_pushes = parse_int(need(arg));
        } else if (arg == "--sa-aggressive") {
            opt.sa_aggressive = true;
        } else if (arg == "--ils-iters") {
            opt.ils_iters = parse_int(need(arg));
        } else if (arg == "--ils-sa-restarts") {
            opt.ils_sa_restarts = parse_int(need(arg));
        } else if (arg == "--ils-sa-iters") {
            opt.ils_sa_iters = parse_int(need(arg));
        } else if (arg == "--ils-accept-sa") {
            opt.ils_accept_sa = true;
        } else if (arg == "--ils-t0") {
            opt.ils_t0 = parse_double(need(arg));
        } else if (arg == "--ils-t1") {
            opt.ils_t1 = parse_double(need(arg));
        } else if (arg == "--ils-alpha-min") {
            opt.ils_alpha_min = parse_double(need(arg));
        } else if (arg == "--ils-alpha-max") {
            opt.ils_alpha_max = parse_double(need(arg));
        } else if (arg == "--ils-p-aniso") {
            opt.ils_p_aniso = parse_double(need(arg));
        } else if (arg == "--ils-shear-max") {
            opt.ils_shear_max = parse_double(need(arg));
        } else if (arg == "--ils-jitter-frac") {
            opt.ils_jitter_frac = parse_double(need(arg));
        } else if (arg == "--ils-subset-frac") {
            opt.ils_subset_frac = parse_double(need(arg));
        } else if (arg == "--ils-rot-prob") {
            opt.ils_rot_prob = parse_double(need(arg));
        } else if (arg == "--ils-rot-deg-max") {
            opt.ils_rot_deg_max = parse_double(need(arg));
        } else if (arg == "--ils-repair-mtv-passes") {
            opt.ils_repair_mtv_passes = parse_int(need(arg));
        } else if (arg == "--ils-repair-mtv-damping") {
            opt.ils_repair_mtv_damping = parse_double(need(arg));
        } else if (arg == "--ils-repair-mtv-split") {
            opt.ils_repair_mtv_split = parse_double(need(arg));
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
        } else if (arg == "--shift-search") {
            std::string s = need(arg);
            if (s == "off") {
                opt.shift_search = ShiftSearchMode::kOff;
            } else if (s == "multires") {
                opt.shift_search = ShiftSearchMode::kMultires;
            } else {
                throw std::runtime_error("--shift-search precisa ser 'off' ou 'multires'.");
            }
        } else if (arg == "--shift-grid") {
            opt.shift_grid = parse_int(need(arg));
        } else if (arg == "--shift-levels") {
            opt.shift_levels = parse_int(need(arg));
        } else if (arg == "--shift-keep") {
            opt.shift_keep = parse_int(need(arg));
        } else if (arg == "--shift-pool-size") {
            opt.shift_pool_size = parse_int(need(arg));
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
    if (opt.pool_size <= 0) {
        throw std::runtime_error("--pool-size precisa ser > 0.");
    }
    if (opt.pool_size < opt.n_max) {
        throw std::runtime_error("--pool-size precisa ser >= 200.");
    }
    if (opt.pool_window_radius < 0) {
        throw std::runtime_error("--pool-window-radius precisa ser >= 0.");
    }
    if (opt.prefix_order != "central" && opt.prefix_order != "greedy") {
        throw std::runtime_error("--prefix-order precisa ser 'central' ou 'greedy'.");
    }
    if (opt.tile_score_pool_size < 0) {
        throw std::runtime_error("--tile-score-pool-size precisa ser >= 0.");
    }
    if (opt.tile_score_nmax <= 0 || opt.tile_score_nmax > opt.n_max) {
        throw std::runtime_error("--tile-score-nmax precisa estar em [1, 200].");
    }
    if (opt.tile_score_full_every <= 0) {
        throw std::runtime_error("--tile-score-full-every precisa ser > 0.");
    }
    if (!(opt.lattice_v_ratio > 0.0)) {
        throw std::runtime_error("--lattice-v-ratio precisa ser > 0.");
    }
    if (opt.lattice_v_ratio < 0.50 || opt.lattice_v_ratio > 2.00) {
        throw std::runtime_error("--lattice-v-ratio precisa estar em [0.50, 2.00].");
    }
    if (opt.lattice_theta_deg < 20.0 || opt.lattice_theta_deg > 160.0) {
        throw std::runtime_error("--lattice-theta precisa estar em [20, 160] (evita lattice degenerado).");
    }
    if (!(opt.spacing_safety >= 1.0)) {
        throw std::runtime_error("--spacing-safety precisa ser >= 1.0.");
    }
    if (opt.shift_grid <= 0) {
        throw std::runtime_error("--shift-grid precisa ser > 0.");
    }
    if (opt.shift_levels <= 0) {
        throw std::runtime_error("--shift-levels precisa ser > 0.");
    }
    if (opt.shift_keep <= 0) {
        throw std::runtime_error("--shift-keep precisa ser > 0.");
    }
    if (opt.shift_pool_size < 0) {
        throw std::runtime_error("--shift-pool-size precisa ser >= 0.");
    }
    if (opt.sa_chain_base_iters < 0 || opt.sa_chain_iters_per_n < 0) {
        throw std::runtime_error("--sa-chain-base-iters/--sa-chain-iters-per-n precisam ser >= 0.");
    }
    if (opt.sa_chain) {
        if (opt.sa_chain_min_n < 1 || opt.sa_chain_min_n > opt.n_max) {
            throw std::runtime_error("--sa-chain-min-n precisa estar em [1, 200].");
        }
        if (!(opt.sa_chain_band_layers > 0.0)) {
            throw std::runtime_error("--sa-chain-band-layers precisa ser > 0.");
        }
        if (opt.sa_chain_band_layers > 20.0) {
            throw std::runtime_error("--sa-chain-band-layers muito alto (use <= 20).");
        }
    }
    if (opt.sa_beam) {
        if (opt.sa_beam_width <= 0) {
            throw std::runtime_error("--sa-beam-width precisa ser > 0.");
        }
        if (opt.sa_beam_remove <= 0) {
            throw std::runtime_error("--sa-beam-remove precisa ser > 0.");
        }
        if (opt.sa_beam_micro_iters < 0) {
            throw std::runtime_error("--sa-beam-micro-iters precisa ser >= 0.");
        }
        if (opt.sa_beam_init_iters < 0) {
            throw std::runtime_error("--sa-beam-init-iters precisa ser >= 0.");
        }
    }
    if (opt.sa_w_micro < 0.0 || opt.sa_w_swap_rot < 0.0 || opt.sa_w_relocate < 0.0 ||
        opt.sa_w_block_translate < 0.0 || opt.sa_w_block_rotate < 0.0 || opt.sa_w_lns < 0.0 ||
        opt.sa_w_resolve_overlap < 0.0 || opt.sa_w_push_contact < 0.0 || opt.sa_w_squeeze < 0.0) {
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
    if (!(opt.sa_overlap_weight >= 0.0)) {
        throw std::runtime_error("--sa-overlap-weight precisa ser >= 0.");
    }
    if (!(opt.sa_overlap_weight_power > 0.0)) {
        throw std::runtime_error("--sa-overlap-weight-power precisa ser > 0.");
    }
    if (!(opt.sa_overlap_eps_area >= 0.0)) {
        throw std::runtime_error("--sa-overlap-eps-area precisa ser >= 0.");
    }
    if (!(opt.sa_overlap_cost_cap >= 0.0)) {
        throw std::runtime_error("--sa-overlap-cost-cap precisa ser >= 0.");
    }
    if (!(opt.sa_plateau_eps >= 0.0)) {
        throw std::runtime_error("--sa-plateau-eps precisa ser >= 0.");
    }
    if (opt.sa_resolve_attempts <= 0) {
        throw std::runtime_error("--sa-resolve-attempts precisa ser > 0.");
    }
    if (!(opt.sa_resolve_step_frac_max > 0.0) || !(opt.sa_resolve_step_frac_min > 0.0) ||
        opt.sa_resolve_step_frac_min > opt.sa_resolve_step_frac_max) {
        throw std::runtime_error("--sa-resolve-step-frac-min/max inválidos.");
    }
    if (!(opt.sa_resolve_noise_frac >= 0.0)) {
        throw std::runtime_error("--sa-resolve-noise-frac precisa ser >= 0.");
    }
    if (!(opt.sa_push_max_step_frac > 0.0)) {
        throw std::runtime_error("--sa-push-max-step-frac precisa ser > 0.");
    }
    if (opt.sa_push_bisect_iters <= 0) {
        throw std::runtime_error("--sa-push-bisect-iters precisa ser > 0.");
    }
    if (opt.sa_push_overshoot_frac < 0.0 || opt.sa_push_overshoot_frac > 1.0) {
        throw std::runtime_error("--sa-push-overshoot-frac precisa estar em [0, 1].");
    }
    if (opt.sa_squeeze_pushes < 0) {
        throw std::runtime_error("--sa-squeeze-pushes precisa ser >= 0.");
    }
    if (opt.ils_iters < 0) {
        throw std::runtime_error("--ils-iters precisa ser >= 0.");
    }
    if (opt.ils_sa_restarts < 0) {
        throw std::runtime_error("--ils-sa-restarts precisa ser >= 0.");
    }
    if (opt.ils_sa_iters < 0) {
        throw std::runtime_error("--ils-sa-iters precisa ser >= 0.");
    }
    if (opt.ils_accept_sa) {
        if (!(opt.ils_t0 > 0.0) || !(opt.ils_t1 > 0.0) || opt.ils_t1 > opt.ils_t0) {
            throw std::runtime_error("--ils-t0/--ils-t1 inválidos (0 < t1 <= t0).");
        }
    }
    if (!(opt.ils_alpha_min > 0.0) || !(opt.ils_alpha_max > 0.0) ||
        opt.ils_alpha_min > opt.ils_alpha_max || opt.ils_alpha_max > 1.0) {
        throw std::runtime_error("--ils-alpha-min/max inválidos (0 < min <= max <= 1).");
    }
    if (opt.ils_p_aniso < 0.0 || opt.ils_p_aniso > 1.0) {
        throw std::runtime_error("--ils-p-aniso precisa estar em [0,1].");
    }
    if (opt.ils_shear_max < 0.0 || opt.ils_shear_max > 0.50) {
        throw std::runtime_error("--ils-shear-max precisa estar em [0,0.50].");
    }
    if (opt.ils_jitter_frac < 0.0 || opt.ils_jitter_frac > 0.50) {
        throw std::runtime_error("--ils-jitter-frac precisa estar em [0,0.50].");
    }
    if (opt.ils_subset_frac < 0.0 || opt.ils_subset_frac > 1.0) {
        throw std::runtime_error("--ils-subset-frac precisa estar em [0,1].");
    }
    if (opt.ils_rot_prob < 0.0 || opt.ils_rot_prob > 1.0) {
        throw std::runtime_error("--ils-rot-prob precisa estar em [0,1].");
    }
    if (!(opt.ils_rot_deg_max >= 0.0)) {
        throw std::runtime_error("--ils-rot-deg-max precisa ser >= 0.");
    }
    if (opt.ils_repair_mtv_passes < 0) {
        throw std::runtime_error("--ils-repair-mtv-passes precisa ser >= 0.");
    }
    if (!(opt.ils_repair_mtv_damping > 0.0)) {
        throw std::runtime_error("--ils-repair-mtv-damping precisa ser > 0.");
    }
    if (opt.ils_repair_mtv_split < 0.0 || opt.ils_repair_mtv_split > 1.0) {
        throw std::runtime_error("--ils-repair-mtv-split precisa estar em [0,1].");
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
                p.hh_segment = opt.sa_hh_segment;
                p.hh_reaction = opt.sa_hh_reaction;
                p.overlap_metric = opt.sa_overlap_metric;
                p.overlap_weight = opt.sa_overlap_weight;
                p.overlap_weight_start = opt.sa_overlap_weight_start;
                p.overlap_weight_end = opt.sa_overlap_weight_end;
                p.overlap_weight_power = opt.sa_overlap_weight_power;
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
