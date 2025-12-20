#include "solver_tile_cli.hpp"

#include <sstream>
#include <stdexcept>
#include <string>

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
        } else if (arg == "--sa-lns-candidates") {
            opt.sa_lns_candidates = parse_int(need(arg));
        } else if (arg == "--sa-lns-eval-attempts") {
            opt.sa_lns_eval_attempts_per_tree = parse_int(need(arg));
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
        } else if (arg == "--sa-overlap-weight-geometric") {
            opt.sa_overlap_weight_geometric = true;
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
        } else if (arg == "--mz-its-iters") {
            opt.mz_its_iters = parse_int(need(arg));
        } else if (arg == "--mz-perturb-depth") {
            opt.mz_perturb_depth = parse_int(need(arg));
        } else if (arg == "--mz-tabu-depth") {
            opt.mz_tabu_depth = parse_int(need(arg));
        } else if (arg == "--mz-tabu-samples") {
            opt.mz_tabu_samples = parse_int(need(arg));
        } else if (arg == "--mz-phase-a-iters") {
            opt.mz_phase_a_iters = parse_int(need(arg));
        } else if (arg == "--mz-phase-b-iters") {
            opt.mz_phase_b_iters = parse_int(need(arg));
        } else if (arg == "--mz-a-t0") {
            opt.mz_a_t0 = parse_double(need(arg));
        } else if (arg == "--mz-a-t1") {
            opt.mz_a_t1 = parse_double(need(arg));
        } else if (arg == "--mz-b-t0") {
            opt.mz_b_t0 = parse_double(need(arg));
        } else if (arg == "--mz-b-t1") {
            opt.mz_b_t1 = parse_double(need(arg));
        } else if (arg == "--mz-overlap-a") {
            opt.mz_overlap_a = parse_double(need(arg));
        } else if (arg == "--mz-overlap-b-start") {
            opt.mz_overlap_b_start = parse_double(need(arg));
        } else if (arg == "--mz-overlap-b-end") {
            opt.mz_overlap_b_end = parse_double(need(arg));
        } else if (arg == "--mz-overlap-b-geometric") {
            opt.mz_overlap_b_geometric = true;
        } else if (arg == "--mz-w-push-contact" || arg == "--mz-w-squeeze") {
            opt.mz_w_push_contact = parse_double(need(arg));
        } else if (arg == "--mz-push-overshoot-a") {
            opt.mz_push_overshoot_a = parse_double(need(arg));
        } else if (arg == "--mz-push-overshoot-b") {
            opt.mz_push_overshoot_b = parse_double(need(arg));
        } else if (arg == "--mz-w-resolve-overlap-b") {
            opt.mz_w_resolve_overlap_b = parse_double(need(arg));
        } else if (arg == "--no-final-rigid" || arg == "--no-sa-rigid") {
            opt.final_rigid = false;
        } else if (arg == "--micro-rot-eps") {
            opt.micro_rot_eps = parse_double(need(arg));
        } else if (arg == "--micro-rot-steps") {
            opt.micro_rot_steps = parse_int(need(arg));
        } else if (arg == "--micro-shift-eps") {
            opt.micro_shift_eps = parse_double(need(arg));
        } else if (arg == "--micro-shift-steps") {
            opt.micro_shift_steps = parse_int(need(arg));
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
    if (opt.sa_lns_candidates < 1) {
        throw std::runtime_error("--sa-lns-candidates precisa ser >= 1.");
    }
    if (opt.sa_lns_eval_attempts_per_tree < 0) {
        throw std::runtime_error("--sa-lns-eval-attempts precisa ser >= 0.");
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
    if (opt.mz_its_iters < 0) {
        throw std::runtime_error("--mz-its-iters precisa ser >= 0.");
    }
    if (opt.mz_tabu_depth < -1) {
        throw std::runtime_error("--mz-tabu-depth precisa ser >= -1.");
    }
    if (opt.mz_perturb_depth < -1) {
        throw std::runtime_error("--mz-perturb-depth precisa ser >= -1.");
    }
    if (opt.mz_tabu_samples < 0) {
        throw std::runtime_error("--mz-tabu-samples precisa ser >= 0.");
    }
    if (opt.mz_phase_a_iters < 0 || opt.mz_phase_b_iters < 0) {
        throw std::runtime_error("--mz-phase-a-iters/--mz-phase-b-iters precisam ser >= 0.");
    }
    if (opt.mz_its_iters > 0) {
        if (!(opt.mz_a_t0 > 0.0) || !(opt.mz_a_t1 > 0.0) || opt.mz_a_t1 > opt.mz_a_t0) {
            throw std::runtime_error("--mz-a-t0/--mz-a-t1 inválidos (0 < t1 <= t0).");
        }
        if (!(opt.mz_b_t0 > 0.0) || !(opt.mz_b_t1 > 0.0) || opt.mz_b_t1 > opt.mz_b_t0) {
            throw std::runtime_error("--mz-b-t0/--mz-b-t1 inválidos (0 < t1 <= t0).");
        }
        if (!(opt.mz_overlap_a >= 0.0)) {
            throw std::runtime_error("--mz-overlap-a precisa ser >= 0.");
        }
        if (!(opt.mz_overlap_b_start >= 0.0) || !(opt.mz_overlap_b_end >= 0.0)) {
            throw std::runtime_error("--mz-overlap-b-start/end precisam ser >= 0.");
        }
        if (opt.mz_w_push_contact < 0.0) {
            throw std::runtime_error("--mz-w-push-contact precisa ser >= 0.");
        }
        if (opt.mz_push_overshoot_a < 0.0 || opt.mz_push_overshoot_a > 1.0 ||
            opt.mz_push_overshoot_b < 0.0 || opt.mz_push_overshoot_b > 1.0) {
            throw std::runtime_error("--mz-push-overshoot-a/b precisam estar em [0,1].");
        }
        if (opt.mz_w_resolve_overlap_b < 0.0) {
            throw std::runtime_error("--mz-w-resolve-overlap-b precisa ser >= 0.");
        }
    }
    if (opt.micro_rot_steps < 0 || opt.micro_shift_steps < 0) {
        throw std::runtime_error("--micro-rot-steps/--micro-shift-steps precisam ser >= 0.");
    }
    if (opt.micro_rot_eps < 0.0 || opt.micro_shift_eps < 0.0) {
        throw std::runtime_error("--micro-rot-eps/--micro-shift-eps precisam ser >= 0.");
    }
    if (opt.micro_rot_steps > 0 && !(opt.micro_rot_eps > 0.0)) {
        throw std::runtime_error("--micro-rot-eps precisa ser > 0 quando steps > 0.");
    }
    if (opt.micro_shift_steps > 0 && !(opt.micro_shift_eps > 0.0)) {
        throw std::runtime_error("--micro-shift-eps precisa ser > 0 quando steps > 0.");
    }
    return opt;
}
