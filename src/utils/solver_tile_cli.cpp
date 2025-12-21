#include "solver_tile_cli.hpp"

#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "cli_parse.hpp"

namespace {

SARefiner::OverlapMetric parse_overlap_metric(const std::string& s) {
    if (s == "area") {
        return SARefiner::OverlapMetric::kArea;
    }
    if (s == "mtv2" || s == "mtv") {
        return SARefiner::OverlapMetric::kMtv2;
    }
    throw std::runtime_error("--sa-overlap-metric must be 'area' or 'mtv2'.");
}

void parse_hh_mode(Options& opt, const std::string& s) {
    if (s == "auto") {
        opt.sa_hh_auto = true;
        return;
    }
    if (s == "off" || s == "default") {
        opt.sa_hh_auto = false;
        return;
    }
    throw std::runtime_error("--sa-hh-mode must be 'off' or 'auto'.");
}

TileObjective parse_tile_objective(const std::string& s) {
    if (s == "density") {
        return TileObjective::kDensity;
    }
    if (s == "score") {
        return TileObjective::kScore;
    }
    throw std::runtime_error("--tile-obj must be 'density' or 'score'.");
}

ShiftSearchMode parse_shift_search(const std::string& s) {
    if (s == "off") {
        return ShiftSearchMode::kOff;
    }
    if (s == "multires") {
        return ShiftSearchMode::kMultires;
    }
    throw std::runtime_error("--shift-search must be 'off' or 'multires'.");
}

void parse_shift(Options& opt, const std::string& s) {
    std::stringstream ss(s);
    std::string a;
    std::string b;
    if (!std::getline(ss, a, ',') || !std::getline(ss, b, ',')) {
        throw std::runtime_error("--shift expects 'a,b'.");
    }
    opt.shift_a = parse_double(a);
    opt.shift_b = parse_double(b);
}

void parse_angles(Options& opt, const std::string& s) {
    opt.angle_candidates = parse_double_list(s);
    if (opt.angle_candidates.empty()) {
        throw std::runtime_error("--angles cannot be empty.");
    }
}

void ensure(bool cond, const char* message) {
    if (!cond) {
        throw std::runtime_error(message);
    }
}

void validate_tile_options(const Options& opt) {
    ensure(opt.k > 0, "--k must be > 0.");
    ensure(opt.pool_size > 0, "--pool-size must be > 0.");
    ensure(opt.pool_size >= opt.n_max, "--pool-size must be >= 200.");
    ensure(opt.pool_window_radius >= 0, "--pool-window-radius must be >= 0.");
    ensure(opt.prefix_order == "central" || opt.prefix_order == "greedy",
           "--prefix-order must be 'central' or 'greedy'.");
    ensure(opt.tile_score_pool_size >= 0, "--tile-score-pool-size must be >= 0.");
    ensure(opt.tile_score_nmax > 0 && opt.tile_score_nmax <= opt.n_max,
           "--tile-score-nmax must be in [1, 200].");
    ensure(opt.tile_score_full_every > 0, "--tile-score-full-every must be > 0.");
}

void validate_lattice_options(const Options& opt) {
    ensure(opt.lattice_v_ratio > 0.0, "--lattice-v-ratio must be > 0.");
    ensure(opt.lattice_v_ratio >= 0.50 && opt.lattice_v_ratio <= 2.00,
           "--lattice-v-ratio must be in [0.50, 2.00].");
    ensure(opt.lattice_theta_deg >= 20.0 && opt.lattice_theta_deg <= 160.0,
           "--lattice-theta must be in [20, 160] (avoids degenerate lattices).");
}

void validate_shift_options(const Options& opt) {
    ensure(opt.spacing_safety >= 1.0, "--spacing-safety must be >= 1.0.");
    ensure(opt.shift_grid > 0, "--shift-grid must be > 0.");
    ensure(opt.shift_levels > 0, "--shift-levels must be > 0.");
    ensure(opt.shift_keep > 0, "--shift-keep must be > 0.");
    ensure(opt.shift_pool_size >= 0, "--shift-pool-size must be >= 0.");
}

void validate_sa_options(const Options& opt) {
    ensure(opt.sa_chain_base_iters >= 0 && opt.sa_chain_iters_per_n >= 0,
           "--sa-chain-base-iters/--sa-chain-iters-per-n must be >= 0.");
    ensure(opt.sa_w_micro >= 0.0 && opt.sa_w_swap_rot >= 0.0 && opt.sa_w_relocate >= 0.0 &&
               opt.sa_w_block_translate >= 0.0 && opt.sa_w_block_rotate >= 0.0 &&
               opt.sa_w_lns >= 0.0 && opt.sa_w_resolve_overlap >= 0.0 &&
               opt.sa_w_push_contact >= 0.0 && opt.sa_w_slide_contact >= 0.0 &&
               opt.sa_w_squeeze >= 0.0,
           "SA weights must be >= 0.");
    ensure(opt.sa_block_size > 0, "--sa-block-size must be > 0.");
    ensure(opt.sa_lns_remove >= 0, "--sa-lns-remove must be >= 0.");
    ensure(opt.sa_lns_candidates >= 1, "--sa-lns-candidates must be >= 1.");
    ensure(opt.sa_lns_eval_attempts_per_tree >= 0, "--sa-lns-eval-attempts must be >= 0.");
    ensure(opt.sa_hh_segment >= 0, "--sa-hh-segment must be >= 0.");
    ensure(opt.sa_hh_reaction >= 0.0 && opt.sa_hh_reaction <= 1.0,
           "--sa-hh-reaction must be in [0, 1].");
    ensure(opt.sa_overlap_weight >= 0.0, "--sa-overlap-weight must be >= 0.");
    ensure(opt.sa_overlap_weight_power > 0.0, "--sa-overlap-weight-power must be > 0.");
    ensure(opt.sa_overlap_eps_area >= 0.0, "--sa-overlap-eps-area must be >= 0.");
    ensure(opt.sa_overlap_cost_cap >= 0.0, "--sa-overlap-cost-cap must be >= 0.");
    ensure(opt.sa_plateau_eps >= 0.0, "--sa-plateau-eps must be >= 0.");
    ensure(opt.sa_resolve_attempts > 0, "--sa-resolve-attempts must be > 0.");
    ensure(opt.sa_resolve_step_frac_max > 0.0 && opt.sa_resolve_step_frac_min > 0.0 &&
               opt.sa_resolve_step_frac_min <= opt.sa_resolve_step_frac_max,
           "Invalid --sa-resolve-step-frac-min/max.");
    ensure(opt.sa_resolve_noise_frac >= 0.0, "--sa-resolve-noise-frac must be >= 0.");
    ensure(opt.sa_push_max_step_frac > 0.0, "--sa-push-max-step-frac must be > 0.");
    ensure(opt.sa_push_bisect_iters > 0, "--sa-push-bisect-iters must be > 0.");
    ensure(opt.sa_push_overshoot_frac >= 0.0 && opt.sa_push_overshoot_frac <= 1.0,
           "--sa-push-overshoot-frac must be in [0, 1].");
    ensure(opt.sa_slide_dirs == 4 || opt.sa_slide_dirs == 8,
           "--sa-slide-dirs must be 4 or 8.");
    ensure(opt.sa_slide_dir_bias >= 0.0, "--sa-slide-dir-bias must be >= 0.");
    ensure(opt.sa_slide_max_step_frac > 0.0, "--sa-slide-max-step-frac must be > 0.");
    ensure(opt.sa_slide_bisect_iters > 0, "--sa-slide-bisect-iters must be > 0.");
    ensure(opt.sa_slide_min_gain >= 0.0, "--sa-slide-min-gain must be >= 0.");
    ensure(opt.sa_slide_schedule_max_frac >= 0.0 && opt.sa_slide_schedule_max_frac < 1.0,
           "--sa-slide-schedule-max-frac must be in [0, 1).");
    ensure(opt.sa_squeeze_pushes >= 0, "--sa-squeeze-pushes must be >= 0.");

    if (opt.sa_chain) {
        ensure(opt.sa_chain_min_n >= 1 && opt.sa_chain_min_n <= opt.n_max,
               "--sa-chain-min-n must be in [1, 200].");
        ensure(opt.sa_chain_band_layers > 0.0, "--sa-chain-band-layers must be > 0.");
        ensure(opt.sa_chain_band_layers <= 20.0,
               "--sa-chain-band-layers too high (use <= 20).");
    }

    if (opt.sa_beam) {
        ensure(opt.sa_beam_width > 0, "--sa-beam-width must be > 0.");
        ensure(opt.sa_beam_remove > 0, "--sa-beam-remove must be > 0.");
        ensure(opt.sa_beam_micro_iters >= 0, "--sa-beam-micro-iters must be >= 0.");
        ensure(opt.sa_beam_init_iters >= 0, "--sa-beam-init-iters must be >= 0.");
    }
}

void validate_ils_options(const Options& opt) {
    ensure(opt.ils_iters >= 0, "--ils-iters must be >= 0.");
    ensure(opt.ils_sa_restarts >= 0, "--ils-sa-restarts must be >= 0.");
    ensure(opt.ils_sa_iters >= 0, "--ils-sa-iters must be >= 0.");
    ensure(opt.ils_alpha_min > 0.0 && opt.ils_alpha_max > 0.0 &&
               opt.ils_alpha_min <= opt.ils_alpha_max && opt.ils_alpha_max <= 1.0,
           "Invalid --ils-alpha-min/max (0 < min <= max <= 1).");
    ensure(opt.ils_p_aniso >= 0.0 && opt.ils_p_aniso <= 1.0,
           "--ils-p-aniso must be in [0,1].");
    ensure(opt.ils_shear_max >= 0.0 && opt.ils_shear_max <= 0.50,
           "--ils-shear-max must be in [0,0.50].");
    ensure(opt.ils_jitter_frac >= 0.0 && opt.ils_jitter_frac <= 0.50,
           "--ils-jitter-frac must be in [0,0.50].");
    ensure(opt.ils_subset_frac >= 0.0 && opt.ils_subset_frac <= 1.0,
           "--ils-subset-frac must be in [0,1].");
    ensure(opt.ils_rot_prob >= 0.0 && opt.ils_rot_prob <= 1.0,
           "--ils-rot-prob must be in [0,1].");
    ensure(opt.ils_rot_deg_max >= 0.0, "--ils-rot-deg-max must be >= 0.");
    ensure(opt.ils_repair_mtv_passes >= 0, "--ils-repair-mtv-passes must be >= 0.");
    ensure(opt.ils_repair_mtv_damping > 0.0, "--ils-repair-mtv-damping must be > 0.");
    ensure(opt.ils_repair_mtv_split >= 0.0 && opt.ils_repair_mtv_split <= 1.0,
           "--ils-repair-mtv-split must be in [0,1].");

    if (opt.ils_accept_sa) {
        ensure(opt.ils_t0 > 0.0 && opt.ils_t1 > 0.0 && opt.ils_t1 <= opt.ils_t0,
               "--ils-t0/--ils-t1 invalid (0 < t1 <= t0).");
    }
}

void validate_mz_options(const Options& opt) {
    ensure(opt.mz_its_iters >= 0, "--mz-its-iters must be >= 0.");
    ensure(opt.mz_tabu_depth >= -1, "--mz-tabu-depth must be >= -1.");
    ensure(opt.mz_perturb_depth >= -1, "--mz-perturb-depth must be >= -1.");
    ensure(opt.mz_tabu_samples >= 0, "--mz-tabu-samples must be >= 0.");
    ensure(opt.mz_phase_a_iters >= 0 && opt.mz_phase_b_iters >= 0,
           "--mz-phase-a-iters/--mz-phase-b-iters must be >= 0.");

    if (opt.mz_its_iters > 0) {
        ensure(opt.mz_a_t0 > 0.0 && opt.mz_a_t1 > 0.0 && opt.mz_a_t1 <= opt.mz_a_t0,
               "Invalid --mz-a-t0/--mz-a-t1 (0 < t1 <= t0).");
        ensure(opt.mz_b_t0 > 0.0 && opt.mz_b_t1 > 0.0 && opt.mz_b_t1 <= opt.mz_b_t0,
               "Invalid --mz-b-t0/--mz-b-t1 (0 < t1 <= t0).");
        ensure(opt.mz_overlap_a >= 0.0, "--mz-overlap-a must be >= 0.");
        ensure(opt.mz_overlap_b_start >= 0.0 && opt.mz_overlap_b_end >= 0.0,
               "--mz-overlap-b-start/end must be >= 0.");
        ensure(opt.mz_w_push_contact >= 0.0, "--mz-w-push-contact must be >= 0.");
        ensure(opt.mz_push_overshoot_a >= 0.0 && opt.mz_push_overshoot_a <= 1.0 &&
                   opt.mz_push_overshoot_b >= 0.0 && opt.mz_push_overshoot_b <= 1.0,
               "--mz-push-overshoot-a/b must be in [0,1].");
        ensure(opt.mz_w_resolve_overlap_b >= 0.0, "--mz-w-resolve-overlap-b must be >= 0.");
    }
}

void validate_micro_options(const Options& opt) {
    ensure(opt.micro_rot_steps >= 0 && opt.micro_shift_steps >= 0,
           "--micro-rot-steps/--micro-shift-steps must be >= 0.");
    ensure(opt.micro_rot_eps >= 0.0 && opt.micro_shift_eps >= 0.0,
           "--micro-rot-eps/--micro-shift-eps must be >= 0.");
    ensure(opt.micro_rot_steps == 0 || opt.micro_rot_eps > 0.0,
           "--micro-rot-eps must be > 0 when steps > 0.");
    ensure(opt.micro_shift_steps == 0 || opt.micro_shift_eps > 0.0,
           "--micro-shift-eps must be > 0 when steps > 0.");
}

void validate_target_options(const Options& opt) {
    ensure(opt.target_cover >= 0.0 && opt.target_cover <= 1.0,
           "--target-cover must be in [0,1].");
    ensure(opt.target_m_min > 0, "--target-m-min must be > 0.");
    ensure(opt.target_m_max >= opt.target_m_min,
           "--target-m-max must be >= --target-m-min.");
    ensure(opt.target_m >= 0, "--target-M must be >= 0.");
    ensure(opt.target_tier_a >= 0 && opt.target_tier_b >= 0,
           "--target-tierA/--target-tierB must be >= 0.");
    ensure(opt.target_budget_scale > 0.0, "--target-budget-scale must be > 0.");
    ensure(opt.target_soft_overlap_cut > 0.0 && opt.target_soft_overlap_cut <= 1.0,
           "--target-soft-overlap-cut must be in (0,1].");
    ensure(opt.target_sa_check_interval > 0,
           "--target-sa-check-interval must be > 0.");
}

struct ArgHandlers {
    std::unordered_map<std::string, std::function<void(const std::string&)>> with_value;
    std::unordered_map<std::string, std::function<void()>> flags;
};

ArgHandlers make_arg_handlers(Options& opt) {
    ArgHandlers handlers;

    handlers.with_value.emplace("--k", [&](const std::string& v) { opt.k = parse_int(v); });
    handlers.with_value.emplace("--pool-size", [&](const std::string& v) {
        opt.pool_size = parse_int(v);
    });
    handlers.flags.emplace("--pool-window", [&]() { opt.pool_window_scan = true; });
    handlers.with_value.emplace("--pool-window-radius", [&](const std::string& v) {
        opt.pool_window_radius = parse_int(v);
    });
    handlers.with_value.emplace("--prefix-order", [&](const std::string& v) {
        opt.prefix_order = v;
    });
    handlers.with_value.emplace("--tile-iters", [&](const std::string& v) {
        opt.tile_iters = parse_int(v);
    });
    handlers.with_value.emplace("--tile-obj", [&](const std::string& v) {
        opt.tile_obj = parse_tile_objective(v);
    });
    handlers.with_value.emplace("--tile-score-pool-size", [&](const std::string& v) {
        opt.tile_score_pool_size = parse_int(v);
    });
    handlers.with_value.emplace("--tile-score-nmax", [&](const std::string& v) {
        opt.tile_score_nmax = parse_int(v);
    });
    handlers.with_value.emplace("--tile-score-full-every", [&](const std::string& v) {
        opt.tile_score_full_every = parse_int(v);
    });
    handlers.flags.emplace("--no-tile-opt-lattice", [&]() { opt.tile_opt_lattice = false; });
    handlers.with_value.emplace("--lattice-v-ratio", [&](const std::string& v) {
        opt.lattice_v_ratio = parse_double(v);
    });
    handlers.with_value.emplace("--lattice-theta", [&](const std::string& v) {
        opt.lattice_theta_deg = parse_double(v);
    });
    handlers.with_value.emplace("--refine-iters", [&](const std::string& v) {
        opt.refine_iters = parse_int(v);
    });
    handlers.with_value.emplace("--sa-restarts", [&](const std::string& v) {
        opt.sa_restarts = parse_int(v);
    });
    handlers.with_value.emplace("--sa-base-iters", [&](const std::string& v) {
        opt.sa_base_iters = parse_int(v);
    });
    handlers.with_value.emplace("--sa-iters-per-n", [&](const std::string& v) {
        opt.sa_iters_per_n = parse_int(v);
    });
    handlers.flags.emplace("--sa-chain", [&]() { opt.sa_chain = true; });
    handlers.flags.emplace("--sa-beam", [&]() {
        opt.sa_beam = true;
        if (opt.sa_beam_width <= 0) {
            opt.sa_beam_width = 16;
        }
    });
    handlers.with_value.emplace("--sa-beam-width", [&](const std::string& v) {
        opt.sa_beam_width = parse_int(v);
        opt.sa_beam = (opt.sa_beam_width > 0);
    });
    handlers.with_value.emplace("--sa-beam-remove", [&](const std::string& v) {
        opt.sa_beam_remove = parse_int(v);
    });
    handlers.with_value.emplace("--sa-beam-micro-iters", [&](const std::string& v) {
        opt.sa_beam_micro_iters = parse_int(v);
    });
    handlers.with_value.emplace("--sa-beam-init-iters", [&](const std::string& v) {
        opt.sa_beam_init_iters = parse_int(v);
    });
    handlers.with_value.emplace("--sa-chain-base-iters", [&](const std::string& v) {
        opt.sa_chain_base_iters = parse_int(v);
    });
    handlers.with_value.emplace("--sa-chain-iters-per-n", [&](const std::string& v) {
        opt.sa_chain_iters_per_n = parse_int(v);
    });
    handlers.with_value.emplace("--sa-chain-min-n", [&](const std::string& v) {
        opt.sa_chain_min_n = parse_int(v);
    });
    handlers.with_value.emplace("--sa-chain-band-layers", [&](const std::string& v) {
        opt.sa_chain_band_layers = parse_double(v);
    });
    handlers.with_value.emplace("--sa-w-micro", [&](const std::string& v) {
        opt.sa_w_micro = parse_double(v);
    });
    handlers.with_value.emplace("--sa-w-swap-rot", [&](const std::string& v) {
        opt.sa_w_swap_rot = parse_double(v);
    });
    handlers.with_value.emplace("--sa-w-relocate", [&](const std::string& v) {
        opt.sa_w_relocate = parse_double(v);
    });
    handlers.with_value.emplace("--sa-w-block-translate", [&](const std::string& v) {
        opt.sa_w_block_translate = parse_double(v);
    });
    handlers.with_value.emplace("--sa-w-block-rotate", [&](const std::string& v) {
        opt.sa_w_block_rotate = parse_double(v);
    });
    handlers.with_value.emplace("--sa-w-lns", [&](const std::string& v) {
        opt.sa_w_lns = parse_double(v);
    });
    handlers.with_value.emplace("--sa-w-push-contact", [&](const std::string& v) {
        opt.sa_w_push_contact = parse_double(v);
    });
    handlers.with_value.emplace("--sa-w-slide-contact", [&](const std::string& v) {
        opt.sa_w_slide_contact = parse_double(v);
    });
    handlers.with_value.emplace("--sa-w-squeeze", [&](const std::string& v) {
        opt.sa_w_squeeze = parse_double(v);
    });
    handlers.with_value.emplace("--sa-block-size", [&](const std::string& v) {
        opt.sa_block_size = parse_int(v);
    });
    handlers.with_value.emplace("--sa-lns-remove", [&](const std::string& v) {
        opt.sa_lns_remove = parse_int(v);
    });
    handlers.with_value.emplace("--sa-lns-candidates", [&](const std::string& v) {
        opt.sa_lns_candidates = parse_int(v);
    });
    handlers.with_value.emplace("--sa-lns-eval-attempts", [&](const std::string& v) {
        opt.sa_lns_eval_attempts_per_tree = parse_int(v);
    });
    handlers.with_value.emplace("--sa-hh-segment", [&](const std::string& v) {
        opt.sa_hh_segment = parse_int(v);
    });
    handlers.with_value.emplace("--sa-hh-reaction", [&](const std::string& v) {
        opt.sa_hh_reaction = parse_double(v);
    });
    handlers.with_value.emplace("--sa-hh-mode", [&](const std::string& v) {
        parse_hh_mode(opt, v);
    });
    handlers.with_value.emplace("--sa-overlap-metric", [&](const std::string& v) {
        opt.sa_overlap_metric = parse_overlap_metric(v);
    });
    handlers.with_value.emplace("--sa-overlap-weight", [&](const std::string& v) {
        opt.sa_overlap_weight = parse_double(v);
    });
    handlers.with_value.emplace("--sa-overlap-weight-start", [&](const std::string& v) {
        opt.sa_overlap_weight_start = parse_double(v);
    });
    handlers.with_value.emplace("--sa-overlap-weight-end", [&](const std::string& v) {
        opt.sa_overlap_weight_end = parse_double(v);
    });
    handlers.with_value.emplace("--sa-overlap-weight-power", [&](const std::string& v) {
        opt.sa_overlap_weight_power = parse_double(v);
    });
    handlers.flags.emplace("--sa-overlap-weight-geometric", [&]() {
        opt.sa_overlap_weight_geometric = true;
    });
    handlers.with_value.emplace("--sa-overlap-eps-area", [&](const std::string& v) {
        opt.sa_overlap_eps_area = parse_double(v);
    });
    handlers.with_value.emplace("--sa-overlap-cost-cap", [&](const std::string& v) {
        opt.sa_overlap_cost_cap = parse_double(v);
    });
    handlers.with_value.emplace("--sa-plateau-eps", [&](const std::string& v) {
        opt.sa_plateau_eps = parse_double(v);
    });
    handlers.with_value.emplace("--sa-w-resolve-overlap", [&](const std::string& v) {
        opt.sa_w_resolve_overlap = parse_double(v);
    });
    handlers.with_value.emplace("--sa-resolve-attempts", [&](const std::string& v) {
        opt.sa_resolve_attempts = parse_int(v);
    });
    handlers.with_value.emplace("--sa-resolve-step-frac-max", [&](const std::string& v) {
        opt.sa_resolve_step_frac_max = parse_double(v);
    });
    handlers.with_value.emplace("--sa-resolve-step-frac-min", [&](const std::string& v) {
        opt.sa_resolve_step_frac_min = parse_double(v);
    });
    handlers.with_value.emplace("--sa-resolve-noise-frac", [&](const std::string& v) {
        opt.sa_resolve_noise_frac = parse_double(v);
    });
    handlers.with_value.emplace("--sa-push-max-step-frac", [&](const std::string& v) {
        opt.sa_push_max_step_frac = parse_double(v);
    });
    handlers.with_value.emplace("--sa-push-bisect-iters", [&](const std::string& v) {
        opt.sa_push_bisect_iters = parse_int(v);
    });
    handlers.with_value.emplace("--sa-push-overshoot-frac", [&](const std::string& v) {
        opt.sa_push_overshoot_frac = parse_double(v);
    });
    handlers.with_value.emplace("--sa-slide-dirs", [&](const std::string& v) {
        opt.sa_slide_dirs = parse_int(v);
    });
    handlers.with_value.emplace("--sa-slide-dir-bias", [&](const std::string& v) {
        opt.sa_slide_dir_bias = parse_double(v);
    });
    handlers.with_value.emplace("--sa-slide-max-step-frac", [&](const std::string& v) {
        opt.sa_slide_max_step_frac = parse_double(v);
    });
    handlers.with_value.emplace("--sa-slide-bisect-iters", [&](const std::string& v) {
        opt.sa_slide_bisect_iters = parse_int(v);
    });
    handlers.with_value.emplace("--sa-slide-min-gain", [&](const std::string& v) {
        opt.sa_slide_min_gain = parse_double(v);
    });
    handlers.with_value.emplace("--sa-slide-schedule-max-frac", [&](const std::string& v) {
        opt.sa_slide_schedule_max_frac = parse_double(v);
    });
    handlers.with_value.emplace("--sa-squeeze-pushes", [&](const std::string& v) {
        opt.sa_squeeze_pushes = parse_int(v);
    });
    handlers.flags.emplace("--sa-aggressive", [&]() { opt.sa_aggressive = true; });
    handlers.with_value.emplace("--ils-iters", [&](const std::string& v) {
        opt.ils_iters = parse_int(v);
    });
    handlers.with_value.emplace("--ils-sa-restarts", [&](const std::string& v) {
        opt.ils_sa_restarts = parse_int(v);
    });
    handlers.with_value.emplace("--ils-sa-iters", [&](const std::string& v) {
        opt.ils_sa_iters = parse_int(v);
    });
    handlers.flags.emplace("--ils-accept-sa", [&]() { opt.ils_accept_sa = true; });
    handlers.with_value.emplace("--ils-t0", [&](const std::string& v) {
        opt.ils_t0 = parse_double(v);
    });
    handlers.with_value.emplace("--ils-t1", [&](const std::string& v) {
        opt.ils_t1 = parse_double(v);
    });
    handlers.with_value.emplace("--ils-alpha-min", [&](const std::string& v) {
        opt.ils_alpha_min = parse_double(v);
    });
    handlers.with_value.emplace("--ils-alpha-max", [&](const std::string& v) {
        opt.ils_alpha_max = parse_double(v);
    });
    handlers.with_value.emplace("--ils-p-aniso", [&](const std::string& v) {
        opt.ils_p_aniso = parse_double(v);
    });
    handlers.with_value.emplace("--ils-shear-max", [&](const std::string& v) {
        opt.ils_shear_max = parse_double(v);
    });
    handlers.with_value.emplace("--ils-jitter-frac", [&](const std::string& v) {
        opt.ils_jitter_frac = parse_double(v);
    });
    handlers.with_value.emplace("--ils-subset-frac", [&](const std::string& v) {
        opt.ils_subset_frac = parse_double(v);
    });
    handlers.with_value.emplace("--ils-rot-prob", [&](const std::string& v) {
        opt.ils_rot_prob = parse_double(v);
    });
    handlers.with_value.emplace("--ils-rot-deg-max", [&](const std::string& v) {
        opt.ils_rot_deg_max = parse_double(v);
    });
    handlers.with_value.emplace("--ils-repair-mtv-passes", [&](const std::string& v) {
        opt.ils_repair_mtv_passes = parse_int(v);
    });
    handlers.with_value.emplace("--ils-repair-mtv-damping", [&](const std::string& v) {
        opt.ils_repair_mtv_damping = parse_double(v);
    });
    handlers.with_value.emplace("--ils-repair-mtv-split", [&](const std::string& v) {
        opt.ils_repair_mtv_split = parse_double(v);
    });
    handlers.with_value.emplace("--mz-its-iters", [&](const std::string& v) {
        opt.mz_its_iters = parse_int(v);
    });
    handlers.with_value.emplace("--mz-perturb-depth", [&](const std::string& v) {
        opt.mz_perturb_depth = parse_int(v);
    });
    handlers.with_value.emplace("--mz-tabu-depth", [&](const std::string& v) {
        opt.mz_tabu_depth = parse_int(v);
    });
    handlers.with_value.emplace("--mz-tabu-samples", [&](const std::string& v) {
        opt.mz_tabu_samples = parse_int(v);
    });
    handlers.with_value.emplace("--mz-phase-a-iters", [&](const std::string& v) {
        opt.mz_phase_a_iters = parse_int(v);
    });
    handlers.with_value.emplace("--mz-phase-b-iters", [&](const std::string& v) {
        opt.mz_phase_b_iters = parse_int(v);
    });
    handlers.with_value.emplace("--mz-a-t0", [&](const std::string& v) {
        opt.mz_a_t0 = parse_double(v);
    });
    handlers.with_value.emplace("--mz-a-t1", [&](const std::string& v) {
        opt.mz_a_t1 = parse_double(v);
    });
    handlers.with_value.emplace("--mz-b-t0", [&](const std::string& v) {
        opt.mz_b_t0 = parse_double(v);
    });
    handlers.with_value.emplace("--mz-b-t1", [&](const std::string& v) {
        opt.mz_b_t1 = parse_double(v);
    });
    handlers.with_value.emplace("--mz-overlap-a", [&](const std::string& v) {
        opt.mz_overlap_a = parse_double(v);
    });
    handlers.with_value.emplace("--mz-overlap-b-start", [&](const std::string& v) {
        opt.mz_overlap_b_start = parse_double(v);
    });
    handlers.with_value.emplace("--mz-overlap-b-end", [&](const std::string& v) {
        opt.mz_overlap_b_end = parse_double(v);
    });
    handlers.flags.emplace("--mz-overlap-b-geometric", [&]() { opt.mz_overlap_b_geometric = true; });
    handlers.with_value.emplace("--mz-w-push-contact", [&](const std::string& v) {
        opt.mz_w_push_contact = parse_double(v);
    });
    handlers.with_value.emplace("--mz-w-squeeze", [&](const std::string& v) {
        opt.mz_w_push_contact = parse_double(v);
    });
    handlers.with_value.emplace("--mz-push-overshoot-a", [&](const std::string& v) {
        opt.mz_push_overshoot_a = parse_double(v);
    });
    handlers.with_value.emplace("--mz-push-overshoot-b", [&](const std::string& v) {
        opt.mz_push_overshoot_b = parse_double(v);
    });
    handlers.with_value.emplace("--mz-w-resolve-overlap-b", [&](const std::string& v) {
        opt.mz_w_resolve_overlap_b = parse_double(v);
    });
    handlers.flags.emplace("--no-final-rigid", [&]() { opt.final_rigid = false; });
    handlers.flags.emplace("--no-sa-rigid", [&]() { opt.final_rigid = false; });
    handlers.with_value.emplace("--micro-rot-eps", [&](const std::string& v) {
        opt.micro_rot_eps = parse_double(v);
    });
    handlers.with_value.emplace("--micro-rot-steps", [&](const std::string& v) {
        opt.micro_rot_steps = parse_int(v);
    });
    handlers.with_value.emplace("--micro-shift-eps", [&](const std::string& v) {
        opt.micro_shift_eps = parse_double(v);
    });
    handlers.with_value.emplace("--micro-shift-steps", [&](const std::string& v) {
        opt.micro_shift_steps = parse_int(v);
    });
    handlers.flags.emplace("--target-refine", [&]() { opt.target_refine = true; });
    handlers.with_value.emplace("--target-cover", [&](const std::string& v) {
        opt.target_cover = parse_double(v);
    });
    handlers.with_value.emplace("--target-m-min", [&](const std::string& v) {
        opt.target_m_min = parse_int(v);
    });
    handlers.with_value.emplace("--target-m-max", [&](const std::string& v) {
        opt.target_m_max = parse_int(v);
    });
    handlers.with_value.emplace("--target-M", [&](const std::string& v) {
        opt.target_m = parse_int(v);
    });
    handlers.with_value.emplace("--target-m", [&](const std::string& v) {
        opt.target_m = parse_int(v);
    });
    handlers.with_value.emplace("--target-tierA", [&](const std::string& v) {
        opt.target_tier_a = parse_int(v);
    });
    handlers.with_value.emplace("--target-tierB", [&](const std::string& v) {
        opt.target_tier_b = parse_int(v);
    });
    handlers.with_value.emplace("--target-budget-scale", [&](const std::string& v) {
        opt.target_budget_scale = parse_double(v);
    });
    handlers.with_value.emplace("--target-soft-overlap", [&](const std::string& v) {
        opt.target_soft_overlap = (parse_int(v) != 0);
    });
    handlers.with_value.emplace("--target-soft-overlap-tierA-only", [&](const std::string& v) {
        opt.target_soft_overlap_tier_a_only = (parse_int(v) != 0);
    });
    handlers.with_value.emplace("--target-soft-overlap-cut", [&](const std::string& v) {
        opt.target_soft_overlap_cut = parse_double(v);
    });
    handlers.with_value.emplace("--target-early-stop", [&](const std::string& v) {
        opt.target_early_stop = (parse_int(v) != 0);
    });
    handlers.with_value.emplace("--target-sa-check-interval", [&](const std::string& v) {
        opt.target_sa_check_interval = parse_int(v);
    });
    handlers.with_value.emplace("--seed", [&](const std::string& v) {
        opt.seed = parse_u64(v);
    });
    handlers.with_value.emplace("--spacing-safety", [&](const std::string& v) {
        opt.spacing_safety = parse_double(v);
    });
    handlers.with_value.emplace("--shift-a", [&](const std::string& v) {
        opt.shift_a = parse_double(v);
    });
    handlers.with_value.emplace("--shift-b", [&](const std::string& v) {
        opt.shift_b = parse_double(v);
    });
    handlers.with_value.emplace("--shift", [&](const std::string& v) { parse_shift(opt, v); });
    handlers.with_value.emplace("--shift-search", [&](const std::string& v) {
        opt.shift_search = parse_shift_search(v);
    });
    handlers.with_value.emplace("--shift-grid", [&](const std::string& v) {
        opt.shift_grid = parse_int(v);
    });
    handlers.with_value.emplace("--shift-levels", [&](const std::string& v) {
        opt.shift_levels = parse_int(v);
    });
    handlers.with_value.emplace("--shift-keep", [&](const std::string& v) {
        opt.shift_keep = parse_int(v);
    });
    handlers.with_value.emplace("--shift-pool-size", [&](const std::string& v) {
        opt.shift_pool_size = parse_int(v);
    });
    handlers.with_value.emplace("--angles", [&](const std::string& v) { parse_angles(opt, v); });
    handlers.with_value.emplace("--output", [&](const std::string& v) { opt.output_path = v; });
    handlers.flags.emplace("--no-prune", [&]() { opt.prune = false; });

    return handlers;
}

void validate_options(const Options& opt) {
    validate_tile_options(opt);
    validate_lattice_options(opt);
    validate_shift_options(opt);
    validate_sa_options(opt);
    validate_ils_options(opt);
    validate_mz_options(opt);
    validate_micro_options(opt);
    validate_target_options(opt);
}

}  // namespace

Options parse_args(int argc, char** argv) {
    Options opt;
    opt.angle_candidates = {0.0, 3.0, 6.0, 9.0, 12.0, 15.0,
                            18.0, 21.0, 24.0, 27.0, 30.0};

    ArgHandlers handlers = make_arg_handlers(opt);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto with_value = handlers.with_value.find(arg);
        if (with_value != handlers.with_value.end()) {
            with_value->second(require_arg(i, argc, argv, arg));
            continue;
        }
        auto flag = handlers.flags.find(arg);
        if (flag != handlers.flags.end()) {
            flag->second();
            continue;
        }
        throw std::runtime_error("Unknown argument: " + arg);
    }

    validate_options(opt);
    return opt;
}
