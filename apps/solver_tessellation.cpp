#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils/cli_parse.hpp"
#include "geometry/collision.hpp"
#include "solvers/compaction_contact.hpp"
#include "solvers/ga.hpp"
#include "geometry/geom.hpp"
#include "solvers/sa.hpp"
#include "utils/score.hpp"
#include "utils/submission_io.hpp"
#include "utils/wrap_utils.hpp"

namespace {

constexpr double kSqrt3 = 1.732050807568877293527446341505872366942805254;
constexpr int kOutputDecimals = 9;

struct Options {
    int n_max = 200;
    uint64_t seed = 123456789ULL;
    double spacing_safety = 1.001;
    double shift_a = 0.0;
    double shift_b = 0.0;
    std::vector<double> angle_candidates = {0.0, 15.0, 30.0, 45.0, 60.0, 75.0};
    std::string output_path = "submission_tessellation_cpp.csv";
    bool use_ga = false;
    int ga_pop = 40;
    int ga_gens = 60;
    int ga_elite = 2;
    int ga_tournament = 3;
    double ga_spacing_min = 1.000;
    double ga_spacing_max = 1.010;
    std::vector<double> ga_rot_candidates = {0.0, 180.0};
    int sa_restarts = 0;
    int sa_base_iters = 0;
    int sa_iters_per_n = 0;
    double sa_w_micro = 1.0;
    double sa_w_swap_rot = 0.25;
    double sa_w_relocate = 0.15;
    double sa_w_block_translate = 0.05;
    double sa_w_block_rotate = 0.02;
    double sa_w_lns = 0.001;
    double sa_w_push_contact = 0.0;
    double sa_w_slide_contact = 0.0;
    double sa_w_squeeze = 0.0;
    int sa_block_size = 6;
    int sa_lns_remove = 6;
    int sa_lns_candidates = 1;
    int sa_lns_eval_attempts_per_tree = 0;
    int sa_hh_segment = 50;
    double sa_hh_reaction = 0.20;
    bool sa_hh_auto = false;
    SARefiner::OverlapMetric sa_overlap_metric = SARefiner::OverlapMetric::kArea;
    double sa_overlap_weight = 0.0;
    double sa_overlap_weight_start = -1.0;
    double sa_overlap_weight_end = -1.0;
    double sa_overlap_weight_power = 1.0;
    bool sa_overlap_weight_geometric = false;
    double sa_overlap_eps_area = 1e-12;
    double sa_overlap_cost_cap = 0.0;
    double sa_plateau_eps = 0.0;
    double sa_w_resolve_overlap = 0.0;
    int sa_resolve_attempts = 6;
    double sa_resolve_step_frac_max = 0.20;
    double sa_resolve_step_frac_min = 0.02;
    double sa_resolve_noise_frac = 0.05;
    double sa_push_max_step_frac = 0.60;
    int sa_push_bisect_iters = 10;
    double sa_push_overshoot_frac = 0.0;
    int sa_slide_dirs = 8;
    double sa_slide_dir_bias = 2.0;
    double sa_slide_max_step_frac = 0.60;
    int sa_slide_bisect_iters = 10;
    double sa_slide_min_gain = 1e-4;
    double sa_slide_schedule_max_frac = 0.0;
    int sa_squeeze_pushes = 6;
    bool sa_aggressive = false;
    bool final_rigid = true;
    bool target_refine = false;
    double target_cover = 0.78;
    int target_m_min = 24;
    int target_m_max = 64;
    int target_m = 0;
    int target_tier_a = 12;
    int target_tier_b = 24;
    double target_budget_scale = 1.0;
    bool target_soft_overlap = false;
    bool target_soft_overlap_tier_a_only = true;
    double target_soft_overlap_cut = 0.8;
    bool target_early_stop = true;
    int target_sa_check_interval = 200;
};

std::string to_lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

void apply_preset(Options& opt, const std::string& name) {
    std::string preset = to_lower_ascii(name);
    if (preset == "fast") {
        preset = "quick";
    } else if (preset == "slow") {
        preset = "quality";
    } else if (preset == "medium") {
        preset = "balanced";
    } else if (preset == "moderate") {
        preset = "balanced";
    }

    if (preset == "quick") {
        opt.use_ga = false;
        opt.angle_candidates = {0.0, 15.0, 30.0, 45.0, 60.0, 75.0};
        opt.sa_restarts = 0;
        opt.sa_base_iters = 0;
        opt.sa_iters_per_n = 0;
        opt.sa_w_push_contact = 0.0;
        opt.sa_w_slide_contact = 0.0;
        opt.sa_w_squeeze = 0.0;
        opt.sa_squeeze_pushes = 6;
        opt.sa_aggressive = false;
        opt.final_rigid = true;
        return;
    }
    if (preset == "balanced") {
        opt.use_ga = false;
        opt.angle_candidates = {0.0, 15.0, 30.0, 45.0, 60.0, 75.0};
        opt.sa_restarts = 2;
        opt.sa_base_iters = 200;
        opt.sa_iters_per_n = 10;
        opt.sa_w_push_contact = 0.1;
        opt.sa_w_slide_contact = 0.0;
        opt.sa_w_squeeze = 0.02;
        opt.sa_squeeze_pushes = 4;
        opt.sa_aggressive = false;
        opt.final_rigid = true;
        return;
    }
    if (preset == "quality") {
        opt.use_ga = true;
        opt.angle_candidates = {0.0, 7.5, 15.0, 22.5, 30.0, 37.5,
                                45.0, 52.5, 60.0, 67.5, 75.0};
        opt.sa_restarts = 3;
        opt.sa_base_iters = 500;
        opt.sa_iters_per_n = 20;
        opt.sa_w_push_contact = 0.2;
        opt.sa_w_slide_contact = 0.0;
        opt.sa_w_squeeze = 0.05;
        opt.sa_squeeze_pushes = 6;
        opt.sa_aggressive = false;
        opt.final_rigid = true;
        return;
    }

    throw std::runtime_error("--preset must be quick|balanced|quality.");
}

void apply_presets_from_args(Options& opt, int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--preset") {
            std::string preset = require_arg(i, argc, argv, arg);
            apply_preset(opt, preset);
        }
    }
}

struct Candidate {
    TreePose pose;
    double key1;
    double key2;
};

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
    std::string mode = to_lower_ascii(s);
    if (mode == "auto") {
        opt.sa_hh_auto = true;
        return;
    }
    if (mode == "off" || mode == "default") {
        opt.sa_hh_auto = false;
        return;
    }
    throw std::runtime_error("--sa-hh-mode must be 'off' or 'auto'.");
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

void parse_ga_rots(Options& opt, const std::string& s) {
    opt.ga_rot_candidates = parse_double_list(s);
    if (opt.ga_rot_candidates.empty()) {
        throw std::runtime_error("--ga-rots cannot be empty.");
    }
}

void ensure(bool cond, const char* message) {
    if (!cond) {
        throw std::runtime_error(message);
    }
}

void validate_basic_options(const Options& opt) {
    ensure(opt.n_max > 0 && opt.n_max <= 200, "--n-max must be in [1, 200].");
    ensure(opt.spacing_safety >= 1.0, "--spacing-safety must be >= 1.0.");
}

void validate_ga_options(const Options& opt) {
    ensure(opt.ga_pop > 0, "--ga-pop must be > 0.");
    ensure(opt.ga_gens >= 0, "--ga-gens must be >= 0.");
    ensure(opt.ga_elite >= 0 && opt.ga_elite <= opt.ga_pop, "--ga-elite must be in [0, ga-pop].");
    ensure(opt.ga_tournament > 0, "--ga-tournament must be > 0.");
    ensure(opt.ga_spacing_min >= 1.0 && opt.ga_spacing_max >= opt.ga_spacing_min,
           "--ga-spacing-min/max invalid (min>=1 and max>=min required).");
}

void validate_sa_options(const Options& opt) {
    ensure(opt.sa_restarts >= 0 && opt.sa_base_iters >= 0 && opt.sa_iters_per_n >= 0,
           "SA parameters must be >= 0.");
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
           "--sa-resolve-step-frac-min/max invalid.");
    ensure(opt.sa_resolve_noise_frac >= 0.0, "--sa-resolve-noise-frac must be >= 0.");
    ensure(opt.sa_push_max_step_frac > 0.0, "--sa-push-max-step-frac must be > 0.");
    ensure(opt.sa_push_bisect_iters > 0, "--sa-push-bisect-iters must be > 0.");
    ensure(opt.sa_push_overshoot_frac >= 0.0 && opt.sa_push_overshoot_frac <= 1.0,
           "--sa-push-overshoot-frac must be in [0, 1].");
    ensure(opt.sa_slide_dirs == 4 || opt.sa_slide_dirs == 8, "--sa-slide-dirs must be 4 or 8.");
    ensure(opt.sa_slide_dir_bias >= 0.0, "--sa-slide-dir-bias must be >= 0.");
    ensure(opt.sa_slide_max_step_frac > 0.0, "--sa-slide-max-step-frac must be > 0.");
    ensure(opt.sa_slide_bisect_iters > 0, "--sa-slide-bisect-iters must be > 0.");
    ensure(opt.sa_slide_min_gain >= 0.0, "--sa-slide-min-gain must be >= 0.");
    ensure(opt.sa_slide_schedule_max_frac >= 0.0 && opt.sa_slide_schedule_max_frac < 1.0,
           "--sa-slide-schedule-max-frac must be in [0, 1).");
    ensure(opt.sa_squeeze_pushes >= 0, "--sa-squeeze-pushes must be >= 0.");
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

void validate_options(const Options& opt) {
    validate_basic_options(opt);
    validate_ga_options(opt);
    validate_sa_options(opt);
    validate_target_options(opt);
}

struct ArgHandlers {
    std::unordered_map<std::string, std::function<void(const std::string&)>> with_value;
    std::unordered_map<std::string, std::function<void()>> flags;
};

ArgHandlers make_arg_handlers(Options& opt) {
    ArgHandlers handlers;

    handlers.with_value.emplace("--n-max", [&](const std::string& v) { opt.n_max = parse_int(v); });
    handlers.with_value.emplace("--seed", [&](const std::string& v) { opt.seed = parse_u64(v); });
    handlers.with_value.emplace("--spacing-safety", [&](const std::string& v) {
        opt.spacing_safety = parse_double(v);
    });
    handlers.with_value.emplace("--preset", [&](const std::string&) {});
    handlers.flags.emplace("--use-ga", [&]() { opt.use_ga = true; });
    handlers.with_value.emplace("--ga-pop", [&](const std::string& v) { opt.ga_pop = parse_int(v); });
    handlers.with_value.emplace("--ga-gens", [&](const std::string& v) { opt.ga_gens = parse_int(v); });
    handlers.with_value.emplace("--ga-elite", [&](const std::string& v) { opt.ga_elite = parse_int(v); });
    handlers.with_value.emplace("--ga-tournament", [&](const std::string& v) {
        opt.ga_tournament = parse_int(v);
    });
    handlers.with_value.emplace("--ga-spacing-min", [&](const std::string& v) {
        opt.ga_spacing_min = parse_double(v);
    });
    handlers.with_value.emplace("--ga-spacing-max", [&](const std::string& v) {
        opt.ga_spacing_max = parse_double(v);
    });
    handlers.with_value.emplace("--ga-rots", [&](const std::string& v) { parse_ga_rots(opt, v); });
    handlers.with_value.emplace("--shift-a", [&](const std::string& v) { opt.shift_a = parse_double(v); });
    handlers.with_value.emplace("--shift-b", [&](const std::string& v) { opt.shift_b = parse_double(v); });
    handlers.with_value.emplace("--shift", [&](const std::string& v) { parse_shift(opt, v); });
    handlers.with_value.emplace("--angles", [&](const std::string& v) { parse_angles(opt, v); });
    handlers.with_value.emplace("--output", [&](const std::string& v) { opt.output_path = v; });
    handlers.with_value.emplace("--sa-restarts", [&](const std::string& v) {
        opt.sa_restarts = parse_int(v);
    });
    handlers.with_value.emplace("--sa-base-iters", [&](const std::string& v) {
        opt.sa_base_iters = parse_int(v);
    });
    handlers.with_value.emplace("--sa-iters-per-n", [&](const std::string& v) {
        opt.sa_iters_per_n = parse_int(v);
    });
    handlers.with_value.emplace("--sa-w-micro", [&](const std::string& v) { opt.sa_w_micro = parse_double(v); });
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
    handlers.with_value.emplace("--sa-w-lns", [&](const std::string& v) { opt.sa_w_lns = parse_double(v); });
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
    handlers.flags.emplace("--sa-overlap-weight-geometric", [&]() { opt.sa_overlap_weight_geometric = true; });
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
    handlers.flags.emplace("--sa-aggressive", [&]() { opt.sa_aggressive = true; });
    handlers.flags.emplace("--no-final-rigid", [&]() { opt.final_rigid = false; });
    handlers.flags.emplace("--no-sa-rigid", [&]() { opt.final_rigid = false; });

    return handlers;
}

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
                      double eps = 1e-9) {
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

double find_safe_hex_spacing_upper_bound(const Polygon& base_poly,
                                         double radius,
                                         double eps = 1e-9) {
    double hi = 2.0 * radius;
    if (safe_hex_spacing(base_poly, radius, hi, eps)) {
        return hi;
    }

    // By definition of radius as the enclosing circle, hi should be safe.
    // Keep a fallback loop for robustness.
    for (int it = 0; it < 30; ++it) {
        hi *= 1.5;
        if (safe_hex_spacing(base_poly, radius, hi, eps)) {
            break;
        }
    }
    return hi;
}

double find_min_safe_hex_spacing(const Polygon& base_poly,
                                double radius,
                                double eps = 1e-9) {
    double lo = 0.0;
    double hi = find_safe_hex_spacing_upper_bound(base_poly, radius, eps);

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

std::vector<TreePose> generate_hex_lattice_poses(int n,
                                                 double spacing,
                                                 double angle_deg,
                                                 double shift_a,
                                                 double shift_b) {
    if (n <= 0) {
        return {};
    }

    auto [u, v] = hex_basis(spacing, angle_deg);

    int m = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(n)))) + 8;

    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<size_t>((2 * m + 1) * (2 * m + 1)));
    for (int i = -m; i <= m; ++i) {
        for (int j = -m; j <= m; ++j) {
            double ci = static_cast<double>(i) - shift_a;
            double cj = static_cast<double>(j) - shift_b;
            double x = ci * u.x + cj * v.x;
            double y = ci * u.y + cj * v.y;
            double key1 = std::max(std::abs(x), std::abs(y));
            double key2 = std::hypot(x, y);
            candidates.push_back({TreePose{x, y, rotation_pattern_deg(i, j)},
                                  key1,
                                  key2});
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

Options parse_args(int argc, char** argv) {
    Options opt;

    apply_presets_from_args(opt, argc, argv);

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

struct GeometryContext {
    const Polygon& base_poly;
    double radius;
};

struct LatticeContext {
    GeometryContext geom;
    double spacing;
    double shift_a;
    double shift_b;
    const std::vector<double>& angle_candidates;
};

struct GAContext {
    GeometryContext geom;
    const Options& opt;
};

struct SAContext {
    GeometryContext geom;
    SARefiner& sa;
    const Options& opt;
    int n;
    SARefiner::HHState* hh_state;
};

struct RigidContext {
    GeometryContext geom;
    const Options& opt;
};

struct BestSolution {
    double score = std::numeric_limits<double>::infinity();
    std::vector<TreePose> poses;
};

void consider_candidate(BestSolution& best,
                        double score,
                        std::vector<TreePose> poses,
                        double eps = 0.0) {
    if (score + eps < best.score) {
        best.score = score;
        best.poses = std::move(poses);
    }
}

BestSolution best_lattice_solution(const LatticeContext& ctx, int n) {
    BestSolution best;
    for (double angle_deg : ctx.angle_candidates) {
        auto poses = generate_hex_lattice_poses(
            n, ctx.spacing, angle_deg, ctx.shift_a, ctx.shift_b);
        if (any_overlap(ctx.geom.base_poly, poses, ctx.geom.radius)) {
            continue;
        }
        auto poses_q = quantize_poses(poses);
        if (any_overlap(ctx.geom.base_poly, poses_q, ctx.geom.radius)) {
            continue;
        }
        double score_n = score_instance(ctx.geom.base_poly, poses_q);
        consider_candidate(best, score_n, std::move(poses_q));
    }
    return best;
}

void consider_ga_prefix(const GeometryContext& geom,
                        const std::vector<TreePose>& ga_poses_sorted,
                        int n,
                        BestSolution& best) {
    if (ga_poses_sorted.empty() || static_cast<int>(ga_poses_sorted.size()) < n) {
        return;
    }
    std::vector<TreePose> ga_prefix(ga_poses_sorted.begin(), ga_poses_sorted.begin() + n);
    double score_n = score_instance(geom.base_poly, ga_prefix);
    consider_candidate(best, score_n, std::move(ga_prefix), 1e-15);
}

bool pose_prefix_less(const TreePose& a, const TreePose& b) {
    double a1 = std::max(std::abs(a.x), std::abs(a.y));
    double b1 = std::max(std::abs(b.x), std::abs(b.y));
    if (a1 != b1) {
        return a1 < b1;
    }
    double a2 = std::hypot(a.x, a.y);
    double b2 = std::hypot(b.x, b.y);
    if (a2 != b2) {
        return a2 < b2;
    }
    if (a.x != b.x) {
        return a.x < b.x;
    }
    if (a.y != b.y) {
        return a.y < b.y;
    }
    return a.deg < b.deg;
}

std::vector<TreePose> maybe_run_ga(const GAContext& ctx) {
    if (!ctx.opt.use_ga) {
        return {};
    }

    GAParams gp;
    gp.pop_size = ctx.opt.ga_pop;
    gp.generations = ctx.opt.ga_gens;
    gp.elite = ctx.opt.ga_elite;
    gp.tournament_k = ctx.opt.ga_tournament;
    gp.spacing_safety_min = ctx.opt.ga_spacing_min;
    gp.spacing_safety_max = ctx.opt.ga_spacing_max;
    gp.lattice_angle_candidates = ctx.opt.angle_candidates;
    gp.rotation_candidates = ctx.opt.ga_rot_candidates;

    GlobalSearchGA ga(ctx.geom.base_poly, ctx.geom.radius);
    GAResult gr = ga.solve(
        ctx.opt.n_max,
        ctx.opt.seed ^ 0xA7F4C3B2D1E0F987ULL,
        gp);

    if (!gr.best_poses.empty() && std::isfinite(gr.best_side)) {
        auto q = quantize_poses(gr.best_poses);
        if (!any_overlap(ctx.geom.base_poly, q, ctx.geom.radius)) {
            std::sort(q.begin(), q.end(), pose_prefix_less);
            std::cout << "GA (n_max=" << ctx.opt.n_max << ") best_side: "
                      << std::fixed << std::setprecision(9) << gr.best_side
                      << " (spacing=" << gr.best_spacing
                      << ", angle=" << gr.best_angle_deg
                      << ", shift=" << gr.best_shift_a << "," << gr.best_shift_b
                      << ")\n";
            return q;
        }
        std::cerr << "Warning: GA produced overlap after quantization; ignoring GA.\n";
        return {};
    }

    std::cerr << "Warning: GA did not find a solution; ignoring GA.\n";
    return {};
}

bool sa_enabled(const Options& opt) {
    return opt.sa_restarts > 0 && opt.sa_base_iters > 0;
}

size_t hh_bucket_for_n(int n) {
    if (n <= 25) {
        return 0;
    }
    if (n <= 80) {
        return 1;
    }
    return 2;
}

SARefiner::Params make_sa_params(const Options& opt, int n) {
    SARefiner::Params p;
    p.iters = opt.sa_base_iters + opt.sa_iters_per_n * n;
    p.w_micro = opt.sa_w_micro;
    p.w_swap_rot = opt.sa_w_swap_rot;
    p.w_relocate = opt.sa_w_relocate;
    p.w_block_translate = opt.sa_w_block_translate;
    p.w_block_rotate = opt.sa_w_block_rotate;
    p.w_lns = opt.sa_w_lns;
    p.w_push_contact = opt.sa_w_push_contact;
    p.w_slide_contact = opt.sa_w_slide_contact;
    p.w_squeeze = opt.sa_w_squeeze;
    p.block_size = opt.sa_block_size;
    p.lns_remove = opt.sa_lns_remove;
    p.lns_candidates = opt.sa_lns_candidates;
    p.lns_eval_attempts_per_tree = opt.sa_lns_eval_attempts_per_tree;
    p.hh_segment = opt.sa_hh_segment;
    p.hh_reaction = opt.sa_hh_reaction;
    p.hh_auto = opt.sa_hh_auto;
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
    p.slide_dirs = opt.sa_slide_dirs;
    p.slide_dir_bias = opt.sa_slide_dir_bias;
    p.slide_max_step_frac = opt.sa_slide_max_step_frac;
    p.slide_bisect_iters = opt.sa_slide_bisect_iters;
    p.slide_min_gain = opt.sa_slide_min_gain;
    p.slide_schedule_max_frac = opt.sa_slide_schedule_max_frac;
    p.squeeze_pushes = opt.sa_squeeze_pushes;
    if (opt.sa_aggressive) {
        SARefiner::apply_aggressive_preset(p);
    }
    if (opt.sa_hh_auto) {
        SARefiner::apply_hh_auto_preset(p);
    }
    return p;
}

void apply_sa_refinement(const SAContext& ctx, BestSolution& best) {
    if (!sa_enabled(ctx.opt) || best.poses.empty()) {
        return;
    }

    SARefiner::Params p = make_sa_params(ctx.opt, ctx.n);
    for (int r = 0; r < ctx.opt.sa_restarts; ++r) {
        uint64_t seed =
            ctx.opt.seed ^
            (0x9e3779b97f4a7c15ULL +
             static_cast<uint64_t>(ctx.n) * 0xbf58476d1ce4e5b9ULL +
             static_cast<uint64_t>(r) * 0x94d049bb133111ebULL);
        SARefiner::Result res =
            ctx.sa.refine_min_side(best.poses, seed, p, nullptr, ctx.hh_state);
        auto cand_q = quantize_poses(res.best_poses);
        if (any_overlap(ctx.geom.base_poly, cand_q, ctx.geom.radius)) {
            continue;
        }
        double cand_score_n = score_instance(ctx.geom.base_poly, cand_q);
        consider_candidate(best, cand_score_n, std::move(cand_q), 1e-15);
    }
}

void apply_final_rigid(const RigidContext& ctx, BestSolution& best) {
    if (!ctx.opt.final_rigid || best.poses.empty()) {
        return;
    }

    std::vector<TreePose> rigid_sol = best.poses;
    optimize_rigid_rotation(ctx.geom.base_poly, rigid_sol);
    auto rigid_q = quantize_poses(rigid_sol);
    if (any_overlap(ctx.geom.base_poly, rigid_q, ctx.geom.radius)) {
        return;
    }
    double rigid_score_n = score_instance(ctx.geom.base_poly, rigid_q);
    consider_candidate(best, rigid_score_n, std::move(rigid_q), 1e-15);
}

enum class TargetTier {
    kA,
    kB,
    kC,
};

struct SideArea {
    double side = 0.0;
    double area = 0.0;
};

struct TargetEntry {
    int n = 0;
    double term = 0.0;
    TargetTier tier = TargetTier::kC;
};

struct TierEarlyStop {
    int min_passes = 0;
    int patience_passes = 0;
    int min_iters = 0;
    int patience_iters = 0;
};

int clamp_int(int value, int lo, int hi) {
    return std::min(hi, std::max(lo, value));
}

double normalized_budget_scale(double scale) {
    return (scale > 0.0) ? scale : 1.0;
}

SideArea eval_side_area(const Polygon& base_poly,
                        const std::vector<TreePose>& poses) {
    if (poses.empty()) {
        return {};
    }
    double min_x = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    for (const auto& pose : poses) {
        BoundingBox bb = bounding_box(transform_polygon(base_poly, pose));
        min_x = std::min(min_x, bb.min_x);
        max_x = std::max(max_x, bb.max_x);
        min_y = std::min(min_y, bb.min_y);
        max_y = std::max(max_y, bb.max_y);
    }
    const double width = max_x - min_x;
    const double height = max_y - min_y;
    SideArea out;
    out.side = std::max(width, height);
    out.area = width * height;
    return out;
}

bool improves_side_area(const SideArea& cand,
                        const SideArea& best,
                        double plateau_eps) {
    if (!std::isfinite(best.side)) {
        return std::isfinite(cand.side);
    }
    if (cand.side < best.side - 1e-12) {
        return true;
    }
    if (cand.side <= best.side + plateau_eps && cand.area < best.area - 1e-12) {
        return true;
    }
    return false;
}

TargetTier tier_for_rank(int idx, int tier_a, int tier_b) {
    if (idx < tier_a) {
        return TargetTier::kA;
    }
    if (idx < tier_a + tier_b) {
        return TargetTier::kB;
    }
    return TargetTier::kC;
}

int scale_count(int base, double scale, int min_floor, int max_cap) {
    const double s = normalized_budget_scale(scale);
    int scaled = static_cast<int>(std::round(static_cast<double>(base) * s));
    scaled = std::max(min_floor, scaled);
    if (max_cap > 0) {
        scaled = std::min(max_cap, scaled);
    }
    return scaled;
}

TierEarlyStop early_stop_defaults(TargetTier tier) {
    switch (tier) {
    case TargetTier::kA:
        return TierEarlyStop{6, 3, 3000, 1200};
    case TargetTier::kB:
        return TierEarlyStop{4, 2, 1500, 800};
    case TargetTier::kC:
        return TierEarlyStop{3, 2, 800, 600};
    }
    return TierEarlyStop{};
}

std::vector<TargetEntry> select_target_entries(
    const Polygon& base_poly,
    const std::vector<std::vector<TreePose>>& solutions_by_n,
    const Options& opt,
    double* total_score_out) {
    struct TermRow {
        int n = 0;
        double term = 0.0;
    };
    std::vector<TermRow> rows;
    rows.reserve(static_cast<size_t>(opt.n_max));
    double total_score = 0.0;
    for (int n = 1; n <= opt.n_max; ++n) {
        const auto& sol = solutions_by_n[static_cast<size_t>(n)];
        if (static_cast<int>(sol.size()) != n) {
            continue;
        }
        auto sol_q = quantize_poses(sol);
        const double term = score_instance(base_poly, sol_q);
        total_score += term;
        rows.push_back(TermRow{n, term});
    }
    if (total_score_out) {
        *total_score_out = total_score;
    }
    if (rows.empty()) {
        return {};
    }
    std::sort(rows.begin(), rows.end(), [](const TermRow& a, const TermRow& b) {
        return a.term > b.term;
    });

    int target_count = 0;
    if (opt.target_m > 0) {
        target_count = clamp_int(opt.target_m, opt.target_m_min, opt.target_m_max);
    } else {
        const double cover = std::max(0.0, std::min(1.0, opt.target_cover));
        const double target_score = cover * total_score;
        double acc = 0.0;
        for (const auto& row : rows) {
            acc += row.term;
            ++target_count;
            if (acc >= target_score) {
                break;
            }
        }
        target_count = clamp_int(target_count, opt.target_m_min, opt.target_m_max);
    }
    const double scale = normalized_budget_scale(opt.target_budget_scale);
    if (std::abs(scale - 1.0) > 1e-12) {
        int scaled = static_cast<int>(std::round(target_count * std::sqrt(scale)));
        target_count = clamp_int(scaled, opt.target_m_min, opt.target_m_max);
    }
    target_count = std::min(target_count, static_cast<int>(rows.size()));

    const int tier_a = std::min(opt.target_tier_a, target_count);
    const int tier_b =
        std::min(opt.target_tier_b, std::max(0, target_count - tier_a));

    std::vector<TargetEntry> entries;
    entries.reserve(static_cast<size_t>(target_count));
    for (int i = 0; i < target_count; ++i) {
        entries.push_back(TargetEntry{
            rows[static_cast<size_t>(i)].n,
            rows[static_cast<size_t>(i)].term,
            tier_for_rank(i, tier_a, tier_b)});
    }
    return entries;
}

void apply_compact_early_stop(compaction_contact::Params& params,
                              TargetTier tier,
                              const Options& opt) {
    if (!opt.target_early_stop) {
        params.early_stop.enabled = false;
        return;
    }
    TierEarlyStop base = early_stop_defaults(tier);
    params.early_stop.enabled = true;
    params.early_stop.min_passes =
        scale_count(base.min_passes, opt.target_budget_scale, 2, params.passes);
    params.early_stop.patience_passes =
        scale_count(base.patience_passes, opt.target_budget_scale, 1, params.passes);
}

void apply_sa_early_stop(SARefiner::Params& params,
                         TargetTier tier,
                         const Options& opt) {
    if (!opt.target_early_stop) {
        params.early_stop = false;
        return;
    }
    TierEarlyStop base = early_stop_defaults(tier);
    params.early_stop = true;
    params.early_stop_check_interval = opt.target_sa_check_interval;
    params.early_stop_min_iters =
        scale_count(base.min_iters, opt.target_budget_scale, 400, params.iters);
    params.early_stop_patience_iters =
        scale_count(base.patience_iters, opt.target_budget_scale, 200, params.iters);
}

compaction_contact::Params make_compact_params(TargetTier tier, const Options& opt) {
    compaction_contact::Params params;
    params.passes = 12;
    params.attempts_per_pass = 32;
    params.patience = 4;
    params.boundary_topk = 14;
    params.push_bisect_iters = 12;
    params.push_max_step_frac = 0.9;
    params.plateau_eps = 1e-4;
    params.alt_axis = true;
    params.final_rigid = false;
    params.quantize_decimals = kOutputDecimals;
    apply_compact_early_stop(params, tier, opt);
    return params;
}

double tier_scale(TargetTier tier) {
    switch (tier) {
    case TargetTier::kA:
        return 1.0;
    case TargetTier::kB:
        return 0.70;
    case TargetTier::kC:
        return 0.50;
    }
    return 1.0;
}

SARefiner::Params make_targeted_sa_params(TargetTier tier,
                                          const Options& opt,
                                          int n) {
    SARefiner::Params p;
    const int base_iters = 2500 + 10 * n;
    const double scale = normalized_budget_scale(opt.target_budget_scale);
    p.iters = static_cast<int>(std::round(base_iters * tier_scale(tier) * scale));
    p.t0 = 0.15;
    p.t1 = 0.01;
    p.w_micro = 0.8;
    p.w_swap_rot = 0.25;
    p.w_relocate = 0.15;
    p.w_block_translate = (tier == TargetTier::kA) ? 0.08 : (tier == TargetTier::kB ? 0.06 : 0.04);
    p.w_block_rotate = (tier == TargetTier::kA) ? 0.03 : (tier == TargetTier::kB ? 0.02 : 0.015);
    p.w_lns = (tier == TargetTier::kA) ? 0.02 : (tier == TargetTier::kB ? 0.015 : 0.01);
    p.w_push_contact = (tier == TargetTier::kA) ? 0.20 : (tier == TargetTier::kB ? 0.15 : 0.10);
    p.w_squeeze = (tier == TargetTier::kA) ? 0.08 : (tier == TargetTier::kB ? 0.06 : 0.04);
    p.block_size = 6;
    p.lns_remove = std::max(6, n / 25);
    p.lns_attempts_per_tree = (tier == TargetTier::kA) ? 40 : (tier == TargetTier::kB ? 30 : 20);
    p.lns_candidates = (tier == TargetTier::kA) ? 3 : (tier == TargetTier::kB ? 2 : 1);
    p.lns_p_contact = 0.55;
    p.lns_p_uniform = 0.15;
    p.push_bisect_iters = 12;
    p.push_max_step_frac = 0.9;
    p.plateau_eps = 1e-4;
    p.quantize_decimals = kOutputDecimals;
    p.hh_auto = opt.sa_hh_auto;
    if (opt.sa_hh_auto) {
        SARefiner::apply_hh_auto_preset(p);
    }
    return p;
}

bool use_soft_overlap(TargetTier tier, const Options& opt) {
    if (!opt.target_soft_overlap) {
        return false;
    }
    if (!opt.target_soft_overlap_tier_a_only) {
        return true;
    }
    return tier == TargetTier::kA;
}

std::vector<TreePose> run_targeted_sa(SARefiner& sa,
                                      const std::vector<TreePose>& start,
                                      const Options& opt,
                                      TargetTier tier,
                                      int n,
                                      uint64_t seed_base,
                                      SARefiner::HHState* hh_state) {
    SARefiner::Params base = make_targeted_sa_params(tier, opt, n);
    if (base.iters <= 0) {
        return start;
    }
    SARefiner::HHState* state = opt.sa_hh_auto ? hh_state : nullptr;

    if (use_soft_overlap(tier, opt)) {
        const int total_iters = base.iters;
        const int soft_iters = std::max(
            1, std::min(total_iters,
                        static_cast<int>(std::round(total_iters * opt.target_soft_overlap_cut))));
        const int hard_iters = total_iters - soft_iters;

        std::vector<TreePose> seed_poses = start;
        if (soft_iters > 0) {
            SARefiner::Params soft = base;
            soft.iters = soft_iters;
            soft.overlap_metric = SARefiner::OverlapMetric::kArea;
            soft.overlap_weight = 0.2;
            soft.overlap_weight_start = 0.2;
            soft.overlap_weight_end = 2e4;
            soft.overlap_weight_geometric = true;
            soft.w_resolve_overlap = 0.05;
            soft.push_overshoot_frac = 0.10;
            apply_sa_early_stop(soft, tier, opt);
            SARefiner::Result res =
                sa.refine_min_side(seed_poses,
                                   seed_base ^ 0xB5C8D57A3E4F29B1ULL,
                                   soft,
                                   nullptr,
                                   state);
            seed_poses = res.best_poses;
        }

        if (hard_iters > 0) {
            SARefiner::Params hard = base;
            hard.iters = hard_iters;
            hard.overlap_weight = 0.0;
            hard.overlap_weight_start = -1.0;
            hard.overlap_weight_end = -1.0;
            hard.overlap_weight_geometric = false;
            hard.push_overshoot_frac = 0.0;
            hard.w_resolve_overlap = 0.0;
            apply_sa_early_stop(hard, tier, opt);
            SARefiner::Result res =
                sa.refine_min_side(seed_poses,
                                   seed_base ^ 0x8D1F5A9C7E3B2A41ULL,
                                   hard,
                                   nullptr,
                                   state);
            return res.best_poses;
        }
        return seed_poses;
    }

    apply_sa_early_stop(base, tier, opt);
    SARefiner::Result res = sa.refine_min_side(start, seed_base, base, nullptr, state);
    return res.best_poses;
}

double recompute_total_score(const Polygon& base_poly,
                             const std::vector<std::vector<TreePose>>& solutions_by_n,
                             int n_max) {
    double total = 0.0;
    for (int n = 1; n <= n_max; ++n) {
        const auto& sol = solutions_by_n[static_cast<size_t>(n)];
        if (static_cast<int>(sol.size()) != n) {
            continue;
        }
        total += score_instance(base_poly, sol);
    }
    return total;
}

void run_targeted_refine(const GeometryContext& geom,
                         std::vector<std::vector<TreePose>>& solutions_by_n,
                         const Options& opt,
                         double& total_score) {
    if (!opt.target_refine) {
        return;
    }

    double total_score_est = 0.0;
    std::vector<TargetEntry> targets =
        select_target_entries(geom.base_poly, solutions_by_n, opt, &total_score_est);
    if (targets.empty()) {
        return;
    }

    const double plateau_eps = 1e-4;
    SARefiner sa(geom.base_poly, geom.radius);
    std::array<SARefiner::HHState, 3> hh_states;
    int improved_count = 0;
    int tier_a = 0;
    int tier_b = 0;
    for (const auto& entry : targets) {
        auto& sol = solutions_by_n[static_cast<size_t>(entry.n)];
        if (static_cast<int>(sol.size()) != entry.n) {
            continue;
        }

        if (entry.tier == TargetTier::kA) {
            ++tier_a;
        } else if (entry.tier == TargetTier::kB) {
            ++tier_b;
        }

        auto base_q = quantize_poses(sol);
        if (any_overlap(geom.base_poly, base_q, geom.radius)) {
            continue;
        }

        SideArea best_eval = eval_side_area(geom.base_poly, base_q);
        SideArea orig_eval = best_eval;
        std::vector<TreePose> best_sol = base_q;

        uint64_t seed = opt.seed ^
                        (0xC6BC279692B5CC83ULL +
                         static_cast<uint64_t>(entry.n) * 0x9E3779B97F4A7C15ULL);
        std::mt19937_64 rng(seed);
        compaction_contact::Params cparams = make_compact_params(entry.tier, opt);
        std::vector<TreePose> cand = best_sol;
        compaction_contact::Stats cstats =
            compaction_contact::compact_contact(geom.base_poly, cand, cparams, rng);
        if (cstats.ok) {
            auto cand_q = quantize_poses(cand);
            if (!any_overlap(geom.base_poly, cand_q, geom.radius)) {
                SideArea cand_eval = eval_side_area(geom.base_poly, cand_q);
                if (improves_side_area(cand_eval, best_eval, plateau_eps)) {
                    best_eval = cand_eval;
                    best_sol = std::move(cand_q);
                }
            }
        }

        uint64_t sa_seed = opt.seed ^
                           (0x9E3779B97F4A7C15ULL +
                            static_cast<uint64_t>(entry.n) * 0xBF58476D1CE4E5B9ULL);
        SARefiner::HHState* hh_state =
            opt.sa_hh_auto ? &hh_states[hh_bucket_for_n(entry.n)] : nullptr;
        std::vector<TreePose> sa_out =
            run_targeted_sa(sa, best_sol, opt, entry.tier, entry.n, sa_seed, hh_state);
        if (!sa_out.empty()) {
            auto sa_q = quantize_poses(sa_out);
            if (!any_overlap(geom.base_poly, sa_q, geom.radius)) {
                SideArea sa_eval = eval_side_area(geom.base_poly, sa_q);
                if (improves_side_area(sa_eval, best_eval, plateau_eps)) {
                    best_eval = sa_eval;
                    best_sol = std::move(sa_q);
                }
            }
        }

        if (improves_side_area(best_eval, orig_eval, plateau_eps)) {
            ++improved_count;
        }
        sol = std::move(best_sol);
    }

    const int tier_c = static_cast<int>(targets.size()) - tier_a - tier_b;
    std::cout << "Targeted refine: " << targets.size() << " targets"
              << " (tierA=" << tier_a
              << ", tierB=" << tier_b
              << ", tierC=" << tier_c
              << ", cover=" << std::fixed << std::setprecision(3)
              << std::min(1.0, std::max(0.0, opt.target_cover))
              << ", improved=" << improved_count << ")\n";
    total_score = recompute_total_score(geom.base_poly, solutions_by_n, opt.n_max);
}

void write_solution(std::ofstream& out, int n, const std::vector<TreePose>& poses) {
    for (int i = 0; i < n; ++i) {
        const auto& pose = poses[i];
        out << std::setw(3) << std::setfill('0') << n << "_" << i << ","
            << fmt_submission_value(pose.x) << ","
            << fmt_submission_value(pose.y) << ","
            << fmt_submission_value(pose.deg) << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);

        Polygon base_poly = get_tree_polygon();
        const double radius = enclosing_circle_radius(base_poly);
        const double shift_a = wrap01(opt.shift_a);
        const double shift_b = wrap01(opt.shift_b);

        const double min_spacing = find_min_safe_hex_spacing(base_poly, radius);
        const double spacing = min_spacing * opt.spacing_safety;
        const GeometryContext geom{base_poly, radius};
        const LatticeContext lattice{geom, spacing, shift_a, shift_b, opt.angle_candidates};
        const GAContext ga_ctx{geom, opt};

        std::ofstream out(opt.output_path);
        if (!out) {
            throw std::runtime_error("Failed to open output file: " + opt.output_path);
        }

        out << "id,x,y,deg\n";

        SARefiner sa(base_poly, radius);
        std::array<SARefiner::HHState, 3> hh_states;
        std::vector<TreePose> ga_poses_sorted = maybe_run_ga(ga_ctx);

        std::vector<std::vector<TreePose>> solutions_by_n;
        solutions_by_n.resize(static_cast<size_t>(opt.n_max + 1));

        for (int n = 1; n <= opt.n_max; ++n) {
            BestSolution best = best_lattice_solution(lattice, n);

            consider_ga_prefix(geom, ga_poses_sorted, n, best);

            if (!std::isfinite(best.score)) {
                throw std::runtime_error(
                    "Could not generate valid poses for n=" +
                    std::to_string(n) + ".");
            }

            if (!opt.target_refine) {
                SARefiner::HHState* hh_state =
                    opt.sa_hh_auto ? &hh_states[hh_bucket_for_n(n)] : nullptr;
                SAContext sa_ctx{geom, sa, opt, n, hh_state};
                apply_sa_refinement(sa_ctx, best);
            }

            solutions_by_n[static_cast<size_t>(n)] = best.poses;
        }

        double total_score = 0.0;
        if (opt.target_refine) {
            run_targeted_refine(geom, solutions_by_n, opt, total_score);
        }

        if (opt.final_rigid) {
            total_score = 0.0;
            RigidContext rigid_ctx{geom, opt};
            for (int n = 1; n <= opt.n_max; ++n) {
                auto& sol = solutions_by_n[static_cast<size_t>(n)];
                if (static_cast<int>(sol.size()) != n) {
                    throw std::runtime_error("Invalid poses for n=" + std::to_string(n));
                }
                BestSolution best;
                best.poses = sol;
                best.score = score_instance(geom.base_poly, sol);
                apply_final_rigid(rigid_ctx, best);
                sol = std::move(best.poses);
                total_score += best.score;
            }
        } else {
            total_score = recompute_total_score(geom.base_poly, solutions_by_n, opt.n_max);
        }

        for (int n = 1; n <= opt.n_max; ++n) {
            const auto& sol = solutions_by_n[static_cast<size_t>(n)];
            if (static_cast<int>(sol.size()) != n) {
                throw std::runtime_error("Invalid poses for n=" + std::to_string(n));
            }
            write_solution(out, n, sol);
        }

        std::cout << "Submission written to " << opt.output_path << "\n";
        std::cout << "Local score: " << std::fixed << std::setprecision(9)
                  << total_score << "\n";
        std::cout << "Hex spacing: " << std::fixed << std::setprecision(9)
                  << spacing << "\n";
        std::cout << "Angles: " << opt.angle_candidates.size() << "\n";
        std::cout << "SA restarts: " << opt.sa_restarts << "\n";
        std::cout << "SA base iters: " << opt.sa_base_iters << "\n";
        std::cout << "SA iters per n: " << opt.sa_iters_per_n << "\n";
        std::cout << "SA HH mode: " << (opt.sa_hh_auto ? "auto" : "off") << "\n";
        std::cout << "SA aggressive: " << (opt.sa_aggressive ? "on" : "off") << "\n";
        std::cout << "Final rigid: " << (opt.final_rigid ? "on" : "off") << "\n";
        std::cout << "Target refine: " << (opt.target_refine ? "on" : "off") << "\n";

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
