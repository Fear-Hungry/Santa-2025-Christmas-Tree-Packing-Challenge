#pragma once

#include <cstdint>
#include <string>
#include <vector>

enum class TileObjective {
    kDensity,
    kScore,
};

enum class ShiftSearchMode {
    kOff,
    kMultires,
};

struct Options {
    int n_max = 200;
    int k = 1;
    int pool_size = 600;
    std::string prefix_order = "central";  // central | greedy
    int tile_iters = 0;
    TileObjective tile_obj = TileObjective::kDensity;
    int tile_score_pool_size = 320;
    int tile_score_nmax = 200;
    int tile_score_full_every = 50;
    bool tile_opt_lattice = true;
    double lattice_v_ratio = 1.0;
    double lattice_theta_deg = 60.0;
    int refine_iters = 0;
    bool prune = true;
    int sa_restarts = 0;
    int sa_base_iters = 0;
    int sa_iters_per_n = 0;
    bool sa_chain = false;
    int sa_chain_base_iters = 40;
    int sa_chain_iters_per_n = 0;
    int sa_chain_min_n = 1;
    double sa_chain_band_layers = 2.5;
    bool sa_beam = false;
    int sa_beam_width = 0;
    int sa_beam_remove = 8;
    int sa_beam_micro_iters = 40;
    int sa_beam_init_iters = 0;
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
    double sa_overlap_weight = 0.0;
    double sa_overlap_eps_area = 1e-12;
    double sa_overlap_cost_cap = 0.0;
    double sa_plateau_eps = 0.0;
    double sa_w_resolve_overlap = 0.0;
    double sa_w_push_contact = 0.0;
    double sa_w_squeeze = 0.0;
    int sa_resolve_attempts = 6;
    double sa_resolve_step_frac_max = 0.20;
    double sa_resolve_step_frac_min = 0.02;
    double sa_resolve_noise_frac = 0.05;
    double sa_push_max_step_frac = 0.60;
    int sa_push_bisect_iters = 10;
    int sa_squeeze_pushes = 6;
    bool final_rigid = true;
    uint64_t seed = 123456789ULL;
    double spacing_safety = 1.001;
    double shift_a = 0.0;
    double shift_b = 0.0;
    bool pool_window_scan = false;
    int pool_window_radius = 4;
    ShiftSearchMode shift_search = ShiftSearchMode::kOff;
    int shift_grid = 8;
    int shift_levels = 4;
    int shift_keep = 6;
    int shift_pool_size = 0;
    std::vector<double> angle_candidates;
    std::string output_path = "submission_tile_cpp.csv";
};
