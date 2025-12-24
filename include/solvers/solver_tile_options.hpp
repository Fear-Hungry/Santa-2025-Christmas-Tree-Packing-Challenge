#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "solvers/sa.hpp"

enum class TileObjective {
    kDensity,
    kScore,
};

enum class ShiftSearchMode {
    kOff,
    kMultires,
};

enum class ChainDropMode {
    kGreedy,
    kBest,
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
    int tile_score_fast_angles = 1;
    int tile_density_warmup_iters = 0;
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
    ChainDropMode sa_chain_drop = ChainDropMode::kGreedy;
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
    double sa_w_push_contact = 0.0;
    double sa_w_slide_contact = 0.0;
    double sa_w_squeeze = 0.0;
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

    // ILS / basin-hopping (n_max): shake afim + repair + SA local.
    int ils_iters = 0;
    int ils_sa_restarts = 1;
    int ils_sa_iters = 0;
    bool ils_accept_sa = false;
    double ils_t0 = 0.05;
    double ils_t1 = 0.01;
    double ils_alpha_min = 0.97;
    double ils_alpha_max = 0.995;
    double ils_p_aniso = 0.70;
    double ils_shear_max = 0.0;
    double ils_jitter_frac = 0.02;
    double ils_subset_frac = 0.20;
    double ils_rot_prob = 0.20;
    double ils_rot_deg_max = 30.0;
    int ils_repair_mtv_passes = 300;
    double ils_repair_mtv_damping = 1.0;
    double ils_repair_mtv_split = 0.5;

    // Global contraction (n_max): escala constante + relaxamento por forças.
    int global_contract_steps = 0;
    double global_contract_scale = 0.999;
    int global_contract_relax_iters = 0;
    double global_contract_overlap_force = 1.0;
    double global_contract_center_force = 0.02;
    double global_contract_step_frac = 0.05;
    int global_contract_repair_passes = 200;
    int global_contract_sa_restarts = 0;
    int global_contract_sa_iters = 0;

    // Milenkovic-Zeng (soft overlap) + ILS (shift + tabu).
    // Aplica um SA em 2 fases (liquefação -> solidificação) com soft overlap,
    // e em volta executa um ILS estilo paper (shift perturb + tabu em swap-rot).
    int mz_its_iters = 0;
    int mz_perturb_depth = -1;      // <0 => 10*n
    int mz_tabu_depth = -1;         // <0 => 10*n
    int mz_tabu_samples = 24;       // amostras por iteração do tabu
    int mz_phase_a_iters = 0;
    int mz_phase_b_iters = 0;
    double mz_a_t0 = 0.50;
    double mz_a_t1 = 0.10;
    double mz_b_t0 = 0.15;
    double mz_b_t1 = 0.01;
    double mz_overlap_a = 1.0;
    double mz_overlap_b_start = 1.0;
    double mz_overlap_b_end = 1e5;
    bool mz_overlap_b_geometric = true;
    double mz_w_push_contact = 100.0;   // "squeeze" via push_to_contact (+ overshoot)
    double mz_push_overshoot_a = 1.0;   // fração de max_step na fase A
    double mz_push_overshoot_b = 0.05;  // fração de max_step na fase B
    double mz_w_resolve_overlap_b = 10.0;

    bool final_rigid = true;
    double micro_rot_eps = 0.0;
    int micro_rot_steps = 0;
    double micro_shift_eps = 0.0;
    int micro_shift_steps = 0;
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
    int target_rounds = 1;
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

    // Pós-otimizador estilo "Tree Packer v18".
    bool post_opt = false;
    int post_iters = 15000;
    int post_restarts = 16;
    double post_t0 = 2.5;
    double post_tm = 0.0000005;
    bool post_enable_squeeze = true;
    bool post_enable_compaction = true;
    bool post_enable_edge_slide = true;
    bool post_enable_local_search = true;
    double post_remove_ratio = 0.50;
    int post_free_area_min_n = 10;
    bool post_enable_free_area = true;

    // Scheduler por termo (s_n^2/n): tiers A/B/C e épocas.
    bool post_term_scheduler = false;
    int post_term_epochs = 3;
    int post_term_tier_a = 20;
    int post_term_tier_b = 120;
    int post_term_min_n = 1;
    double post_tier_a_iters_mult = 1.50;
    double post_tier_a_restarts_mult = 1.00;
    double post_tier_b_iters_mult = 0.50;
    double post_tier_b_restarts_mult = 0.25;
    double post_tier_c_iters_mult = 0.00;
    double post_tier_c_restarts_mult = 0.00;
    double post_tier_a_tighten_mult = 1.00;
    double post_tier_b_tighten_mult = 0.40;
    double post_tier_c_tighten_mult = 0.10;
    double post_accept_term_eps = 0.0;

    // Reinserção guiada (remove/reinsert) para Tier A/B.
    bool post_guided_reinsert = false;
    int post_reinsert_attempts_tier_a = 4000;
    int post_reinsert_attempts_tier_b = 1200;
    int post_reinsert_attempts_tier_c = 200;
    int post_reinsert_shell_anchors = 32;
    int post_reinsert_core_anchors = 64;
    int post_reinsert_jitter_attempts = 256;
    double post_reinsert_angle_jitter_deg = 30.0;
    double post_reinsert_early_stop_rel = 0.002;

    // Backprop exploratório (fontes além do gate side_src < side_k).
    bool post_backprop_explore = false;
    int post_backprop_span_a = 10;
    int post_backprop_span_b = 7;
    int post_backprop_span_c = 5;
    int post_backprop_max_combos_a = 400;
    int post_backprop_max_combos_b = 250;
    int post_backprop_max_combos_c = 200;
    bool post_enable_backprop = true;
    int post_backprop_passes = 10;
    int post_backprop_span = 5;
    int post_backprop_max_combos = 200;
    int post_threads = 0;
};
