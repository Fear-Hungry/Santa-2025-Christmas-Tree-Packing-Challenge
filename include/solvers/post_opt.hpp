#pragma once

#include <cstdint>
#include <vector>

#include "geometry/geom.hpp"

struct PostOptTermEpochStats {
    int epoch = 0;
    int tier_a_count = 0;
    int tier_b_count = 0;
    double tier_a_term_before = 0.0;
    double tier_a_term_after = 0.0;
    double tier_b_term_before = 0.0;
    double tier_b_term_after = 0.0;
    std::vector<int> tier_a_ns;
};

struct PostOptTermSummary {
    int tier_a_count = 0;
    int tier_b_count = 0;
    double tier_a_term_before = 0.0;
    double tier_a_term_after = 0.0;
    double tier_b_term_before = 0.0;
    double tier_b_term_after = 0.0;
    std::vector<int> tier_a_ns;
};

struct PostOptOptions {
    bool enabled = false;
    int iters = 15000;
    int restarts = 16;
    double t0 = 2.5;
    double tm = 0.0000005;
    bool enable_squeeze = true;
    bool enable_compaction = true;
    bool enable_edge_slide = true;
    bool enable_local_search = true;
    double remove_ratio = 0.50;
    int free_area_min_n = 10;
    bool enable_free_area = true;
    bool enable_backprop = true;
    int backprop_passes = 10;
    int backprop_span = 5;
    int backprop_max_combos = 200;

    // Term-based scheduler: allocate budget to high-impact instances.
    bool enable_term_scheduler = false;
    int term_epochs = 3;
    int term_tier_a = 20;
    int term_tier_b = 120;
    int term_min_n = 1;
    double tier_a_iters_mult = 1.50;
    double tier_a_restarts_mult = 1.00;
    double tier_b_iters_mult = 0.50;
    double tier_b_restarts_mult = 0.25;
    double tier_c_iters_mult = 0.00;
    double tier_c_restarts_mult = 0.00;
    double tier_a_tighten_mult = 1.00;
    double tier_b_tighten_mult = 0.40;
    double tier_c_tighten_mult = 0.10;
    double accept_term_eps = 0.0;

    // Reinsertion (remove/reinsert) options.
    bool enable_guided_reinsert = false;
    int reinsert_attempts_tier_a = 4000;
    int reinsert_attempts_tier_b = 1200;
    int reinsert_attempts_tier_c = 200;
    int reinsert_shell_anchors = 32;
    int reinsert_core_anchors = 64;
    int reinsert_jitter_attempts = 256;
    double reinsert_angle_jitter_deg = 30.0;
    double reinsert_early_stop_rel = 0.002;

    // Exploratory backprop: allow sources even if side_src >= side_k.
    bool enable_backprop_explore = false;
    int backprop_span_tier_a = 10;
    int backprop_span_tier_b = 7;
    int backprop_span_tier_c = 5;
    int backprop_max_combos_tier_a = 400;
    int backprop_max_combos_tier_b = 250;
    int backprop_max_combos_tier_c = 200;
    int threads = 0;
    uint64_t seed = 42ULL;
};

struct PostOptStats {
    double initial_score = 0.0;
    double final_score = 0.0;
    int phase1_improved = 0;
    int backprop_improved = 0;
    double elapsed_sec = 0.0;
    PostOptTermSummary term_summary;
    std::vector<PostOptTermEpochStats> term_epochs;
};

bool post_optimize_submission(const Polygon& base_poly,
                              std::vector<std::vector<TreePose>>& solutions_by_n,
                              const PostOptOptions& opt,
                              PostOptStats* stats_out);
