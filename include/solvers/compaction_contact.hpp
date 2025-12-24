#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "geometry/geom.hpp"

namespace compaction_contact {

struct EarlyStop {
    bool enabled = false;
    int min_passes = 0;
    int patience_passes = 0;
};

struct Params {
    int passes = 12;
    int attempts_per_pass = 32;
    int patience = 4;
    int boundary_topk = 14;
    int push_bisect_iters = 12;
    double push_max_step_frac = 0.9;
    double plateau_eps = 0.0;
    bool alt_axis = true;
    double diag_frac = 0.0;
    double diag_rand = 0.0;
    double center_bias = 0.0;
    double interior_prob = 0.0;
    double shake_pos = 0.0;
    double shake_rot_deg = 0.0;
    double shake_prob = 1.0;
    bool final_rigid = true;
    int quantize_decimals = 9;
    EarlyStop early_stop;
};

struct Stats {
    double side_before = 0.0;
    double side_after = 0.0;
    double area_before = 0.0;
    double area_after = 0.0;
    int moves = 0;
    int passes_run = 0;
    bool ok = false;
};

Stats compact_contact(const Polygon& base_poly,
                      std::vector<TreePose>& poses,
                      const Params& params,
                      std::mt19937_64& rng);

}  // namespace compaction_contact
