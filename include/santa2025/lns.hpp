#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "santa2025/nfp.hpp"
#include "santa2025/orientation.hpp"

namespace santa2025 {

enum class LNSDestroyMode {
    kMixRandomBoundary = 0,
    kRandom = 1,
    kBoundary = 2,
    kCluster = 3,
    kGap = 4,
};

struct LNSOptions {
    int n = 200;
    std::vector<double> angles_deg{45.0};

    // Optional per-item orientation cycle (e.g., 0,180).
    std::vector<double> cycle_deg{};
    OrientationMode orientation_mode = OrientationMode::kTryAll;
    int cycle_prefix = 0;

    // Staged shrink: try to fit into progressively smaller squares.
    int stages = 50;
    int stage_attempts = 50;
    double shrink_factor = 0.999;
    double shrink_delta = 0.0;

    // Destroy & repair parameters (per attempt).
    double remove_frac = 0.10;   // fraction of items removed each attempt
    double remove_frac_max = 0.10;  // if > remove_frac, can grow within a stage (see remove_frac_growth)
    double remove_frac_growth = 1.0;  // per-attempt multiplier when adaptive removal is enabled (>1)
    int remove_frac_growth_every = 1;  // apply growth every k attempts (>=1)
    double boundary_prob = 0.7;  // chance to remove a boundary item (helps s reduction)
    LNSDestroyMode destroy_mode = LNSDestroyMode::kMixRandomBoundary;

    // Gap-guided destroy (used when destroy_mode == kGap).
    // Finds a large empty region via a coarse occupancy grid and removes trees around it.
    int gap_grid = 48;
    bool gap_try_hole_center = true;

    // Candidate generation / repair.
    int slide_iters = 60;
    double gap = 1e-6;

    // Optional post-repair local search (SA) applied only to the removed subset (others fixed).
    int repair_sa_iters = 0;           // 0 disables
    int repair_sa_best_of = 2;         // sample k proposals per iter and keep the best
    double repair_sa_t0 = 0.02;        // start temperature (on s200 delta)
    double repair_sa_t1 = 1e-4;        // end temperature
    int repair_sa_anchor_samples = 2;  // anchors sampled per proposal

    // Safety margin used in collision checks (treat near-contacts as collision).
    double safety_eps = 0.0;
    int max_offsets_per_delta = 512;

    std::uint64_t seed = 1;
    int log_every = 0;
    std::string log_prefix = "[lns]";
};

struct LNSResult {
    std::vector<Pose> best_poses;

    double init_s200 = 0.0;
    double best_s200 = 0.0;

    int stages_done = 0;
    double last_target = 0.0;
    int attempted = 0;
    int succeeded = 0;
};

// Large Neighborhood Search: destroy a subset, reinsert using bottom-left-style repair.
// Returns best packing found (always non-overlapping) and a small summary.
LNSResult lns_shrink_wrap(
    const Polygon& tree_poly,
    const std::vector<Pose>& initial,
    const LNSOptions& opt,
    double eps = 1e-12
);

}  // namespace santa2025
