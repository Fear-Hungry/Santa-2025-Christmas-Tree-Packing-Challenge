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
    double boundary_prob = 0.7;  // chance to remove a boundary item (helps s reduction)
    LNSDestroyMode destroy_mode = LNSDestroyMode::kMixRandomBoundary;

    // Candidate generation / repair.
    int slide_iters = 60;
    double gap = 1e-6;

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
