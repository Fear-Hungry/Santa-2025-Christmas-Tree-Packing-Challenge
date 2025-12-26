#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "santa2025/lns.hpp"
#include "santa2025/nfp.hpp"
#include "santa2025/simulated_annealing.hpp"

namespace santa2025 {

enum class HHObjective {
    kS = 0,
    kPrefixScore = 1,
};

struct HHOptions {
    int n = 200;
    HHObjective objective = HHObjective::kS;
    int nmax_score = 200;  // used when objective == kPrefixScore

    // HH budget = number of operator applications.
    int hh_iters = 200;

    // Late Acceptance Hill Climbing (LAHC) history length.
    int lahc_length = 0;  // if 0, uses 10*n (clamped)

    // Multi-armed bandit selection (UCB1).
    double ucb_c = 0.25;

    // Base options used by SA/LNS operators.
    SAOptions sa_base;
    LNSOptions lns_base;

    // Deterministic seed for the HH outer loop.
    std::uint64_t seed = 1;

    // If >0, print progress every k HH iterations (stderr).
    int log_every = 0;
    std::string log_prefix = "[hh]";
};

struct HHOperatorStats {
    std::string name;
    int selected = 0;
    int feasible = 0;
    int accepted = 0;
    double mean_reward = 0.0;
};

struct HHResult {
    std::vector<Pose> best_poses;

    double init_s = 0.0;
    double best_s = 0.0;

    double init_score = 0.0;
    double best_score = 0.0;

    double init_obj = 0.0;
    double best_obj = 0.0;

    int attempted = 0;
    int feasible = 0;
    int accepted = 0;

    std::vector<HHOperatorStats> ops;
};

// Hyper-heuristic (selection + acceptance) on top of existing SA/LNS operators.
// - Operator selection: UCB1
// - Move acceptance: LAHC (late acceptance hill climbing)
HHResult hyper_heuristic_optimize(
    const Polygon& tree_poly,
    const std::vector<Pose>& initial,
    const HHOptions& opt,
    double eps = 1e-12
);

}  // namespace santa2025

