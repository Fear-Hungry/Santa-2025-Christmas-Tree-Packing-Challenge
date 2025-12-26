#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "santa2025/geometry.hpp"
#include "santa2025/nfp.hpp"

namespace santa2025 {

struct OrderHHOptions {
    int n = 200;
    int nmax_score = 200;

    // HH budget = number of operator applications.
    int iters = 20'000;

    // Late Acceptance Hill Climbing (LAHC) history length.
    int lahc_length = 0;  // if 0, uses 10*n (clamped)

    // Multi-armed bandit selection (UCB1).
    double ucb_c = 0.25;

    std::uint64_t seed = 1;

    // If >0, print progress every k iterations (stderr).
    int log_every = 0;
    std::string log_prefix = "[order_hh]";
};

struct OrderHHOperatorStats {
    std::string name;
    int selected = 0;
    int accepted = 0;
    double mean_reward = 0.0;
};

struct OrderHHResult {
    std::vector<Pose> best_poses;

    double init_score = 0.0;
    double best_score = 0.0;

    int attempted = 0;
    int accepted = 0;

    std::vector<OrderHHOperatorStats> ops;
};

// Hyper-heuristic in the permutation (order) space for prefix-score minimization:
// - LLHs: order swap / segment reverse / segment shuffle / reinsertion
// - Operator selection: UCB1
// - Move acceptance: LAHC (late acceptance hill climbing)
OrderHHResult optimize_prefix_order_hh(
    const Polygon& tree_poly,
    const std::vector<Pose>& poses,
    const OrderHHOptions& opt,
    double eps = 1e-12
);

}  // namespace santa2025

