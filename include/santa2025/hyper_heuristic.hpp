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

    // Adaptive Operator Selection (ALNS-style roulette) inside each macro group.
    // weight <- (1-rho)*weight + rho*reward, clamped to >= alns_min_weight.
    double alns_rho = 0.20;
    double alns_min_weight = 1e-12;

    // Reward shaping: reward = max(0, old_obj - new_obj) / (elapsed_ms + reward_time_eps_ms).
    double reward_time_eps_ms = 1e-3;

    // SA bursts (only used when HH detects stagnation).
    // If 0, uses an auto threshold based on hh_iters.
    int sa_burst_after = 0;
    // If 0, uses sa_burst_after.
    int sa_burst_cooldown = 0;
    // If >0, overrides SA iters when running bursts.
    int sa_burst_iters = 5000;

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
    double weight = 1.0;
};

struct HHMacroStats {
    std::string name;
    int selected = 0;
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

    std::vector<HHMacroStats> macros;
    std::vector<HHOperatorStats> ops;
};

// Hyper-heuristic (selection + acceptance) on top of existing SA/LNS operators.
// - Macro selection: UCB1 (small/medium/large)
// - Operator selection (within macro): ALNS roulette (adaptive weights)
// - Move acceptance: LAHC (late acceptance hill climbing)
// - Credit assignment: reward = max(0, old_obj - new_obj) / (elapsed_ms + eps)
HHResult hyper_heuristic_optimize(
    const Polygon& tree_poly,
    const std::vector<Pose>& initial,
    const HHOptions& opt,
    double eps = 1e-12
);

}  // namespace santa2025
