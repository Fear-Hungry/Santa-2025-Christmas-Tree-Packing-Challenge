#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "santa2025/geometry.hpp"
#include "santa2025/nfp.hpp"
#include "santa2025/orientation.hpp"

namespace santa2025 {

enum class SAObjective {
    kS200 = 0,
    kPrefixScore = 1,
    kTargetSide = 2,
};

enum class SADeltaMode {
    kLinear = 0,
    kSquared = 1,
    kSquaredOverS = 2,
    kSquaredOverS2 = 3,
};

enum class SASecondary {
    kNone = 0,
    kPerimeter = 1,
    kArea = 2,
    kAspect = 3,
};

enum class SASchedule {
    kGeometric = 0,
    kPolynomial = 1,
};

struct SAOptions {
    int n = 200;
    std::vector<double> angles_deg{45.0};

    // Optional per-item orientation cycle (e.g., 0,180 to alternate upright/upside-down).
    std::vector<double> cycle_deg{};
    OrientationMode orientation_mode = OrientationMode::kTryAll;
    int cycle_prefix = 0;  // if >0, apply cycle only to i < cycle_prefix

    SAObjective objective = SAObjective::kS200;
    int nmax_score = 200;  // used when objective == kPrefixScore

    // Used when objective == kTargetSide.
    double target_side = 0.0;
    int target_power = 2;  // 1 => |s-target|_+, 2 => |s-target|_+^2

    // Acceptance Δ computation (used when objective == kS200).
    // Default uses (s_new^2 - s_old^2) / s_old, which behaves like ~2*(s_new - s_old) for small changes.
    SADeltaMode delta_mode = SADeltaMode::kSquaredOverS;
    double delta_scale = 1.0;

    // Secondary tie-breaker applied only when s_new ~= s_old (helps when s is piecewise-constant).
    SASecondary secondary = SASecondary::kNone;
    double secondary_weight = 0.0;

    std::uint64_t seed = 1;
    int iters = 200'000;
    int tries_per_iter = 8;

    // Temperature schedule: geometric from t0 -> t1.
    double t0 = 0.25;
    double t1 = 1e-4;

    // Schedule type and parameters.
    SASchedule schedule = SASchedule::kGeometric;
    double poly_power = 1.0;  // used when schedule == kPolynomial

    // Optional adaptive temperature scaling to target an acceptance rate (on feasible moves).
    // Set adaptive_window > 0 to enable.
    int adaptive_window = 0;
    double target_accept = 0.35;
    double adapt_up = 1.25;
    double adapt_down = 0.9;
    double adapt_min_scale = 0.05;
    double adapt_max_scale = 20.0;

    // Move parameters.
    double boundary_prob = 0.5;  // probability to pick a boundary item (helps focus on s reduction)
    double cluster_prob = 0.08;  // probability to do a cluster translation move
    int cluster_min = 2;
    int cluster_max = 8;
    double cluster_radius_mult = 4.0;  // radius = mult * tree_radius
    double cluster_sigma_mult = 2.5;   // translation sigma multiplier for cluster moves

    double touch_prob = 0.25;  // probability to propose a "touch" relocation (NFP vertex around random anchor)
    int touch_best_of = 1;     // if >1, sample k offsets and keep the best by objective
    double rot_prob = 0.15;    // for translation moves, probability to also change the angle

    double trans_sigma0 = 0.20;  // translation stddev at start (world units)
    double trans_sigma1 = 0.01;  // translation stddev at end

    double rot_jitter_deg = 0.0;  // optional extra small jitter added after snapping (kept 0 by default)

    // NFP "gap" when placing just outside a forbidden boundary.
    double gap = 1e-6;

    // Safety margin used in collision checks (treat near-contacts as collision).
    double safety_eps = 0.0;

    // If > 0, limit the number of NFP vertex offsets sampled per delta.
    int max_offsets_per_delta = 512;

    // If > 0, print progress every k iterations to stderr.
    int log_every = 0;
    std::string log_prefix = "[sa]";
};

struct SAResult {
    std::vector<Pose> best_poses;

    double init_cost = 0.0;
    double best_cost = 0.0;

    double init_s200 = 0.0;
    double best_s200 = 0.0;

    double init_prefix_score = 0.0;
    double best_prefix_score = 0.0;

    int attempted = 0;
    int feasible = 0;
    int accepted = 0;
};

// Runs simulated annealing starting from `initial` (size n).
SAResult simulated_annealing(
    const Polygon& tree_poly,
    const std::vector<Pose>& initial,
    const SAOptions& opt,
    double eps = 1e-12
);

// Utility objective components (using per-tree axis-aligned bboxes of rotated trees).
double packing_s200(const Polygon& tree_poly, const std::vector<Pose>& poses);
double packing_prefix_score(const Polygon& tree_poly, const std::vector<Pose>& poses, int nmax = 200);

}  // namespace santa2025
