#pragma once

#include <vector>

#include "santa2025/collision_index.hpp"
#include "santa2025/geometry.hpp"
#include "santa2025/orientation.hpp"
#include "santa2025/packing_stats.hpp"

namespace santa2025 {

struct BottomLeftOptions {
    int n = 200;
    std::vector<double> angles_deg{45.0};

    // Optional per-item orientation cycle (e.g., 0,180 to alternate upright/upside-down).
    // When set, behavior is controlled by `orientation_mode`.
    std::vector<double> cycle_deg{};
    OrientationMode orientation_mode = OrientationMode::kTryAll;
    int cycle_prefix = 0;  // if >0, apply cycle only to i < cycle_prefix

    // If > 0, limit the number of NFP vertex offsets considered per delta (sorted by y,x).
    int max_offsets_per_delta = 512;

    // Extra offset applied outward from the NFP boundary to avoid "touching == collision".
    double gap = 1e-6;

    // Safety margin used in collision checks (treat near-contacts as collision).
    double safety_eps = 0.0;

    // Binary-search iterations for the drop/push phases.
    int slide_iters = 32;

    // Bottom-left is defined inside a square container [0, side] x [0, side].
    // If side <= 0, we start from an area-based guess and grow until feasible.
    double side = 0.0;
    double density_guess = 0.4;
    double side_grow = 1.05;
    int max_restarts = 40;
};

std::vector<Pose> bottom_left_pack(const Polygon& tree_poly, const BottomLeftOptions& opt, double eps = 1e-12);

}  // namespace santa2025
