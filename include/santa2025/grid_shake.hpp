#pragma once

#include <vector>

#include "santa2025/geometry.hpp"
#include "santa2025/nfp.hpp"
#include "santa2025/orientation.hpp"

namespace santa2025 {

struct GridShakeOptions {
    int n = 200;
    std::vector<double> angles_deg{45.0};

    // Optional per-item orientation cycle (e.g., 0,180 to alternate upright/upside-down).
    // When set, behavior is controlled by `orientation_mode`.
    std::vector<double> cycle_deg{};
    OrientationMode orientation_mode = OrientationMode::kTryAll;
    int cycle_prefix = 0;  // if >0, apply cycle only to i < cycle_prefix

    // Grid spacing between centers. If <= 0, a safe default based on the tree radius is used.
    double step = 0.0;

    // A small safety margin used for the default step (and to avoid "touching == collision" artifacts).
    double gap = 1e-3;

    // Safety margin used in collision checks (treat near-contacts as collision).
    double safety_eps = 0.0;

    // Number of global compaction passes.
    int passes = 10;

    // Binary-search iterations for the drop/push phases.
    int slide_iters = 32;

    // Compaction is performed inside a square container [0, side] x [0, side].
    // If side <= 0, we start from a grid-based guess and grow until feasible.
    double side = 0.0;
    double side_grow = 1.05;
    int max_restarts = 20;
};

std::vector<Pose> grid_shake_pack(const Polygon& tree_poly, const GridShakeOptions& opt, double eps = 1e-12);

}  // namespace santa2025
