#pragma once

#include <vector>

#include "geom.hpp"

struct MicroAdjustOptions {
    double rot_eps_deg = 0.0;
    int rot_steps = 0;
    double shift_eps = 0.0;
    int shift_steps = 0;
    int decimals = 9;
};

struct MicroAdjustResult {
    double base_side = 0.0;
    double best_side = 0.0;
    double rot_deg = 0.0;
    double shift_x = 0.0;
    double shift_y = 0.0;
    bool improved = false;
    bool applied_rotation = false;
    bool applied_shift = false;
};

struct MicroAdjustOutcome {
    MicroAdjustResult result;
    std::vector<TreePose> poses;
};

MicroAdjustOutcome apply_micro_adjustments(const Polygon& base_poly,
                                          const std::vector<TreePose>& poses,
                                          double radius,
                                          const MicroAdjustOptions& opt);
