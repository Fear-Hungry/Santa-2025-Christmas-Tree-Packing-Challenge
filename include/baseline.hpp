#pragma once

#include <vector>

#include "geom.hpp"

struct BaselineConfig {
    double spacing_factor = 2.5;
    double center_x = 0.0;
    double center_y = 0.0;
};

std::vector<TreePose> generate_grid_poses_for_n(int n,
                                                const Polygon& base_poly,
                                                const BaselineConfig& cfg);

