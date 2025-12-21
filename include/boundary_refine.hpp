#pragma once

#include <cstdint>
#include <vector>

#include "geom.hpp"

struct BoundaryRefineParams {
    double radius = 0.0;
    int iters = 0;
    uint64_t seed = 0;
    double step_hint = 0.0;
};

void refine_boundary(const Polygon& base_poly,
                     std::vector<TreePose>& poses,
                     const BoundaryRefineParams& params);
