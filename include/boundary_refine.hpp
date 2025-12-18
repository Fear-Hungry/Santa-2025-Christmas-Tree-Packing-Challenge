#pragma once

#include <cstdint>
#include <vector>

#include "geom.hpp"

void refine_boundary(const Polygon& base_poly,
                     double radius,
                     std::vector<TreePose>& poses,
                     int iters,
                     uint64_t seed,
                     double step_hint);

