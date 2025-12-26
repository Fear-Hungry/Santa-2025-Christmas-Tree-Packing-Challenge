#pragma once

#include <vector>

#include "santa2025/geometry.hpp"
#include "santa2025/nfp.hpp"

namespace santa2025 {

struct PackingStats {
    BoundingBox bbox;
    double square_side = 0.0;
};

PackingStats packing_stats(const Polygon& tree_poly, const std::vector<Pose>& poses);

}  // namespace santa2025

