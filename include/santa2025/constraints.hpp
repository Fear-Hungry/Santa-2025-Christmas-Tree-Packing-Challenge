#pragma once

#include <algorithm>

#include "santa2025/nfp.hpp"

namespace santa2025 {

constexpr double kCoordMin = -100.0;
constexpr double kCoordMax = 100.0;

inline bool within_coord_bounds(
    double x,
    double y,
    double min = kCoordMin,
    double max = kCoordMax,
    double eps = 0.0
) {
    return (x >= min - eps) && (x <= max + eps) && (y >= min - eps) && (y <= max + eps);
}

inline bool within_coord_bounds(const Pose& p, double min = kCoordMin, double max = kCoordMax, double eps = 0.0) {
    return within_coord_bounds(p.x, p.y, min, max, eps);
}

inline Pose clamp_pose_xy(Pose p, double min = kCoordMin, double max = kCoordMax) {
    p.x = std::clamp(p.x, min, max);
    p.y = std::clamp(p.y, min, max);
    return p;
}

}  // namespace santa2025

