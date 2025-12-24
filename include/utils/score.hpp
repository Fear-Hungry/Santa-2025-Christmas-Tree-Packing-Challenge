#pragma once

#include <vector>

#include "geometry/geom.hpp"

// Returns s^2 / n, where s is the bounding square side.
double score_from_side(double side, int n);

// Computes the local score for a single instance using the current poses.
double score_instance(const Polygon& base_poly, const std::vector<TreePose>& poses);
