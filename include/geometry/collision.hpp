#pragma once

#include <vector>

#include "geometry/geom.hpp"

// Checa interseção (inclusive toque) entre dois polígonos.
bool polygons_intersect(const Polygon& p1, const Polygon& p2);

// Checa se há qualquer overlap entre árvores para o polígono base.
bool any_overlap(const Polygon& base_poly,
                 const std::vector<TreePose>& poses,
                 double radius = -1.0,
                 double eps = 1e-9);
