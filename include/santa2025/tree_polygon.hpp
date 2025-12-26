#pragma once

#include <array>

#include "santa2025/geometry.hpp"

namespace santa2025 {

inline const std::array<Point, 15>& tree_polygon_points() {
    static const std::array<Point, 15> kPoly = {{
        {0.0, 0.8},         // Tip
        {0.125, 0.5},       // Top tier (outer)
        {0.0625, 0.5},      // Top tier (inner)
        {0.2, 0.25},        // Mid tier (outer)
        {0.1, 0.25},        // Mid tier (inner)
        {0.35, 0.0},        // Bottom tier (outer)
        {0.075, 0.0},       // Trunk (top-right)
        {0.075, -0.2},      // Trunk (bottom-right)
        {-0.075, -0.2},     // Trunk (bottom-left)
        {-0.075, 0.0},      // Trunk (top-left)
        {-0.35, 0.0},       // Bottom tier (outer)
        {-0.1, 0.25},       // Mid tier (inner)
        {-0.2, 0.25},       // Mid tier (outer)
        {-0.0625, 0.5},     // Top tier (inner)
        {-0.125, 0.5},      // Top tier (outer)
    }};
    return kPoly;
}

inline Polygon tree_polygon() {
    const auto& pts = tree_polygon_points();
    return Polygon(pts.begin(), pts.end());
}

}  // namespace santa2025

