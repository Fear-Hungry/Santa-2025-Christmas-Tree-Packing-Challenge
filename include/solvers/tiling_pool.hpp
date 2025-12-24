#pragma once

#include <vector>

#include "geometry/geom.hpp"
#include "solvers/solver_tile_options.hpp"

struct MotifPoint {
    double a;    // coord em u (fração do tile)
    double b;    // coord em v (fração do tile)
    double deg;  // rotação relativa (graus)
};

struct Pattern {
    std::vector<MotifPoint> motif;
    double shift_a = 0.0;
    double shift_b = 0.0;
    double lattice_v_ratio = 1.0;    // |v| = lattice_v_ratio * |u|
    double lattice_theta_deg = 60.0; // ângulo entre u e v (graus)
};

std::vector<TreePose> generate_ordered_tiling(int n,
                                              double spacing,
                                              double angle_deg,
                                              const Pattern& pattern);

std::vector<TreePose> generate_windowed_tiling(int n,
                                               double spacing,
                                               double angle_deg,
                                               const Pattern& pattern,
                                               const Polygon& base_poly,
                                               int window_radius,
                                               int eval_n);

double find_min_safe_spacing(const Polygon& base_poly,
                             const Pattern& pattern,
                             double radius,
                             double eps,
                             double spacing_hint);

Pattern make_initial_pattern(int k);

Pattern optimize_tile_by_spacing(const Polygon& base_poly,
                                 Pattern pattern,
                                 double radius,
                                 const Options& opt);
