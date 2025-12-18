#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#include "geom.hpp"

struct GAParams {
    int pop_size = 40;
    int generations = 60;
    int elite = 2;
    int tournament_k = 3;

    double p_crossover = 0.90;
    double p_mut_swap = 0.25;
    double p_mut_scramble = 0.05;
    double p_mut_rot = 0.10;
    double p_mut_angle = 0.10;
    double p_mut_shift = 0.20;
    double p_mut_spacing = 0.15;

    double shift_sigma = 0.10;
    double spacing_safety_min = 1.000;
    double spacing_safety_max = 1.010;

    int candidate_factor = 50;  // ~ candidate_factor * n pontos candidatos.
    int candidate_min = 2000;

    std::vector<double> lattice_angle_candidates;
    std::vector<double> rotation_candidates = {0.0, 180.0};
};

struct GAResult {
    std::vector<TreePose> best_poses;
    double best_side = std::numeric_limits<double>::infinity();
    double best_spacing = 0.0;
    double best_angle_deg = 0.0;
    double best_shift_a = 0.0;
    double best_shift_b = 0.0;
};

class GlobalSearchGA {
public:
    GlobalSearchGA(const Polygon& base_poly, double radius);

    GAResult solve(int n, uint64_t seed, const GAParams& p) const;

private:
    Polygon base_poly_;
    double radius_ = 0.0;
};

