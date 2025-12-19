#pragma once

#include <limits>
#include <vector>

#include "geom.hpp"
#include "solver_tile_options.hpp"

struct Extents {
    double min_x;
    double max_x;
    double min_y;
    double max_y;
};

struct PruneResult {
    std::vector<std::vector<TreePose>> solutions_by_n;  // index = n
    std::vector<double> side_by_n;  // index = n (lado do bounding square)
    double total_score = 0.0;
};

struct ChainResult {
    std::vector<std::vector<TreePose>> solutions_by_n;  // index = n
    std::vector<double> side_by_n;  // index = n
    double total_score = 0.0;
};

struct ILSResult {
    bool improved = false;
    double start_side = std::numeric_limits<double>::infinity();
    double best_side = std::numeric_limits<double>::infinity();
    int attempts = 0;
    int accepted = 0;
};

std::vector<BoundingBox> bounding_boxes_for_poses(const Polygon& base_poly,
                                                  const std::vector<TreePose>& poses);

std::vector<double> prefix_sides_from_bbs(const std::vector<BoundingBox>& bbs);

std::vector<double> greedy_pruned_sides(const std::vector<BoundingBox>& bbs_in,
                                        int n_max,
                                        double tol);

double total_score_from_sides(const std::vector<double>& side_by_n, int n_max);

std::vector<TreePose> greedy_prefix_min_side(const Polygon& base_poly,
                                             const std::vector<TreePose>& pool,
                                             int n_max);

PruneResult build_greedy_pruned_solutions(const Polygon& base_poly,
                                         const std::vector<TreePose>& poses_pool,
                                         int n_max,
                                         double tol = 1e-12);

ChainResult build_sa_chain_solutions(const Polygon& base_poly,
                                     double radius,
                                     const std::vector<TreePose>& start_nmax,
                                     int n_max,
                                     double band_step,
                                     const Options& opt);

ChainResult build_sa_beam_chain_solutions(const Polygon& base_poly,
                                          double radius,
                                          const std::vector<TreePose>& start_nmax,
                                          int n_max,
                                          double band_step,
                                          const Options& opt);

ILSResult ils_basin_hop_compact(const Polygon& base_poly,
                                double radius,
                                std::vector<TreePose>& poses,
                                uint64_t seed,
                                const Options& opt);

// Milenkovic-Zeng (soft overlap) + ILS estilo paper (shift + tabu/ILS), aplicado
// somente em `n_max` (a própria lista de poses de tamanho n_max).
ILSResult mz_its_soft_compact(const Polygon& base_poly,
                              double radius,
                              std::vector<TreePose>& poses,
                              uint64_t seed,
                              const Options& opt);
