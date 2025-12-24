#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "geometry/geom.hpp"

namespace repair {

struct Extents {
    double min_x;
    double max_x;
    double min_y;
    double max_y;
};

Extents compute_extents(const std::vector<BoundingBox>& bbs);
Extents compute_extents_excluding(const std::vector<BoundingBox>& bbs, int skip);
double side_from_extents(const Extents& e);
Extents merge_extents_bb(const Extents& e, const BoundingBox& bb);
bool aabb_overlap(const BoundingBox& a, const BoundingBox& b);

std::vector<BoundingBox> bounding_boxes_for_poses(const Polygon& base_poly,
                                                  const std::vector<TreePose>& poses);
std::vector<int> build_extreme_pool(const std::vector<BoundingBox>& bbs, int topk);

struct InterlockOptions {
    int passes = 0;
    int attempts = 24;
    int group = 2;
    int rot_steps = 2;
    double rot_deg = 15.0;
    double max_step_frac = 0.50;
    int bisect_iters = 10;
    int boundary_topk = 20;
};

struct PocketOptions {
    int take = 6;
    int grid = 16;
    int attempts = 120;
    double radius_frac = 0.20;
    double rot_deg = 20.0;
    int output_decimals = 9;
};

bool apply_interlock_passes(const Polygon& base_poly,
                            std::vector<TreePose>& poses,
                            const InterlockOptions& opt,
                            uint64_t seed);

bool pocket_repack(const Polygon& base_poly,
                   double radius,
                   std::vector<TreePose>& poses,
                   const PocketOptions& opt,
                   std::mt19937_64& rng);

bool micro_refine_rigid_rotation(const Polygon& base_poly,
                                 double radius,
                                 std::vector<TreePose>& poses,
                                 int micro_rigid_steps,
                                 double micro_rigid_step_deg,
                                 int output_decimals);

}  // namespace repair
