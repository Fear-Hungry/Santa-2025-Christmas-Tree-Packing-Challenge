#include "baseline.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "collision.hpp"
#include "geom.hpp"

namespace {
double squared_dist(double x, double y, const BaselineConfig& cfg) {
    const double dx = x - cfg.center_x;
    const double dy = y - cfg.center_y;
    return dx * dx + dy * dy;
}

std::vector<TreePose> build_grid_candidates(int side,
                                            double step,
                                            const BaselineConfig& cfg) {
    std::vector<TreePose> candidates;
    candidates.reserve(static_cast<size_t>((2 * side + 1) * (2 * side + 1)));
    for (int ix = -side; ix <= side; ++ix) {
        for (int iy = -side; iy <= side; ++iy) {
            double x = cfg.center_x + ix * step;
            double y = cfg.center_y + iy * step;
            candidates.push_back({x, y, 0.0});
        }
    }
    return candidates;
}

void sort_by_center_distance(std::vector<TreePose>& candidates,
                             const BaselineConfig& cfg) {
    std::sort(candidates.begin(),
              candidates.end(),
              [&](const TreePose& a, const TreePose& b) {
                  return squared_dist(a.x, a.y, cfg) <
                         squared_dist(b.x, b.y, cfg);
              });
}

std::vector<TreePose> select_non_overlapping(int n,
                                             const Polygon& base_poly,
                                             double radius,
                                             const std::vector<TreePose>& candidates) {
    std::vector<TreePose> poses;
    poses.reserve(static_cast<size_t>(n));
    for (const auto& pose : candidates) {
        if (static_cast<int>(poses.size()) >= n) {
            break;
        }
        std::vector<TreePose> test = poses;
        test.push_back(pose);
        if (!any_overlap(base_poly, test, radius)) {
            poses.push_back(pose);
        }
    }
    return poses;
}
}  // namespace

std::vector<TreePose> generate_grid_poses_for_n(int n,
                                                const Polygon& base_poly,
                                                const BaselineConfig& cfg) {
    if (n <= 0) {
        return {};
    }

    const double radius = enclosing_circle_radius(base_poly);
    const double step = cfg.spacing_factor * 2.0 * radius;

    const int side = static_cast<int>(std::sqrt(static_cast<double>(n))) + 2;

    std::vector<TreePose> candidates = build_grid_candidates(side, step, cfg);
    sort_by_center_distance(candidates, cfg);
    std::vector<TreePose> poses =
        select_non_overlapping(n, base_poly, radius, candidates);

    if (static_cast<int>(poses.size()) != n) {
        throw std::runtime_error(
            "Não foi possível gerar poses válidas suficientes.");
    }

    return poses;
}
