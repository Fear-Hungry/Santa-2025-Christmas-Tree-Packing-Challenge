#include "baseline.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "collision.hpp"
#include "geom.hpp"

std::vector<TreePose> generate_grid_poses_for_n(int n,
                                                const Polygon& base_poly,
                                                const BaselineConfig& cfg) {
    if (n <= 0) {
        return {};
    }

    double radius = enclosing_circle_radius(base_poly);
    double step = cfg.spacing_factor * 2.0 * radius;

    int side = static_cast<int>(std::sqrt(static_cast<double>(n))) + 2;

    std::vector<TreePose> candidates;
    for (int ix = -side; ix <= side; ++ix) {
        for (int iy = -side; iy <= side; ++iy) {
            double x = cfg.center_x + ix * step;
            double y = cfg.center_y + iy * step;
            candidates.push_back({x, y, 0.0});
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [&](const TreePose& a, const TreePose& b) {
                  double da =
                      (a.x - cfg.center_x) * (a.x - cfg.center_x) +
                      (a.y - cfg.center_y) * (a.y - cfg.center_y);
                  double db =
                      (b.x - cfg.center_x) * (b.x - cfg.center_x) +
                      (b.y - cfg.center_y) * (b.y - cfg.center_y);
                  return da < db;
              });

    std::vector<TreePose> poses;
    poses.reserve(n);
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

    if (static_cast<int>(poses.size()) != n) {
        throw std::runtime_error(
            "Não foi possível gerar poses válidas suficientes.");
    }

    return poses;
}

