#include "utils/micro_adjust.hpp"

#include <cmath>
#include <limits>
#include <utility>

#include "geometry/collision.hpp"
#include "utils/submission_io.hpp"
#include "utils/wrap_utils.hpp"

namespace {

constexpr double kSideEps = 1e-15;
constexpr double kBound = 100.0;

bool within_bounds(const std::vector<TreePose>& poses) {
    for (const auto& p : poses) {
        if (std::abs(p.x) > kBound || std::abs(p.y) > kBound) {
            return false;
        }
    }
    return true;
}

double side_for_solution(const Polygon& base_poly, const std::vector<TreePose>& poses) {
    auto polys = transformed_polygons(base_poly, poses);
    return bounding_square_side(polys);
}

std::vector<TreePose> apply_rigid_transform(const std::vector<TreePose>& poses,
                                            double rot_deg,
                                            double dx,
                                            double dy) {
    if (poses.empty()) {
        return poses;
    }
    std::vector<TreePose> out;
    out.reserve(poses.size());
    for (const auto& pose : poses) {
        Point p = rotate_point(Point{pose.x, pose.y}, rot_deg);
        out.push_back(TreePose{p.x + dx,
                               p.y + dy,
                               wrap_deg(pose.deg + rot_deg)});
    }
    return out;
}

struct MicroAdjustSearch {
    const Polygon& base_poly;
    double radius;
    const MicroAdjustOptions& opt;

    double best_side = std::numeric_limits<double>::infinity();
    double best_rot = 0.0;
    double best_dx = 0.0;
    double best_dy = 0.0;
    std::vector<TreePose> best_unquant;
    std::vector<TreePose> best_quant;

    MicroAdjustSearch(const Polygon& base_poly_in,
                      double radius_in,
                      const MicroAdjustOptions& opt_in,
                      const std::vector<TreePose>& base_unquant,
                      std::vector<TreePose> base_quant,
                      double base_side)
        : base_poly(base_poly_in),
          radius(radius_in),
          opt(opt_in),
          best_side(base_side),
          best_unquant(base_unquant),
          best_quant(std::move(base_quant)) {}

    void try_candidate(std::vector<TreePose> cand_unquant,
                       double rot_deg,
                       double dx,
                       double dy) {
        std::vector<TreePose> cand_quant = quantize_poses(cand_unquant, opt.decimals);
        if (!within_bounds(cand_quant)) {
            return;
        }
        if (any_overlap(base_poly, cand_quant, radius)) {
            return;
        }
        double side = side_for_solution(base_poly, cand_quant);
        if (!(side + kSideEps < best_side)) {
            return;
        }
        best_side = side;
        best_unquant = std::move(cand_unquant);
        best_quant = std::move(cand_quant);
        best_rot = rot_deg;
        best_dx = dx;
        best_dy = dy;
    }

    void search_rotation(const std::vector<TreePose>& base_poses) {
        if (!(opt.rot_steps > 0 && opt.rot_eps_deg > 0.0)) {
            return;
        }
        const double step = opt.rot_eps_deg / static_cast<double>(opt.rot_steps);
        for (int i = -opt.rot_steps; i <= opt.rot_steps; ++i) {
            if (i == 0) {
                continue;
            }
            const double ang = static_cast<double>(i) * step;
            try_candidate(apply_rigid_transform(base_poses, ang, 0.0, 0.0), ang, 0.0, 0.0);
        }
    }

    void search_shift() {
        if (!(opt.shift_steps > 0 && opt.shift_eps > 0.0)) {
            return;
        }

        const double step = opt.shift_eps / static_cast<double>(opt.shift_steps);
        const std::vector<TreePose> shift_base = best_unquant;
        const double rot = best_rot;

        for (int ix = -opt.shift_steps; ix <= opt.shift_steps; ++ix) {
            const double dx = static_cast<double>(ix) * step;
            for (int iy = -opt.shift_steps; iy <= opt.shift_steps; ++iy) {
                const double dy = static_cast<double>(iy) * step;
                if (dx == 0.0 && dy == 0.0) {
                    continue;
                }
                try_candidate(apply_rigid_transform(shift_base, 0.0, dx, dy), rot, dx, dy);
            }
        }
    }
};

}  // namespace

MicroAdjustOutcome apply_micro_adjustments(const Polygon& base_poly,
                                          const std::vector<TreePose>& poses,
                                          double radius,
                                          const MicroAdjustOptions& opt) {
    MicroAdjustOutcome out;

    std::vector<TreePose> base_quant = quantize_poses(poses, opt.decimals);
    out.result.base_side = side_for_solution(base_poly, base_quant);
    out.result.best_side = out.result.base_side;

    if (!within_bounds(base_quant) || any_overlap(base_poly, base_quant, radius)) {
        out.poses = std::move(base_quant);
        return out;
    }

    MicroAdjustSearch search(base_poly, radius, opt, poses, std::move(base_quant), out.result.base_side);
    search.search_rotation(poses);
    search.search_shift();

    out.result.best_side = search.best_side;
    out.result.rot_deg = search.best_rot;
    out.result.shift_x = search.best_dx;
    out.result.shift_y = search.best_dy;
    out.result.improved = (out.result.best_side + kSideEps < out.result.base_side);
    out.result.applied_rotation = (search.best_rot != 0.0);
    out.result.applied_shift = (search.best_dx != 0.0 || search.best_dy != 0.0);

    out.poses = std::move(search.best_quant);
    return out;
}
