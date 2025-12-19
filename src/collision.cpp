#include "collision.hpp"

#include "spatial_grid.hpp"

namespace {

struct OverlapParams {
    double limit;
    double limit_sq;
    double eps;
};

struct OverlapInput {
    const std::vector<TreePose>& poses;
    const std::vector<Polygon>& polys;
    const std::vector<BoundingBox>& bbs;
};

struct OverlapContext {
    const OverlapInput& input;
    const OverlapParams& params;
};

struct PairIndex {
    int i;
    int j;
};

double resolve_radius(const Polygon& base_poly, double radius) {
    if (radius < 0.0) {
        return enclosing_circle_radius(base_poly);
    }
    return radius;
}

std::vector<BoundingBox> compute_bounding_boxes(const std::vector<Polygon>& polys) {
    std::vector<BoundingBox> bbs;
    bbs.reserve(polys.size());
    for (const auto& poly : polys) {
        bbs.push_back(bounding_box(poly));
    }
    return bbs;
}

bool aabb_overlap(const BoundingBox& a, const BoundingBox& b, const OverlapParams& params) {
    if (a.max_x < b.min_x - params.eps || b.max_x < a.min_x - params.eps) {
        return false;
    }
    if (a.max_y < b.min_y - params.eps || b.max_y < a.min_y - params.eps) {
        return false;
    }
    return true;
}

void fill_grid(UniformGridIndex& grid, const std::vector<TreePose>& poses) {
    const int n = static_cast<int>(poses.size());
    for (int i = 0; i < n; ++i) {
        grid.insert(i, poses[static_cast<size_t>(i)].x, poses[static_cast<size_t>(i)].y);
    }
}

bool pair_overlaps(const PairIndex& pair, const OverlapContext& context) {
    const auto& pi = context.input.poses[static_cast<size_t>(pair.i)];
    const auto& pj = context.input.poses[static_cast<size_t>(pair.j)];
    const double dx = pi.x - pj.x;
    const double dy = pi.y - pj.y;
    if (dx * dx + dy * dy > context.params.limit_sq) {
        return false;
    }
    if (!aabb_overlap(context.input.bbs[static_cast<size_t>(pair.i)],
                      context.input.bbs[static_cast<size_t>(pair.j)],
                      context.params)) {
        return false;
    }
    return polygons_intersect(context.input.polys[static_cast<size_t>(pair.i)],
                              context.input.polys[static_cast<size_t>(pair.j)]);
}

bool any_overlap_with_grid(const OverlapInput& input,
                           const OverlapParams& params) {
    const int n = static_cast<int>(input.poses.size());
    UniformGridIndex grid(n, params.limit);
    fill_grid(grid, input.poses);

    std::vector<int> neigh;
    neigh.reserve(64);
    const OverlapContext context{input, params};
    for (int i = 0; i < n; ++i) {
        const auto& pi = input.poses[static_cast<size_t>(i)];
        grid.gather(pi.x, pi.y, neigh);
        for (int j : neigh) {
            if (j <= i) {
                continue;
            }
            if (pair_overlaps(PairIndex{i, j}, context)) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace

bool any_overlap(const Polygon& base_poly,
                 const std::vector<TreePose>& poses,
                 double radius,
                 double eps) {
    if (poses.size() <= 1) {
        return false;
    }

    const double r = resolve_radius(base_poly, radius);

    const double limit = 2.0 * r + eps;
    const double limit_sq = limit * limit;
    const OverlapParams params{limit, limit_sq, eps};

    auto polys = transformed_polygons(base_poly, poses);
    auto bbs = compute_bounding_boxes(polys);
    const OverlapInput input{poses, polys, bbs};
    return any_overlap_with_grid(input, params);
}
