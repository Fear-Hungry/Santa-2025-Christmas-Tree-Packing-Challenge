#include "utils/score.hpp"

double score_from_side(double side, int n) {
    if (n <= 0) {
        return 0.0;
    }
    return (side * side) / static_cast<double>(n);
}

double score_instance(const Polygon& base_poly, const std::vector<TreePose>& poses) {
    if (poses.empty()) {
        return 0.0;
    }
    auto polys = transformed_polygons(base_poly, poses);
    double side = bounding_square_side(polys);
    return score_from_side(side, static_cast<int>(poses.size()));
}
