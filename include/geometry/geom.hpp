#pragma once

#include <string>
#include <vector>

struct Point {
    double x;
    double y;
};

using Polygon = std::vector<Point>;

struct TreePose {
    double x;
    double y;
    double deg;
};

struct BoundingBox {
    double min_x;
    double max_x;
    double min_y;
    double max_y;
};

Point rotate_point(const Point& p, double deg);

Polygon transform_polygon(const Polygon& poly, const TreePose& pose);

std::vector<Polygon> transformed_polygons(
    const Polygon& poly,
    const std::vector<TreePose>& poses);

BoundingBox bounding_box(const Polygon& points);

std::vector<BoundingBox> bounding_boxes(const std::vector<Polygon>& polys);

double bounding_square_side(const std::vector<Polygon>& polys);

double enclosing_circle_radius(const Polygon& poly);

std::string fmt_submission_value(double x, int decimals = 9);

Polygon get_tree_polygon();

// "Final rigid" post-processing: a global rotation applied to all poses.
// Preserves non-overlap and can reduce the axis-aligned bounding square.
double optimize_rigid_rotation(const Polygon& base_poly,
                               std::vector<TreePose>& poses);
