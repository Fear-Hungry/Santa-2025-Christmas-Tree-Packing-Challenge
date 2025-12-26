#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace santa2025 {

struct Point {
    double x = 0.0;
    double y = 0.0;
};

using Polygon = std::vector<Point>;

struct BoundingBox {
    double min_x = 0.0;
    double min_y = 0.0;
    double max_x = 0.0;
    double max_y = 0.0;

    double width() const { return max_x - min_x; }
    double height() const { return max_y - min_y; }
    double square_side() const { return (width() > height()) ? width() : height(); }
};

double polygon_signed_area(const Polygon& poly);
double polygon_area(const Polygon& poly);
Polygon ensure_ccw(Polygon poly);
BoundingBox polygon_bbox(const Polygon& poly);
double polygon_max_radius(const Polygon& poly);

Point rotate_point(const Point& p, double deg);
Polygon rotate_polygon(const Polygon& poly, double deg);
Polygon translate_polygon(const Polygon& poly, double dx, double dy);

bool has_reflection_symmetry_y(const Polygon& poly, double tol = 1e-12);

double bounding_square_side_for_rotation(const Polygon& poly, double deg);
std::pair<double, double> minimize_bounding_square_rotation(
    const Polygon& poly,
    double search_min_deg = 0.0,
    double search_max_deg = 90.0,
    double coarse_step_deg = 0.25,
    double fine_step_deg = 0.01
);

double orient(const Point& a, const Point& b, const Point& c);

Polygon convex_hull(std::vector<Point> pts, double eps = 1e-12);

bool point_in_convex_polygon(const Point& p, const Polygon& poly, double eps = 1e-12);
bool segments_intersect(const Point& a, const Point& b, const Point& c, const Point& d, double eps = 1e-12);
bool point_in_polygon(const Point& p, const Polygon& poly, double eps = 1e-12);
bool polygons_intersect(const Polygon& poly_a, const Polygon& poly_b, double eps = 1e-12);

// Returns true only when polygons overlap with positive area (touching at boundary is NOT considered overlap).
bool polygons_overlap_strict(const Polygon& poly_a, const Polygon& poly_b, double eps = 1e-12);

}  // namespace santa2025
