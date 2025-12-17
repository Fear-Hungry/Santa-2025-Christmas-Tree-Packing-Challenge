#include "geom.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace {
constexpr double kPi = 3.14159265358979323846;

double wrap_deg(double deg) {
    deg = std::fmod(deg, 360.0);
    if (deg <= -180.0) {
        deg += 360.0;
    } else if (deg > 180.0) {
        deg -= 360.0;
    }
    return deg;
}

double rigid_side_for_angle(const std::vector<Point>& verts, double angle_deg) {
    const double rad = angle_deg * kPi / 180.0;
    const double cA = std::cos(rad);
    const double sA = std::sin(rad);

    double min_x = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();

    for (const auto& p : verts) {
        const double x = cA * p.x - sA * p.y;
        const double y = sA * p.x + cA * p.y;
        min_x = std::min(min_x, x);
        max_x = std::max(max_x, x);
        min_y = std::min(min_y, y);
        max_y = std::max(max_y, y);
    }

    return std::max(max_x - min_x, max_y - min_y);
}
}  // namespace

Point rotate_point(const Point& p, double deg) {
    double rad = deg * kPi / 180.0;
    double c = std::cos(rad);
    double s = std::sin(rad);
    return {c * p.x - s * p.y, s * p.x + c * p.y};
}

Polygon transform_polygon(const Polygon& poly, const TreePose& pose) {
    Polygon out;
    out.reserve(poly.size());
    for (const auto& p : poly) {
        Point r = rotate_point(p, pose.deg);
        out.push_back({r.x + pose.x, r.y + pose.y});
    }
    return out;
}

std::vector<Polygon> transformed_polygons(
    const Polygon& poly,
    const std::vector<TreePose>& poses) {
    std::vector<Polygon> result;
    result.reserve(poses.size());
    for (const auto& pose : poses) {
        result.push_back(transform_polygon(poly, pose));
    }
    return result;
}

BoundingBox bounding_box(const Polygon& points) {
    if (points.empty()) {
        return {0.0, 0.0, 0.0, 0.0};
    }
    double min_x = points[0].x;
    double max_x = points[0].x;
    double min_y = points[0].y;
    double max_y = points[0].y;
    for (const auto& p : points) {
        min_x = std::min(min_x, p.x);
        max_x = std::max(max_x, p.x);
        min_y = std::min(min_y, p.y);
        max_y = std::max(max_y, p.y);
    }
    return {min_x, max_x, min_y, max_y};
}

double bounding_square_side(const std::vector<Polygon>& polys) {
    if (polys.empty()) {
        return 0.0;
    }
    double min_x = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();

    for (const auto& poly : polys) {
        auto bb = bounding_box(poly);
        min_x = std::min(min_x, bb.min_x);
        max_x = std::max(max_x, bb.max_x);
        min_y = std::min(min_y, bb.min_y);
        max_y = std::max(max_y, bb.max_y);
    }

    double width = max_x - min_x;
    double height = max_y - min_y;
    return std::max(width, height);
}

double enclosing_circle_radius(const Polygon& poly) {
    double r = 0.0;
    for (const auto& p : poly) {
        double d = std::hypot(p.x, p.y);
        if (d > r) {
            r = d;
        }
    }
    return r;
}

std::string fmt_submission_value(double x, int decimals) {
    std::ostringstream oss;
    oss << 's' << std::fixed << std::setprecision(decimals) << x;
    return oss.str();
}

double optimize_rigid_rotation(const Polygon& base_poly,
                               std::vector<TreePose>& poses) {
    if (poses.empty()) {
        return 0.0;
    }

    // Busca multi-res em [-90, 90] graus.
    std::vector<Point> verts;
    verts.reserve(poses.size() * base_poly.size());
    for (const auto& pose : poses) {
        const double r = pose.deg * kPi / 180.0;
        const double c = std::cos(r);
        const double s = std::sin(r);
        for (const auto& v : base_poly) {
            verts.push_back(Point{c * v.x - s * v.y + pose.x,
                                  s * v.x + c * v.y + pose.y});
        }
    }

    double best_ang = 0.0;
    double best_side = rigid_side_for_angle(verts, 0.0);

    auto sweep = [&](double center, double half_window, double step) {
        double lo = std::max(-90.0, center - half_window);
        double hi = std::min(90.0, center + half_window);
        for (double a = lo; a <= hi + 1e-12; a += step) {
            double side = rigid_side_for_angle(verts, a);
            if (side < best_side) {
                best_side = side;
                best_ang = a;
            }
        }
    };

    // coarse
    for (double a = -90.0; a <= 90.0 + 1e-12; a += 2.0) {
        double side = rigid_side_for_angle(verts, a);
        if (side < best_side) {
            best_side = side;
            best_ang = a;
        }
    }
    // refine
    sweep(best_ang, 4.0, 0.5);
    sweep(best_ang, 1.0, 0.1);
    sweep(best_ang, 0.25, 0.02);

    if (std::abs(best_ang) > 1e-15) {
        for (auto& pose : poses) {
            Point p = rotate_point(Point{pose.x, pose.y}, best_ang);
            pose.x = p.x;
            pose.y = p.y;
            pose.deg = wrap_deg(pose.deg + best_ang);
        }
    }

    // Mantém centros em [-100,100] via translação (não muda o score).
    {
        constexpr double kBound = 100.0;
        constexpr double kMargin = 1e-6;

        double lo_x = -std::numeric_limits<double>::infinity();
        double hi_x = std::numeric_limits<double>::infinity();
        double lo_y = -std::numeric_limits<double>::infinity();
        double hi_y = std::numeric_limits<double>::infinity();
        for (const auto& pose : poses) {
            lo_x = std::max(lo_x, (-kBound + kMargin) - pose.x);
            hi_x = std::min(hi_x, (kBound - kMargin) - pose.x);
            lo_y = std::max(lo_y, (-kBound + kMargin) - pose.y);
            hi_y = std::min(hi_y, (kBound - kMargin) - pose.y);
        }

        double dx = 0.0;
        if (lo_x <= hi_x) {
            if (dx < lo_x) {
                dx = lo_x;
            } else if (dx > hi_x) {
                dx = hi_x;
            }
        }
        double dy = 0.0;
        if (lo_y <= hi_y) {
            if (dy < lo_y) {
                dy = lo_y;
            } else if (dy > hi_y) {
                dy = hi_y;
            }
        }

        if (std::abs(dx) > 0.0 || std::abs(dy) > 0.0) {
            for (auto& pose : poses) {
                pose.x += dx;
                pose.y += dy;
            }
        }
    }

    return best_side;
}

Polygon get_tree_polygon() {
    // Polígono oficial (15 vértices) da competição.
    // Fonte: notebooks públicos (ex.: guntasdhanjal/santa-2025-simple-optimization-v2).
    Polygon poly;
    poly.reserve(15);
    poly.push_back({0.0, 0.8});         // Tip
    poly.push_back({0.125, 0.5});       // Top tier (outer)
    poly.push_back({0.0625, 0.5});      // Top tier (inner)
    poly.push_back({0.2, 0.25});        // Mid tier (outer)
    poly.push_back({0.1, 0.25});        // Mid tier (inner)
    poly.push_back({0.35, 0.0});        // Bottom tier (outer)
    poly.push_back({0.075, 0.0});       // Trunk (top-right)
    poly.push_back({0.075, -0.2});      // Trunk (bottom-right)
    poly.push_back({-0.075, -0.2});     // Trunk (bottom-left)
    poly.push_back({-0.075, 0.0});      // Trunk (top-left)
    poly.push_back({-0.35, 0.0});       // Bottom tier (outer)
    poly.push_back({-0.1, 0.25});       // Mid tier (inner)
    poly.push_back({-0.2, 0.25});       // Mid tier (outer)
    poly.push_back({-0.0625, 0.5});     // Top tier (inner)
    poly.push_back({-0.125, 0.5});      // Top tier (outer)
    return poly;
}
