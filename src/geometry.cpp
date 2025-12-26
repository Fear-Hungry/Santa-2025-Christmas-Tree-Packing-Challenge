#include "santa2025/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace santa2025 {

double polygon_signed_area(const Polygon& poly) {
    if (poly.size() < 3) {
        return 0.0;
    }
    double acc = 0.0;
    for (size_t i = 0; i < poly.size(); ++i) {
        const auto& a = poly[i];
        const auto& b = poly[(i + 1) % poly.size()];
        acc += a.x * b.y - a.y * b.x;
    }
    return acc * 0.5;
}

double polygon_area(const Polygon& poly) {
    return std::abs(polygon_signed_area(poly));
}

Polygon ensure_ccw(Polygon poly) {
    if (polygon_signed_area(poly) < 0.0) {
        std::reverse(poly.begin(), poly.end());
    }
    return poly;
}

BoundingBox polygon_bbox(const Polygon& poly) {
    if (poly.empty()) {
        return BoundingBox{};
    }

    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();

    for (const auto& p : poly) {
        min_x = std::min(min_x, p.x);
        min_y = std::min(min_y, p.y);
        max_x = std::max(max_x, p.x);
        max_y = std::max(max_y, p.y);
    }
    return BoundingBox{min_x, min_y, max_x, max_y};
}

double polygon_max_radius(const Polygon& poly) {
    double r2 = 0.0;
    for (const auto& p : poly) {
        r2 = std::max(r2, p.x * p.x + p.y * p.y);
    }
    return std::sqrt(r2);
}

Point rotate_point(const Point& p, double deg) {
    const double rad = deg * (3.14159265358979323846 / 180.0);
    const double c = std::cos(rad);
    const double s = std::sin(rad);
    return Point{c * p.x - s * p.y, s * p.x + c * p.y};
}

Polygon rotate_polygon(const Polygon& poly, double deg) {
    Polygon out;
    out.reserve(poly.size());
    for (const auto& p : poly) {
        out.push_back(rotate_point(p, deg));
    }
    return out;
}

Polygon translate_polygon(const Polygon& poly, double dx, double dy) {
    Polygon out;
    out.reserve(poly.size());
    for (const auto& p : poly) {
        out.push_back(Point{p.x + dx, p.y + dy});
    }
    return out;
}

bool has_reflection_symmetry_y(const Polygon& poly, double tol) {
    for (const auto& p : poly) {
        bool found = false;
        for (const auto& q : poly) {
            if (std::abs(q.x + p.x) <= tol && std::abs(q.y - p.y) <= tol) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

double bounding_square_side_for_rotation(const Polygon& poly, double deg) {
    return polygon_bbox(rotate_polygon(poly, deg)).square_side();
}

static void validate_rotation_scan(double search_min, double search_max, double coarse_step, double fine_step) {
    if (search_max < search_min) {
        throw std::invalid_argument("search_max_deg must be >= search_min_deg");
    }
    if (!(coarse_step > 0.0) || !(fine_step > 0.0)) {
        throw std::invalid_argument("coarse_step_deg and fine_step_deg must be > 0");
    }
}

std::pair<double, double> minimize_bounding_square_rotation(
    const Polygon& poly,
    double search_min_deg,
    double search_max_deg,
    double coarse_step_deg,
    double fine_step_deg
) {
    validate_rotation_scan(search_min_deg, search_max_deg, coarse_step_deg, fine_step_deg);

    double best_deg = search_min_deg;
    double best_side = std::numeric_limits<double>::infinity();

    auto scan = [&](double lo, double hi, double step) {
        // Include hi when it lands exactly on the grid.
        const int n = static_cast<int>(std::floor((hi - lo) / step + 1e-12));
        for (int i = 0; i <= n; ++i) {
            const double deg = lo + static_cast<double>(i) * step;
            const double side = bounding_square_side_for_rotation(poly, deg);
            if (side < best_side) {
                best_side = side;
                best_deg = deg;
            }
        }
    };

    scan(search_min_deg, search_max_deg, coarse_step_deg);
    const double lo = std::max(search_min_deg, best_deg - coarse_step_deg);
    const double hi = std::min(search_max_deg, best_deg + coarse_step_deg);
    scan(lo, hi, fine_step_deg);

    return {best_deg, best_side};
}

static long double orient_ld(const Point& a, const Point& b, const Point& c) {
    const long double bax = static_cast<long double>(b.x) - static_cast<long double>(a.x);
    const long double bay = static_cast<long double>(b.y) - static_cast<long double>(a.y);
    const long double cax = static_cast<long double>(c.x) - static_cast<long double>(a.x);
    const long double cay = static_cast<long double>(c.y) - static_cast<long double>(a.y);
    return bax * cay - bay * cax;
}

double orient(const Point& a, const Point& b, const Point& c) {
    return static_cast<double>(orient_ld(a, b, c));
}

Polygon convex_hull(std::vector<Point> pts, double eps) {
    if (pts.size() <= 1) {
        return pts;
    }

    std::sort(pts.begin(), pts.end(), [](const Point& a, const Point& b) {
        if (a.x != b.x) {
            return a.x < b.x;
        }
        return a.y < b.y;
    });

    auto nearly_eq = [&](const Point& a, const Point& b) {
        return std::abs(a.x - b.x) <= eps && std::abs(a.y - b.y) <= eps;
    };
    pts.erase(std::unique(pts.begin(), pts.end(), nearly_eq), pts.end());

    if (pts.size() <= 1) {
        return pts;
    }

    std::vector<Point> lower;
    for (const auto& p : pts) {
        while (lower.size() >= 2 &&
               orient_ld(lower[lower.size() - 2], lower.back(), p) <= static_cast<long double>(eps)) {
            lower.pop_back();
        }
        lower.push_back(p);
    }

    std::vector<Point> upper;
    for (size_t i = pts.size(); i-- > 0;) {
        const auto& p = pts[i];
        while (upper.size() >= 2 &&
               orient_ld(upper[upper.size() - 2], upper.back(), p) <= static_cast<long double>(eps)) {
            upper.pop_back();
        }
        upper.push_back(p);
    }

    lower.pop_back();
    upper.pop_back();
    lower.insert(lower.end(), upper.begin(), upper.end());
    return lower;  // CCW
}

bool point_in_convex_polygon(const Point& p, const Polygon& poly, double eps) {
    if (poly.size() < 3) {
        return false;
    }

    // Assume CCW. If not, flip test.
    const bool ccw = polygon_signed_area(poly) >= 0.0;
    for (size_t i = 0; i < poly.size(); ++i) {
        const auto& a = poly[i];
        const auto& b = poly[(i + 1) % poly.size()];
        const long double o = orient_ld(a, b, p);
        if (ccw) {
            if (o < -static_cast<long double>(eps)) {
                return false;
            }
        } else {
            if (o > static_cast<long double>(eps)) {
                return false;
            }
        }
    }
    return true;
}

static bool on_segment(const Point& a, const Point& b, const Point& p, double eps) {
    return (std::min(a.x, b.x) - eps <= p.x && p.x <= std::max(a.x, b.x) + eps &&
            std::min(a.y, b.y) - eps <= p.y && p.y <= std::max(a.y, b.y) + eps &&
            std::abs(orient_ld(a, b, p)) <= static_cast<long double>(eps));
}

bool segments_intersect(const Point& a, const Point& b, const Point& c, const Point& d, double eps) {
    const long double o1 = orient_ld(a, b, c);
    const long double o2 = orient_ld(a, b, d);
    const long double o3 = orient_ld(c, d, a);
    const long double o4 = orient_ld(c, d, b);
    const long double e = static_cast<long double>(eps);

    const bool ab_straddles = (o1 > e && o2 < -e) || (o1 < -e && o2 > e);
    const bool cd_straddles = (o3 > e && o4 < -e) || (o3 < -e && o4 > e);
    if (ab_straddles && cd_straddles) {
        return true;
    }

    if (std::abs(o1) <= e && on_segment(a, b, c, eps)) {
        return true;
    }
    if (std::abs(o2) <= e && on_segment(a, b, d, eps)) {
        return true;
    }
    if (std::abs(o3) <= e && on_segment(c, d, a, eps)) {
        return true;
    }
    if (std::abs(o4) <= e && on_segment(c, d, b, eps)) {
        return true;
    }

    return false;
}

static bool segments_intersect_proper(const Point& a, const Point& b, const Point& c, const Point& d, double eps) {
    const long double o1 = orient_ld(a, b, c);
    const long double o2 = orient_ld(a, b, d);
    const long double o3 = orient_ld(c, d, a);
    const long double o4 = orient_ld(c, d, b);
    const long double e = static_cast<long double>(eps);

    const bool ab_straddles = (o1 > e && o2 < -e) || (o1 < -e && o2 > e);
    const bool cd_straddles = (o3 > e && o4 < -e) || (o3 < -e && o4 > e);
    return ab_straddles && cd_straddles;
}

static bool point_strictly_inside_polygon(const Point& p, const Polygon& poly, double eps) {
    if (poly.size() < 3) {
        return false;
    }

    for (size_t i = 0; i < poly.size(); ++i) {
        const auto& a = poly[i];
        const auto& b = poly[(i + 1) % poly.size()];
        if (on_segment(a, b, p, eps)) {
            return false;  // boundary is not "inside" for strict overlap checks
        }
    }

    bool inside = false;
    for (size_t i = 0; i < poly.size(); ++i) {
        const auto& a = poly[i];
        const auto& b = poly[(i + 1) % poly.size()];

        const bool ay = (a.y > p.y);
        const bool by = (b.y > p.y);
        if (ay != by) {
            const double x_int = (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x;
            if (x_int > p.x + eps) {
                inside = !inside;
            }
        }
    }
    return inside;
}

bool point_in_polygon(const Point& p, const Polygon& poly, double eps) {
    if (poly.size() < 3) {
        return false;
    }

    bool inside = false;
    for (size_t i = 0; i < poly.size(); ++i) {
        const auto& a = poly[i];
        const auto& b = poly[(i + 1) % poly.size()];

        if (on_segment(a, b, p, eps)) {
            return true;
        }

        const bool ay = (a.y > p.y);
        const bool by = (b.y > p.y);
        if (ay != by) {
            const double x_int = (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x;
            if (x_int > p.x + eps) {
                inside = !inside;
            }
        }
    }
    return inside;
}

bool polygons_intersect(const Polygon& poly_a, const Polygon& poly_b, double eps) {
    if (poly_a.empty() || poly_b.empty()) {
        return false;
    }

    const BoundingBox bb_a = polygon_bbox(poly_a);
    const BoundingBox bb_b = polygon_bbox(poly_b);
    if (bb_a.max_x < bb_b.min_x - eps || bb_b.max_x < bb_a.min_x - eps ||
        bb_a.max_y < bb_b.min_y - eps || bb_b.max_y < bb_a.min_y - eps) {
        return false;
    }

    for (size_t i = 0; i < poly_a.size(); ++i) {
        const auto& a1 = poly_a[i];
        const auto& a2 = poly_a[(i + 1) % poly_a.size()];
        for (size_t j = 0; j < poly_b.size(); ++j) {
            const auto& b1 = poly_b[j];
            const auto& b2 = poly_b[(j + 1) % poly_b.size()];
            if (segments_intersect(a1, a2, b1, b2, eps)) {
                return true;
            }
        }
    }

    if (point_in_polygon(poly_a[0], poly_b, eps)) {
        return true;
    }
    if (point_in_polygon(poly_b[0], poly_a, eps)) {
        return true;
    }
    return false;
}

bool polygons_overlap_strict(const Polygon& poly_a, const Polygon& poly_b, double eps) {
    if (poly_a.empty() || poly_b.empty()) {
        return false;
    }

    const BoundingBox bb_a = polygon_bbox(poly_a);
    const BoundingBox bb_b = polygon_bbox(poly_b);
    if (bb_a.max_x < bb_b.min_x - eps || bb_b.max_x < bb_a.min_x - eps ||
        bb_a.max_y < bb_b.min_y - eps || bb_b.max_y < bb_a.min_y - eps) {
        return false;
    }

    for (size_t i = 0; i < poly_a.size(); ++i) {
        const auto& a1 = poly_a[i];
        const auto& a2 = poly_a[(i + 1) % poly_a.size()];
        for (size_t j = 0; j < poly_b.size(); ++j) {
            const auto& b1 = poly_b[j];
            const auto& b2 = poly_b[(j + 1) % poly_b.size()];
            if (segments_intersect_proper(a1, a2, b1, b2, eps)) {
                return true;
            }
        }
    }

    // Strict containment (not counting boundary as overlap).
    if (point_strictly_inside_polygon(poly_a[0], poly_b, eps)) {
        return true;
    }
    if (point_strictly_inside_polygon(poly_b[0], poly_a, eps)) {
        return true;
    }
    return false;
}

}  // namespace santa2025
