#include "collision.hpp"

#include <cmath>
#include <utility>

#include "geom.hpp"

namespace {

using Segment = std::pair<Point, Point>;

std::vector<Segment> make_segments(const Polygon& poly) {
    std::vector<Segment> segs;
    if (poly.size() < 2) {
        return segs;
    }
    segs.reserve(poly.size());
    for (size_t i = 0; i < poly.size(); ++i) {
        segs.emplace_back(poly[i], poly[(i + 1) % poly.size()]);
    }
    return segs;
}

double orient(const Point& a, const Point& b, const Point& c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

bool point_on_segment(const Point& a, const Point& b, const Point& p, double eps) {
    if (std::abs(orient(a, b, p)) > eps) {
        return false;
    }
    return (std::min(a.x, b.x) - eps <= p.x && p.x <= std::max(a.x, b.x) + eps) &&
           (std::min(a.y, b.y) - eps <= p.y && p.y <= std::max(a.y, b.y) + eps);
}

double collinear_overlap_1d(double a1, double a2, double b1, double b2, double eps) {
    double amin = std::min(a1, a2);
    double amax = std::max(a1, a2);
    double bmin = std::min(b1, b2);
    double bmax = std::max(b1, b2);
    double overlap = std::min(amax, bmax) - std::max(amin, bmin);
    return overlap;
}

// Overlap entre segmentos: permite toque em ponto (endpoint), mas bloqueia cruzamento
// "próprio" e sobreposição colinear com comprimento positivo.
bool segments_overlap_strict(const Point& p1,
                             const Point& p2,
                             const Point& q1,
                             const Point& q2,
                             double eps) {
    double o1 = orient(p1, p2, q1);
    double o2 = orient(p1, p2, q2);
    double o3 = orient(q1, q2, p1);
    double o4 = orient(q1, q2, p2);

    bool proper =
        ((o1 > eps && o2 < -eps) || (o1 < -eps && o2 > eps)) &&
        ((o3 > eps && o4 < -eps) || (o3 < -eps && o4 > eps));
    if (proper) {
        return true;
    }

    // Colinear: só consideramos overlap se houver interseção com comprimento positivo.
    if (std::abs(o1) <= eps && std::abs(o2) <= eps && std::abs(o3) <= eps &&
        std::abs(o4) <= eps) {
        if (std::abs(p1.x - p2.x) >= std::abs(p1.y - p2.y)) {
            return collinear_overlap_1d(p1.x, p2.x, q1.x, q2.x, eps) > eps;
        }
        return collinear_overlap_1d(p1.y, p2.y, q1.y, q2.y, eps) > eps;
    }

    // Caso restante: pode haver toque (endpoint em segmento) -> permitido.
    return false;
}

bool point_in_polygon_strict(const Point& pt, const Polygon& poly, double eps) {
    const double x = pt.x;
    const double y = pt.y;
    bool inside = false;

    size_t n = poly.size();
    if (n < 3) {
        return false;
    }

    for (size_t i = 0; i < n; ++i) {
        const Point& a = poly[i];
        const Point& b = poly[(i + 1) % n];
        if (point_on_segment(a, b, pt, eps)) {
            return false;  // toque na borda => permitido, não é "dentro"
        }
    }

    size_t j = n - 1;
    for (size_t i = 0; i < n; ++i) {
        const auto& pi = poly[i];
        const auto& pj = poly[j];
        bool intersect =
            ((pi.y > y) != (pj.y > y)) &&
            (x < (pj.x - pi.x) * (y - pi.y) / ((pj.y - pi.y) + 1e-18) + pi.x);
        if (intersect) {
            inside = !inside;
        }
        j = i;
    }
    return inside;
}

bool polygons_intersect_impl(const Polygon& p1, const Polygon& p2) {
    auto segs1 = make_segments(p1);
    auto segs2 = make_segments(p2);

    const double eps = 1e-12;
    for (const auto& s1 : segs1) {
        for (const auto& s2 : segs2) {
            if (segments_overlap_strict(s1.first, s1.second, s2.first, s2.second, eps)) {
                return true;
            }
        }
    }

    if (p1.size() >= 3 && point_in_polygon_strict(p1[2], p2, eps)) {
        return true;
    }
    if (p2.size() >= 3 && point_in_polygon_strict(p2[2], p1, eps)) {
        return true;
    }
    return false;
}

}  // namespace

bool polygons_intersect(const Polygon& p1, const Polygon& p2) {
    return polygons_intersect_impl(p1, p2);
}

bool any_overlap(const Polygon& base_poly,
                 const std::vector<TreePose>& poses,
                 double radius,
                 double eps) {
    if (poses.size() <= 1) {
        return false;
    }

    double r = radius;
    if (r < 0.0) {
        r = enclosing_circle_radius(base_poly);
    }

    auto polys = transformed_polygons(base_poly, poses);
    std::vector<Point> centers;
    centers.reserve(poses.size());
    for (const auto& pose : poses) {
        centers.push_back({pose.x, pose.y});
    }

    const double limit_sq = (2.0 * r + eps) * (2.0 * r + eps);
    const size_t n = poses.size();
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double dx = centers[i].x - centers[j].x;
            double dy = centers[i].y - centers[j].y;
            if (dx * dx + dy * dy > limit_sq) {
                continue;
            }
            if (polygons_intersect_impl(polys[i], polys[j])) {
                return true;
            }
        }
    }
    return false;
}
