#include "collision.hpp"

#include <cmath>
#include <utility>

namespace {

using Segment = std::pair<Point, Point>;

struct Tolerance {
    double eps;
};

struct OrientationSet {
    double o1;
    double o2;
    double o3;
    double o4;
};

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

bool point_on_segment(const Point& a, const Point& b, const Point& p, const Tolerance& tol) {
    if (std::abs(orient(a, b, p)) > tol.eps) {
        return false;
    }
    return (std::min(a.x, b.x) - tol.eps <= p.x && p.x <= std::max(a.x, b.x) + tol.eps) &&
           (std::min(a.y, b.y) - tol.eps <= p.y && p.y <= std::max(a.y, b.y) + tol.eps);
}

double collinear_overlap_1d(double a1, double a2, double b1, double b2) {
    double amin = std::min(a1, a2);
    double amax = std::max(a1, a2);
    double bmin = std::min(b1, b2);
    double bmax = std::max(b1, b2);
    double overlap = std::min(amax, bmax) - std::max(amin, bmin);
    return overlap;
}

OrientationSet segment_orientations(const Segment& s1, const Segment& s2) {
    return {orient(s1.first, s1.second, s2.first),
            orient(s1.first, s1.second, s2.second),
            orient(s2.first, s2.second, s1.first),
            orient(s2.first, s2.second, s1.second)};
}

bool opposite_sign_strict(double a, double b, const Tolerance& tol) {
    if (std::abs(a) <= tol.eps) {
        return false;
    }
    if (std::abs(b) <= tol.eps) {
        return false;
    }
    return (a > 0.0) != (b > 0.0);
}

bool proper_intersection(const OrientationSet& orientation, const Tolerance& tol) {
    return opposite_sign_strict(orientation.o1, orientation.o2, tol) &&
           opposite_sign_strict(orientation.o3, orientation.o4, tol);
}

bool are_collinear(const OrientationSet& orientation, const Tolerance& tol) {
    return std::abs(orientation.o1) <= tol.eps &&
           std::abs(orientation.o2) <= tol.eps &&
           std::abs(orientation.o3) <= tol.eps &&
           std::abs(orientation.o4) <= tol.eps;
}

bool collinear_overlap_positive(const Segment& s1, const Segment& s2, const Tolerance& tol) {
    const Point& p1 = s1.first;
    const Point& p2 = s1.second;
    const Point& q1 = s2.first;
    const Point& q2 = s2.second;
    if (std::abs(p1.x - p2.x) >= std::abs(p1.y - p2.y)) {
        return collinear_overlap_1d(p1.x, p2.x, q1.x, q2.x) > tol.eps;
    }
    return collinear_overlap_1d(p1.y, p2.y, q1.y, q2.y) > tol.eps;
}

// Overlap entre segmentos: permite toque em ponto (endpoint), mas bloqueia cruzamento
// "próprio" e sobreposição colinear com comprimento positivo.
bool segments_overlap_strict(const Segment& s1, const Segment& s2, const Tolerance& tol) {
    const OrientationSet orientation = segment_orientations(s1, s2);
    if (proper_intersection(orientation, tol)) {
        return true;
    }

    // Colinear: só consideramos overlap se houver interseção com comprimento positivo.
    if (are_collinear(orientation, tol)) {
        return collinear_overlap_positive(s1, s2, tol);
    }

    // Caso restante: pode haver toque (endpoint em segmento) -> permitido.
    return false;
}

bool point_on_polygon_edge(const Point& p, const Polygon& poly, const Tolerance& tol) {
    const size_t n = poly.size();
    for (size_t i = 0; i < n; ++i) {
        const Point& a = poly[i];
        const Point& b = poly[(i + 1) % n];
        if (point_on_segment(a, b, p, tol)) {
            return true;
        }
    }
    return false;
}

bool ray_crosses_edge(const Point& p, const Point& a, const Point& b) {
    const bool crosses = (a.y > p.y) != (b.y > p.y);
    if (!crosses) {
        return false;
    }
    const double x_intersect =
        (b.x - a.x) * (p.y - a.y) / ((b.y - a.y) + 1e-18) + a.x;
    return p.x < x_intersect;
}

bool point_in_polygon_ray_cast(const Point& pt, const Polygon& poly) {
    bool inside = false;
    size_t j = poly.size() - 1;
    for (size_t i = 0; i < poly.size(); ++i) {
        if (ray_crosses_edge(pt, poly[i], poly[j])) {
            inside = !inside;
        }
        j = i;
    }
    return inside;
}

bool point_in_polygon_strict(const Point& pt, const Polygon& poly, const Tolerance& tol) {
    const size_t n = poly.size();
    if (n < 3) {
        return false;
    }

    if (point_on_polygon_edge(pt, poly, tol)) {
        return false;  // toque na borda => permitido, não é "dentro"
    }

    return point_in_polygon_ray_cast(pt, poly);
}

bool segments_overlap_any(const std::vector<Segment>& segs1,
                          const std::vector<Segment>& segs2,
                          const Tolerance& tol) {
    for (const auto& s1 : segs1) {
        for (const auto& s2 : segs2) {
            if (segments_overlap_strict(s1, s2, tol)) {
                return true;
            }
        }
    }
    return false;
}

bool polygon_contains_point_strict(const Polygon& container,
                                   const Polygon& candidate,
                                   const Tolerance& tol) {
    if (candidate.size() < 3) {
        return false;
    }
    return point_in_polygon_strict(candidate[2], container, tol);
}

bool polygons_intersect_impl(const Polygon& p1, const Polygon& p2) {
    auto segs1 = make_segments(p1);
    auto segs2 = make_segments(p2);

    const Tolerance tol{1e-12};
    if (segments_overlap_any(segs1, segs2, tol)) {
        return true;
    }
    if (polygon_contains_point_strict(p2, p1, tol)) {
        return true;
    }
    return polygon_contains_point_strict(p1, p2, tol);
}

}  // namespace

bool polygons_intersect(const Polygon& p1, const Polygon& p2) {
    return polygons_intersect_impl(p1, p2);
}
