#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "geometry/geom.hpp"
#include "sa_helpers.hpp"

static inline double cross_point(const Point& a, const Point& b) {
    return a.x * b.y - a.y * b.x;
}

static inline Point sub_point(const Point& a, const Point& b) {
    return Point{a.x - b.x, a.y - b.y};
}

static inline Point mul_point(const Point& a, double s) {
    return Point{a.x * s, a.y * s};
}

// Minimal translation vector (MTV) para separar dois polígonos convexos (inclui casos
// degenerados com overlap ~ 0). Retorna o vetor para mover `a` para fora de `b`.
static inline bool convex_mtv(const Polygon& a, const Polygon& b, Point& out_mtv) {
    if (a.size() < 3 || b.size() < 3) {
        return false;
    }

    const double kAxisEps = 1e-18;
    const double kSeparationEps = -1e-12;
    const double kNudge = 1e-9;

    sa_detail::BestAxis best;
    const sa_detail::AxisCheck check_ab{a, b, kAxisEps, kSeparationEps, best};
    const sa_detail::AxisCheck check_ba{b, a, kAxisEps, kSeparationEps, best};
    if (!check_ab.add_edge_axes() || !check_ba.add_edge_axes()) {
        return false;
    }
    if (!std::isfinite(best.overlap) || !(std::hypot(best.axis.x, best.axis.y) > 0.0)) {
        return false;
    }

    const Point ca = sa_detail::avg_point(a);
    const Point cb = sa_detail::avg_point(b);
    const Point d = sub_point(cb, ca);
    const double s = sa_detail::dot_point(d, best.axis);
    const double sign = (s > 0.0) ? -1.0 : 1.0;

    const double depth = std::max(best.overlap, kNudge);
    out_mtv = mul_point(best.axis, sign * depth);
    return true;
}

static inline double signed_area_poly(const Polygon& poly) {
    if (poly.size() < 3) {
        return 0.0;
    }
    double a = 0.0;
    for (size_t i = 0; i < poly.size(); ++i) {
        const Point& p = poly[i];
        const Point& q = poly[(i + 1) % poly.size()];
        a += cross_point(p, q);
    }
    return 0.5 * a;
}

static inline double area_poly_abs(const Polygon& poly) {
    return std::abs(signed_area_poly(poly));
}

static inline Point centroid_poly(const Polygon& poly) {
    Point c{0.0, 0.0};
    if (poly.size() < 3) {
        return c;
    }
    double a = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    for (size_t i = 0; i < poly.size(); ++i) {
        const Point& p = poly[i];
        const Point& q = poly[(i + 1) % poly.size()];
        const double cr = cross_point(p, q);
        a += cr;
        cx += (p.x + q.x) * cr;
        cy += (p.y + q.y) * cr;
    }
    if (std::abs(a) < 1e-18) {
        return c;
    }
    const double inv = 1.0 / (3.0 * a);
    c.x = cx * inv;
    c.y = cy * inv;
    return c;
}

static inline bool point_in_tri_ccw(const Point& p,
                                    const Point& a,
                                    const Point& b,
                                    const Point& c,
                                    double eps) {
    // Triângulo CCW: dentro se estiver à esquerda de todas as arestas.
    const double c1 = cross_point(sub_point(b, a), sub_point(p, a));
    const double c2 = cross_point(sub_point(c, b), sub_point(p, b));
    const double c3 = cross_point(sub_point(a, c), sub_point(p, c));
    return (c1 >= -eps) && (c2 >= -eps) && (c3 >= -eps);
}

static inline Polygon convex_intersection(const Polygon& subj_ccw, const Polygon& clip_ccw) {
    Polygon out = subj_ccw;
    if (out.empty() || clip_ccw.size() < 3) {
        return {};
    }
    const double eps = 1e-12;
    auto inside = [&](const Point& p, const Point& a, const Point& b) -> bool {
        return cross_point(sub_point(b, a), sub_point(p, a)) >= -eps;
    };
    auto line_intersection = [&](const Point& s,
                                 const Point& e,
                                 const Point& a,
                                 const Point& b) -> Point {
        const Point r = sub_point(e, s);
        const Point d = sub_point(b, a);
        const double denom = cross_point(r, d);
        if (std::abs(denom) < 1e-18) {
            return e;
        }
        const double t = cross_point(sub_point(a, s), d) / denom;
        return Point{s.x + t * r.x, s.y + t * r.y};
    };

    for (size_t ci = 0; ci < clip_ccw.size(); ++ci) {
        const Point& a = clip_ccw[ci];
        const Point& b = clip_ccw[(ci + 1) % clip_ccw.size()];
        Polygon input = std::move(out);
        out.clear();
        if (input.empty()) {
            break;
        }
        Point S = input.back();
        for (const auto& E : input) {
            const bool Ein = inside(E, a, b);
            const bool Sin = inside(S, a, b);
            if (Ein) {
                if (!Sin) {
                    out.push_back(line_intersection(S, E, a, b));
                }
                out.push_back(E);
            } else if (Sin) {
                out.push_back(line_intersection(S, E, a, b));
            }
            S = E;
        }
    }
    return out;
}

static inline std::vector<std::array<Point, 3>> triangulate_polygon(const Polygon& poly_in) {
    if (poly_in.size() < 3) {
        return {};
    }
    Polygon poly = poly_in;
    if (signed_area_poly(poly) < 0.0) {
        std::reverse(poly.begin(), poly.end());  // garante CCW
    }

    const int n = static_cast<int>(poly.size());
    std::vector<int> idx(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        idx[static_cast<size_t>(i)] = i;
    }

    std::vector<std::array<Point, 3>> tris;
    tris.reserve(std::max(0, n - 2));

    const double eps = 1e-14;
    auto is_convex = [&](int ip, int ic, int in) -> bool {
        const Point& a = poly[static_cast<size_t>(ip)];
        const Point& b = poly[static_cast<size_t>(ic)];
        const Point& c = poly[static_cast<size_t>(in)];
        const double cr = cross_point(sub_point(b, a), sub_point(c, b));
        return cr > eps;
    };

    int guard = 0;
    while (idx.size() > 3 && guard++ < 10000) {
        bool clipped = false;
        const int m = static_cast<int>(idx.size());
        for (int k = 0; k < m; ++k) {
            int ip = idx[static_cast<size_t>((k - 1 + m) % m)];
            int ic = idx[static_cast<size_t>(k)];
            int in = idx[static_cast<size_t>((k + 1) % m)];
            if (!is_convex(ip, ic, in)) {
                continue;
            }
            const Point& a = poly[static_cast<size_t>(ip)];
            const Point& b = poly[static_cast<size_t>(ic)];
            const Point& c = poly[static_cast<size_t>(in)];

            bool any_inside = false;
            for (int t = 0; t < m; ++t) {
                int it = idx[static_cast<size_t>(t)];
                if (it == ip || it == ic || it == in) {
                    continue;
                }
                const Point& p = poly[static_cast<size_t>(it)];
                if (point_in_tri_ccw(p, a, b, c, 1e-14)) {
                    any_inside = true;
                    break;
                }
            }
            if (any_inside) {
                continue;
            }

            tris.push_back({a, b, c});
            idx.erase(idx.begin() + k);
            clipped = true;
            break;
        }
        if (!clipped) {
            break;
        }
    }

    if (idx.size() == 3) {
        const Point& a = poly[static_cast<size_t>(idx[0])];
        const Point& b = poly[static_cast<size_t>(idx[1])];
        const Point& c = poly[static_cast<size_t>(idx[2])];
        tris.push_back({a, b, c});
    }

    if (static_cast<int>(tris.size()) != n - 2) {
        // Fallback: fan triangulation (pode ter triângulos fora se poly for bem concavo,
        // mas aqui o polígono é fixo; mantém como segurança).
        tris.clear();
        for (int i = 1; i + 1 < n; ++i) {
            tris.push_back({poly[0], poly[static_cast<size_t>(i)], poly[static_cast<size_t>(i + 1)]});
        }
    }

    return tris;
}

static inline Polygon transform_tri(const std::array<Point, 3>& tri,
                                    double x,
                                    double y,
                                    double cA,
                                    double sA) {
    Polygon out;
    out.reserve(3);
    for (int i = 0; i < 3; ++i) {
        const Point& p = tri[static_cast<size_t>(i)];
        out.push_back(Point{x + cA * p.x - sA * p.y, y + sA * p.x + cA * p.y});
    }
    return out;
}
