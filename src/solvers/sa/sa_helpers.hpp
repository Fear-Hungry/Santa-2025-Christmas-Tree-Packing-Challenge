#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include "geometry/geom.hpp"

namespace sa_detail {

struct BestAxis {
    double overlap = std::numeric_limits<double>::infinity();
    Point axis{0.0, 0.0};
};

struct AxisCheck {
    const Polygon& a;
    const Polygon& b;
    double axis_eps;
    double separation_eps;
    BestAxis& best;

    bool update_best_axis(const Point& axis_raw) const;
    bool add_edge_axes() const;
};

static inline double dot_point(const Point& a, const Point& b) {
    return a.x * b.x + a.y * b.y;
}

static inline Point avg_point(const Polygon& poly) {
    Point c{0.0, 0.0};
    if (poly.empty()) {
        return c;
    }
    for (const auto& p : poly) {
        c.x += p.x;
        c.y += p.y;
    }
    const double inv = 1.0 / static_cast<double>(poly.size());
    c.x *= inv;
    c.y *= inv;
    return c;
}

static inline void project_poly(const Polygon& poly, const Point& axis, double& out_min, double& out_max) {
    out_min = dot_point(poly[0], axis);
    out_max = out_min;
    for (size_t i = 1; i < poly.size(); ++i) {
        const double v = dot_point(poly[i], axis);
        out_min = std::min(out_min, v);
        out_max = std::max(out_max, v);
    }
}

inline bool AxisCheck::update_best_axis(const Point& axis_raw) const {
    const double len = std::hypot(axis_raw.x, axis_raw.y);
    if (!(len > axis_eps)) {
        return true;
    }
    const Point axis{axis_raw.x / len, axis_raw.y / len};

    double amin = 0.0, amax = 0.0, bmin = 0.0, bmax = 0.0;
    project_poly(a, axis, amin, amax);
    project_poly(b, axis, bmin, bmax);

    const double overlap = std::min(amax, bmax) - std::max(amin, bmin);
    if (overlap < separation_eps) {
        return false;
    }
    if (overlap < best.overlap) {
        best.overlap = overlap;
        best.axis = axis;
    }
    return true;
}

inline bool AxisCheck::add_edge_axes() const {
    const Polygon& poly = a;
    for (size_t i = 0; i < poly.size(); ++i) {
        const Point& p = poly[i];
        const Point& q = poly[(i + 1) % poly.size()];
        const Point e{q.x - p.x, q.y - p.y};
        const Point axis{-e.y, e.x};
        if (!update_best_axis(axis)) {
            return false;
        }
    }
    return true;
}

}  // namespace sa_detail
