#include "geom.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kCalipersEps = 1e-12;

double wrap_deg(double deg) {
    deg = std::fmod(deg, 360.0);
    if (deg <= -180.0) {
        deg += 360.0;
    } else if (deg > 180.0) {
        deg -= 360.0;
    }
    return deg;
}

double wrap_deg_90(double deg) {
    deg = wrap_deg(deg);
    if (deg <= -90.0) {
        deg += 180.0;
    } else if (deg > 90.0) {
        deg -= 180.0;
    }
    return deg;
}

double dot_point(const Point& p, const Point& d) {
    return p.x * d.x + p.y * d.y;
}

double cross_point(const Point& a, const Point& b) {
    return a.x * b.y - a.y * b.x;
}

Point sub_point(const Point& a, const Point& b) {
    return Point{a.x - b.x, a.y - b.y};
}

Point perp_point(const Point& a) {
    return Point{-a.y, a.x};
}

double cross_oab(const Point& o, const Point& a, const Point& b) {
    return cross_point(sub_point(a, o), sub_point(b, o));
}

std::vector<Point> convex_hull(std::vector<Point> pts) {
    std::sort(pts.begin(), pts.end(), [](const Point& a, const Point& b) {
        if (a.x != b.x) {
            return a.x < b.x;
        }
        return a.y < b.y;
    });
    pts.erase(std::unique(pts.begin(),
                          pts.end(),
                          [](const Point& a, const Point& b) {
                              return a.x == b.x && a.y == b.y;
                          }),
              pts.end());
    if (pts.size() <= 1) {
        return pts;
    }

    std::vector<Point> lower;
    lower.reserve(pts.size());
    for (const auto& p : pts) {
        while (lower.size() >= 2 &&
               cross_oab(lower[lower.size() - 2], lower.back(), p) <= kCalipersEps) {
            lower.pop_back();
        }
        lower.push_back(p);
    }

    std::vector<Point> upper;
    upper.reserve(pts.size());
    for (size_t i = pts.size(); i-- > 0;) {
        const auto& p = pts[i];
        while (upper.size() >= 2 &&
               cross_oab(upper[upper.size() - 2], upper.back(), p) <= kCalipersEps) {
            upper.pop_back();
        }
        upper.push_back(p);
    }

    // concatena sem duplicar endpoints
    lower.pop_back();
    upper.pop_back();
    lower.insert(lower.end(), upper.begin(), upper.end());
    return lower;
}

int argmax_dot(const std::vector<Point>& pts, const Point& dir) {
    int best = 0;
    double best_v = dot_point(pts[0], dir);
    for (int i = 1; i < static_cast<int>(pts.size()); ++i) {
        double v = dot_point(pts[static_cast<size_t>(i)], dir);
        if (v > best_v) {
            best_v = v;
            best = i;
        }
    }
    return best;
}

int argmin_dot(const std::vector<Point>& pts, const Point& dir) {
    int best = 0;
    double best_v = dot_point(pts[0], dir);
    for (int i = 1; i < static_cast<int>(pts.size()); ++i) {
        double v = dot_point(pts[static_cast<size_t>(i)], dir);
        if (v < best_v) {
            best_v = v;
            best = i;
        }
    }
    return best;
}

int advance_argmax_dot(const std::vector<Point>& hull, int idx, const Point& dir) {
    const int n = static_cast<int>(hull.size());
    for (int it = 0; it < n; ++it) {
        int next = (idx + 1) % n;
        if (dot_point(hull[static_cast<size_t>(next)], dir) >
            dot_point(hull[static_cast<size_t>(idx)], dir) + kCalipersEps) {
            idx = next;
            continue;
        }
        break;
    }
    return idx;
}

int advance_argmin_dot(const std::vector<Point>& hull, int idx, const Point& dir) {
    const int n = static_cast<int>(hull.size());
    for (int it = 0; it < n; ++it) {
        int next = (idx + 1) % n;
        if (dot_point(hull[static_cast<size_t>(next)], dir) <
            dot_point(hull[static_cast<size_t>(idx)], dir) - kCalipersEps) {
            idx = next;
            continue;
        }
        break;
    }
    return idx;
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

struct CalipersResult {
    double angle_deg = 0.0;
    double side = std::numeric_limits<double>::infinity();
};

CalipersResult min_bounding_square_calipers(const std::vector<Point>& hull) {
    CalipersResult res;
    if (hull.empty()) {
        res.side = 0.0;
        return res;
    }
    if (hull.size() == 1) {
        res.side = 0.0;
        return res;
    }
    if (hull.size() == 2) {
        Point d = sub_point(hull[1], hull[0]);
        double L = std::hypot(d.x, d.y);
        if (!(L > 0.0)) {
            res.side = 0.0;
            return res;
        }
        double ang = std::atan2(d.y, d.x) * 180.0 / kPi;
        // Melhor quando o segmento fica em 45° (diagonal do square):
        // θ = 45° - ang, reduz para período de 90°.
        res.angle_deg = wrap_deg_90(45.0 - ang);
        res.side = L / std::sqrt(2.0);
        return res;
    }

    const int n = static_cast<int>(hull.size());
    Point u0 = sub_point(hull[1], hull[0]);
    if (!(std::hypot(u0.x, u0.y) > 0.0)) {
        // Degenerado: fallback seguro.
        res.angle_deg = 0.0;
        res.side = rigid_side_for_angle(hull, 0.0);
        return res;
    }
    Point v0 = perp_point(u0);

    int i_max_u = argmax_dot(hull, u0);
    int i_min_u = argmin_dot(hull, u0);
    int i_max_v = argmax_dot(hull, v0);
    int i_min_v = argmin_dot(hull, v0);

    for (int i = 0; i < n; ++i) {
        const Point u = sub_point(hull[static_cast<size_t>((i + 1) % n)],
                                  hull[static_cast<size_t>(i)]);
        const double len = std::hypot(u.x, u.y);
        if (!(len > 0.0)) {
            continue;
        }
        const Point v = perp_point(u);

        i_max_u = advance_argmax_dot(hull, i_max_u, u);
        i_min_u = advance_argmin_dot(hull, i_min_u, u);
        i_max_v = advance_argmax_dot(hull, i_max_v, v);
        i_min_v = advance_argmin_dot(hull, i_min_v, v);

        double width_raw =
            dot_point(hull[static_cast<size_t>(i_max_u)], u) -
            dot_point(hull[static_cast<size_t>(i_min_u)], u);
        double height_raw =
            dot_point(hull[static_cast<size_t>(i_max_v)], v) -
            dot_point(hull[static_cast<size_t>(i_min_v)], v);
        width_raw = std::max(0.0, width_raw);
        height_raw = std::max(0.0, height_raw);
        double side = std::max(width_raw, height_raw) / len;

        double ang = -std::atan2(u.y, u.x) * 180.0 / kPi;
        ang = wrap_deg_90(ang);

        if (side + 1e-15 < res.side) {
            res.side = side;
            res.angle_deg = ang;
        }
    }

    if (!std::isfinite(res.side)) {
        res.side = rigid_side_for_angle(hull, 0.0);
        res.angle_deg = 0.0;
    }
    return res;
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
    const double r = pose.deg * kPi / 180.0;
    const double c = std::cos(r);
    const double s = std::sin(r);
    for (const auto& p : poly) {
        out.push_back(Point{c * p.x - s * p.y + pose.x,
                            s * p.x + c * p.y + pose.y});
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

    // Final rigid via convex hull + rotating calipers (minimum bounding square).
    // Fallback: mantém uma busca multi-res local para robustez numérica.
    std::vector<Point> hull = convex_hull(verts);
    const std::vector<Point>& eval_pts = hull.size() >= 3 ? hull : verts;

    CalipersResult cal = min_bounding_square_calipers(hull);
    double best_ang = wrap_deg_90(cal.angle_deg);
    double best_side = rigid_side_for_angle(eval_pts, best_ang);

    auto local_refine = [&](double center) {
        auto sweep = [&](double c, double half_window, double step) {
            double lo = std::max(-90.0, c - half_window);
            double hi = std::min(90.0, c + half_window);
            for (double a = lo; a <= hi + 1e-12; a += step) {
                double side = rigid_side_for_angle(eval_pts, a);
                if (side + 1e-15 < best_side) {
                    best_side = side;
                    best_ang = a;
                }
            }
        };

        sweep(center, 4.0, 0.5);
        sweep(best_ang, 1.0, 0.1);
        sweep(best_ang, 0.25, 0.02);
    };

    local_refine(best_ang);

    // Safety net: uma varredura grossa global, agora no hull (rápida).
    {
        double g_ang = 0.0;
        double g_side = rigid_side_for_angle(eval_pts, 0.0);
        for (double a = -90.0; a <= 90.0 + 1e-12; a += 2.0) {
            double side = rigid_side_for_angle(eval_pts, a);
            if (side + 1e-15 < g_side) {
                g_side = side;
                g_ang = a;
            }
        }
        if (g_side + 1e-15 < best_side) {
            best_side = g_side;
            best_ang = g_ang;
            local_refine(best_ang);
        }
    }

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
