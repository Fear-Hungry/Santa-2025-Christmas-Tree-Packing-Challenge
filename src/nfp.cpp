#include "santa2025/nfp.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace santa2025 {
namespace {

constexpr double kKeyScale = 1e6;  // micro-degrees

double normalize_deg_360(double deg) {
    double v = std::fmod(deg, 360.0);
    if (v < 0.0) {
        v += 360.0;
    }
    if (std::abs(v) == 0.0) {
        v = 0.0;
    }
    return v;
}

std::int64_t deg_key(double deg) {
    const double v = normalize_deg_360(deg);
    return static_cast<std::int64_t>(std::llround(v * kKeyScale));
}

double key_to_deg(std::int64_t key) {
    return static_cast<double>(key) / kKeyScale;
}

Polygon negate_poly(const Polygon& poly) {
    Polygon out;
    out.reserve(poly.size());
    for (const auto& p : poly) {
        out.push_back(Point{-p.x, -p.y});
    }
    return out;
}

bool point_in_triangle_ccw(const Point& p, const Point& a, const Point& b, const Point& c, double eps) {
    return orient(a, b, p) >= -eps && orient(b, c, p) >= -eps && orient(c, a, p) >= -eps;
}

std::vector<Polygon> triangulate_polygon_ear_clipping(const Polygon& poly, double eps) {
    const Polygon pts = ensure_ccw(poly);
    const int n = static_cast<int>(pts.size());
    if (n < 3) {
        return {};
    }
    if (n == 3) {
        return {pts};
    }

    std::vector<int> idxs(n);
    std::iota(idxs.begin(), idxs.end(), 0);
    std::vector<Polygon> out;
    out.reserve(static_cast<size_t>(n - 2));

    auto is_convex = [&](int i_prev, int i, int i_next) {
        return orient(pts[static_cast<size_t>(i_prev)], pts[static_cast<size_t>(i)], pts[static_cast<size_t>(i_next)]) >
               eps;
    };

    int guard = 0;
    while (idxs.size() > 3) {
        if (++guard > n * n) {
            throw std::runtime_error("Ear clipping failed: polygon may be non-simple or degenerate.");
        }

        const int m = static_cast<int>(idxs.size());
        bool ear_found = false;
        for (int k = 0; k < m; ++k) {
            const int i_prev = idxs[(k - 1 + m) % m];
            const int i = idxs[k];
            const int i_next = idxs[(k + 1) % m];

            if (!is_convex(i_prev, i, i_next)) {
                continue;
            }

            const Point a = pts[static_cast<size_t>(i_prev)];
            const Point b = pts[static_cast<size_t>(i)];
            const Point c = pts[static_cast<size_t>(i_next)];
            if (std::abs(orient(a, b, c)) <= eps) {
                continue;
            }

            bool contains_other = false;
            for (const int j : idxs) {
                if (j == i_prev || j == i || j == i_next) {
                    continue;
                }
                if (point_in_triangle_ccw(pts[static_cast<size_t>(j)], a, b, c, eps)) {
                    contains_other = true;
                    break;
                }
            }
            if (contains_other) {
                continue;
            }

            out.push_back(Polygon{a, b, c});
            idxs.erase(idxs.begin() + k);
            ear_found = true;
            break;
        }

        if (!ear_found) {
            throw std::runtime_error("Ear clipping failed: could not find an ear.");
        }
    }

    out.push_back(Polygon{
        pts[static_cast<size_t>(idxs[0])],
        pts[static_cast<size_t>(idxs[1])],
        pts[static_cast<size_t>(idxs[2])],
    });
    return out;
}

Polygon minkowski_sum_convex(const Polygon& a, const Polygon& b, double eps) {
    std::vector<Point> pts;
    pts.reserve(a.size() * b.size());
    for (const auto& pa : a) {
        for (const auto& pb : b) {
            pts.push_back(Point{pa.x + pb.x, pa.y + pb.y});
        }
    }
    Polygon hull = convex_hull(std::move(pts), eps);
    if (hull.size() >= 3 && polygon_area(hull) > eps) {
        return hull;
    }
    return {};
}

}  // namespace

NFPCache::NFPCache(Polygon base_poly, double eps) : eps_(eps), base_poly_(ensure_ccw(std::move(base_poly))) {
    base_tris_ = triangulate_polygon_ear_clipping(base_poly_, eps_);
}

const NFP& NFPCache::get(double delta_deg) {
    const std::int64_t key = deg_key(delta_deg);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        return it->second;
    }

    const double key_deg = key_to_deg(key);

    std::vector<Polygon> tris_b;
    tris_b.reserve(base_tris_.size());
    for (const auto& tri : base_tris_) {
        tris_b.push_back(negate_poly(rotate_polygon(tri, key_deg)));
    }

    NFP nfp;
    nfp.delta_deg = key_deg;
    nfp.pieces.reserve(base_tris_.size() * base_tris_.size());
    nfp.piece_bboxes.reserve(base_tris_.size() * base_tris_.size());

    for (const auto& ta : base_tris_) {
        for (const auto& tb : tris_b) {
            Polygon piece = minkowski_sum_convex(ta, tb, eps_);
            if (piece.empty()) {
                continue;
            }
            nfp.piece_bboxes.push_back(polygon_bbox(piece));
            nfp.pieces.push_back(std::move(piece));
        }
    }

    auto [ins, _ok] = cache_.emplace(key, std::move(nfp));
    return ins->second;
}

bool nfp_contains(const NFP& nfp, const Point& t, double eps) {
    for (size_t i = 0; i < nfp.pieces.size(); ++i) {
        const auto& bb = nfp.piece_bboxes[i];
        if (t.x < bb.min_x - eps || t.x > bb.max_x + eps || t.y < bb.min_y - eps || t.y > bb.max_y + eps) {
            continue;
        }
        if (point_in_convex_polygon(t, nfp.pieces[i], eps)) {
            return true;
        }
    }
    return false;
}

bool trees_overlap_nfp(NFPCache& cache, const Pose& a, const Pose& b) {
    const double delta = b.deg - a.deg;
    const Point t_world{b.x - a.x, b.y - a.y};
    const Point t_a = rotate_point(t_world, -a.deg);
    const NFP& nfp = cache.get(delta);
    return nfp_contains(nfp, t_a, cache.eps());
}

}  // namespace santa2025
