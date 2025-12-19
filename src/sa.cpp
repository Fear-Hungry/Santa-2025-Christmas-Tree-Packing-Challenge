#include "sa.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "collision.hpp"

namespace {

    static double wrap_deg(double deg) {
        deg = std::fmod(deg, 360.0);
        if (deg <= -180.0) {
            deg += 360.0;
        } else if (deg > 180.0) {
            deg -= 360.0;
        }
        return deg;
    }

    static double quantize_value_exact(double x, int decimals) {
        const std::string s = fmt_submission_value(x, decimals);
        return std::stod(s.substr(1));
    }

    static TreePose quantize_pose_wrap_deg_exact(const TreePose& pose, int decimals) {
        return TreePose{quantize_value_exact(pose.x, decimals),
                        quantize_value_exact(pose.y, decimals),
                        quantize_value_exact(wrap_deg(pose.deg), decimals)};
    }

    static double cross_point(const Point& a, const Point& b) {
        return a.x * b.y - a.y * b.x;
    }

    static double dot_point(const Point& a, const Point& b) {
        return a.x * b.x + a.y * b.y;
    }

    static Point sub_point(const Point& a, const Point& b) {
        return Point{a.x - b.x, a.y - b.y};
    }

    static Point add_point(const Point& a, const Point& b) {
        return Point{a.x + b.x, a.y + b.y};
    }

    static Point mul_point(const Point& a, double s) {
        return Point{a.x * s, a.y * s};
    }

    static Point avg_point(const Polygon& poly) {
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

    static void project_poly(const Polygon& poly, const Point& axis, double& out_min, double& out_max) {
        out_min = dot_point(poly[0], axis);
        out_max = out_min;
        for (size_t i = 1; i < poly.size(); ++i) {
            const double v = dot_point(poly[i], axis);
            out_min = std::min(out_min, v);
            out_max = std::max(out_max, v);
        }
    }

    // Minimal translation vector (MTV) para separar dois polígonos convexos (inclui casos
    // degenerados com overlap ~ 0). Retorna o vetor para mover `a` para fora de `b`.
    static bool convex_mtv(const Polygon& a, const Polygon& b, Point& out_mtv) {
        if (a.size() < 3 || b.size() < 3) {
            return false;
        }

        const double kAxisEps = 1e-18;
        const double kNudge = 1e-9;

        double best_overlap = std::numeric_limits<double>::infinity();
        Point best_axis{0.0, 0.0};

        auto consider_axis = [&](const Point& axis_raw) -> bool {
            const double len = std::hypot(axis_raw.x, axis_raw.y);
            if (!(len > kAxisEps)) {
                return true;
            }
            const Point axis{axis_raw.x / len, axis_raw.y / len};

            double amin = 0.0, amax = 0.0, bmin = 0.0, bmax = 0.0;
            project_poly(a, axis, amin, amax);
            project_poly(b, axis, bmin, bmax);

            const double overlap = std::min(amax, bmax) - std::max(amin, bmin);
            if (overlap < -1e-12) {
                return false;
            }
            if (overlap < best_overlap) {
                best_overlap = overlap;
                best_axis = axis;
            }
            return true;
        };

        auto add_edge_axes = [&](const Polygon& poly) -> bool {
            for (size_t i = 0; i < poly.size(); ++i) {
                const Point& p = poly[i];
                const Point& q = poly[(i + 1) % poly.size()];
                const Point e = sub_point(q, p);
                const Point axis{-e.y, e.x};
                if (!consider_axis(axis)) {
                    return false;
                }
            }
            return true;
        };

        if (!add_edge_axes(a) || !add_edge_axes(b)) {
            return false;
        }
        if (!std::isfinite(best_overlap) || !(std::hypot(best_axis.x, best_axis.y) > 0.0)) {
            return false;
        }

        const Point ca = avg_point(a);
        const Point cb = avg_point(b);
        const Point d = sub_point(cb, ca);
        const double s = dot_point(d, best_axis);
        const double sign = (s > 0.0) ? -1.0 : 1.0;

        const double depth = std::max(best_overlap, kNudge);
        out_mtv = mul_point(best_axis, sign * depth);
        return true;
    }

    static double signed_area_poly(const Polygon& poly) {
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

    static double area_poly_abs(const Polygon& poly) {
        return std::abs(signed_area_poly(poly));
    }

    static Point centroid_poly(const Polygon& poly) {
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

    static bool point_in_tri_ccw(const Point& p,
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

    static Polygon convex_intersection(const Polygon& subj_ccw, const Polygon& clip_ccw) {
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

    static std::vector<std::array<Point, 3>> triangulate_polygon(const Polygon& poly_in) {
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

    static Polygon transform_tri(const std::array<Point, 3>& tri,
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

    struct Extents {
        double min_x;
        double max_x;
        double min_y;
        double max_y;
    };

    static Extents compute_extents(const std::vector<BoundingBox>& bbs) {
        Extents e;
        e.min_x = std::numeric_limits<double>::infinity();
        e.max_x = -std::numeric_limits<double>::infinity();
        e.min_y = std::numeric_limits<double>::infinity();
        e.max_y = -std::numeric_limits<double>::infinity();
        for (const auto& bb : bbs) {
            e.min_x = std::min(e.min_x, bb.min_x);
            e.max_x = std::max(e.max_x, bb.max_x);
            e.min_y = std::min(e.min_y, bb.min_y);
            e.max_y = std::max(e.max_y, bb.max_y);
        }
        return e;
    }

    static double side_from_extents(const Extents& e) {
        return std::max(e.max_x - e.min_x, e.max_y - e.min_y);
    }

    static bool aabb_overlap(const BoundingBox& a, const BoundingBox& b) {
        if (a.max_x < b.min_x || b.max_x < a.min_x) {
            return false;
        }
        if (a.max_y < b.min_y || b.max_y < a.min_y) {
            return false;
        }
        return true;
    }

    static std::vector<int> build_extreme_pool(const std::vector<BoundingBox>& bbs,
                                               int topk) {
        const int n = static_cast<int>(bbs.size());
        if (n <= 0) {
            return {};
        }
        topk = std::max(1, std::min(topk, n));

        std::vector<int> idx(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            idx[static_cast<size_t>(i)] = i;
        }

        std::vector<char> mark(static_cast<size_t>(n), 0);
        auto add = [&](auto cmp) {
            std::vector<int> v = idx;
            std::sort(v.begin(), v.end(), cmp);
            for (int i = 0; i < topk; ++i) {
                mark[static_cast<size_t>(v[static_cast<size_t>(i)])] = 1;
            }
        };

        add([&](int a, int b) {
            return bbs[static_cast<size_t>(a)].min_x <
                   bbs[static_cast<size_t>(b)].min_x;
        });
        add([&](int a, int b) {
            return bbs[static_cast<size_t>(a)].max_x >
                   bbs[static_cast<size_t>(b)].max_x;
        });
        add([&](int a, int b) {
            return bbs[static_cast<size_t>(a)].min_y <
                   bbs[static_cast<size_t>(b)].min_y;
        });
        add([&](int a, int b) {
            return bbs[static_cast<size_t>(a)].max_y >
                   bbs[static_cast<size_t>(b)].max_y;
        });

        std::vector<int> pool;
        pool.reserve(static_cast<size_t>(4 * topk));
        for (int i = 0; i < n; ++i) {
            if (mark[static_cast<size_t>(i)]) {
                pool.push_back(i);
            }
        }
        if (pool.empty()) {
            pool = idx;
        }
        return pool;
    }

}  // namespace

SARefiner::SARefiner(const Polygon& base_poly, double radius)
    : base_poly_(base_poly), radius_(radius), base_tris_(triangulate_polygon(base_poly)) {}

void SARefiner::apply_aggressive_preset(Params& p) {
    p.t0 = std::max(p.t0, 0.20) * 1.5;
    p.t1 = std::max(p.t1, 0.012) * 1.3;

    p.step_frac_max = std::min(0.25, p.step_frac_max * 1.7);
    p.step_frac_min = std::min(p.step_frac_max, std::min(0.02, p.step_frac_min * 2.0));

    p.ddeg_max = std::min(45.0, p.ddeg_max * 1.6);
    p.ddeg_min = std::min(p.ddeg_max, std::min(10.0, p.ddeg_min * 1.6));

    p.p_rot = std::min(0.65, p.p_rot + 0.15);
    p.p_random_dir = std::min(0.40, p.p_random_dir + 0.15);
    p.p_pick_extreme = std::min(0.985, p.p_pick_extreme + 0.03);
    p.extreme_topk = std::max(p.extreme_topk, 20);
    if (p.rebuild_extreme_every > 0) {
        p.rebuild_extreme_every = std::max(10, p.rebuild_extreme_every / 2);
    }

    p.kick_prob = std::min(0.08, p.kick_prob + 0.03);
    p.kick_mult = std::min(9.0, p.kick_mult * 2.0);

    p.w_relocate = std::max(p.w_relocate, 0.25);
    p.w_block_translate = std::max(p.w_block_translate, 0.12);
    p.w_block_rotate = std::max(p.w_block_rotate, 0.05);
    p.w_lns = std::max(p.w_lns, 0.01);
    p.w_push_contact = std::max(p.w_push_contact, 0.15);
    p.w_squeeze = std::max(p.w_squeeze, 0.06);

    p.block_size = std::max(p.block_size, 8);
    p.lns_remove = std::max(p.lns_remove, 10);

    p.relocate_attempts = std::max(p.relocate_attempts, 16);
    p.lns_attempts_per_tree = std::max(p.lns_attempts_per_tree, 45);
    p.relocate_noise_frac = std::min(0.18, p.relocate_noise_frac * 1.5);
    p.block_step_frac_max = std::min(0.35, p.block_step_frac_max * 1.4);
    p.block_step_frac_min =
        std::min(p.block_step_frac_max, std::min(0.08, p.block_step_frac_min * 1.5));
    p.block_rot_deg_max = std::min(45.0, p.block_rot_deg_max * 1.4);
    p.block_rot_deg_min =
        std::min(p.block_rot_deg_max, std::min(10.0, p.block_rot_deg_min * 1.4));
    p.block_p_random_dir = std::min(0.25, p.block_p_random_dir + 0.10);

    p.lns_noise_frac = std::min(0.25, p.lns_noise_frac * 1.5);
    p.lns_p_rot = std::min(0.80, p.lns_p_rot + 0.10);

    if (p.hh_segment > 0) {
        p.hh_segment = std::max(20, p.hh_segment / 2);
    }
    if (p.hh_reaction > 0.0) {
        p.hh_reaction = std::min(0.35, p.hh_reaction + 0.10);
    }
    if (p.hh_max_block_weight > 0.0) {
        p.hh_max_block_weight = std::max(p.hh_max_block_weight, 0.35);
    }
    if (p.hh_max_lns_weight > 0.0) {
        p.hh_max_lns_weight = std::max(p.hh_max_lns_weight, 0.15);
    }

    p.push_max_step_frac = std::min(0.85, p.push_max_step_frac * 1.2);
    p.squeeze_pushes = std::max(p.squeeze_pushes, 10);

    if (p.iters > 0 && p.reheat_iters <= 0) {
        p.reheat_iters = std::max(50, p.iters / 25);
    }
    p.reheat_mult = std::max(p.reheat_mult, 1.5);
    p.reheat_step_mult = std::max(p.reheat_step_mult, 1.3);
    p.reheat_max = std::max(p.reheat_max, 3);
}

SARefiner::OverlapInfo SARefiner::overlap_info(const TreePose& a, const TreePose& b) const {
        const double ra = a.deg * 3.14159265358979323846 / 180.0;
        const double rb = b.deg * 3.14159265358979323846 / 180.0;
        const double ca = std::cos(ra);
        const double sa = std::sin(ra);
        const double cb = std::cos(rb);
        const double sb = std::sin(rb);

        double total_area = 0.0;
        Point sum{0.0, 0.0};
        for (const auto& ta : base_tris_) {
            Polygon pa = transform_tri(ta, a.x, a.y, ca, sa);
            for (const auto& tb : base_tris_) {
                Polygon pb = transform_tri(tb, b.x, b.y, cb, sb);
                Polygon inter = convex_intersection(pa, pb);
                double area = area_poly_abs(inter);
                if (!(area > 0.0)) {
                    continue;
                }
                Point c = centroid_poly(inter);
                total_area += area;
                sum.x += c.x * area;
                sum.y += c.y * area;
            }
        }
        OverlapInfo out;
        out.area = total_area;
        if (total_area > 0.0) {
            out.centroid.x = sum.x / total_area;
            out.centroid.y = sum.y / total_area;
        }
        return out;
}

SARefiner::OverlapSeparation SARefiner::overlap_separation(const TreePose& a, const TreePose& b) const {
        const double ra = a.deg * 3.14159265358979323846 / 180.0;
        const double rb = b.deg * 3.14159265358979323846 / 180.0;
        const double ca = std::cos(ra);
        const double sa = std::sin(ra);
        const double cb = std::cos(rb);
        const double sb = std::sin(rb);

        double total_area = 0.0;
        Point mtv_sum{0.0, 0.0};

        for (const auto& ta : base_tris_) {
            Polygon pa = transform_tri(ta, a.x, a.y, ca, sa);
            for (const auto& tb : base_tris_) {
                Polygon pb = transform_tri(tb, b.x, b.y, cb, sb);
                Polygon inter = convex_intersection(pa, pb);
                const double area = area_poly_abs(inter);
                if (!(area > 0.0)) {
                    continue;
                }
                Point mtv{0.0, 0.0};
                if (!convex_mtv(pa, pb, mtv)) {
                    continue;
                }
                total_area += area;
                mtv_sum.x += mtv.x * area;
                mtv_sum.y += mtv.y * area;
            }
        }

        OverlapSeparation out;
        out.area = total_area;
        out.mtv_sum = mtv_sum;
        return out;
}

bool SARefiner::overlap_mtv(const TreePose& a,
                            const TreePose& b,
                            Point& out_mtv,
                            double& out_overlap_area) const {
        OverlapSeparation os = overlap_separation(a, b);
        out_overlap_area = os.area;
        if (!(os.area > 0.0)) {
            out_mtv = Point{0.0, 0.0};
            return false;
        }
        out_mtv = Point{os.mtv_sum.x / os.area, os.mtv_sum.y / os.area};
        return true;
}

SARefiner::Result SARefiner::refine_min_side(const std::vector<TreePose>& start,
                                         uint64_t seed,
                                         const Params& p,
                                         const std::vector<char>* active_mask) const {
        const int n = static_cast<int>(start.size());
        if (n <= 0) {
            return Result{start, 0.0};
        }
        if (active_mask && static_cast<int>(active_mask->size()) != n) {
            throw std::runtime_error("SARefiner: active_mask size inválido.");
        }
        if (p.iters <= 0) {
            auto polys = transformed_polygons(base_poly_, start);
            return Result{start, bounding_square_side(polys)};
        }

        bool use_mask = (active_mask != nullptr);
        std::vector<int> active_indices;
        std::vector<int> active_pos;
        if (use_mask) {
            active_indices.reserve(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) {
                if ((*active_mask)[static_cast<size_t>(i)]) {
                    active_indices.push_back(i);
                }
            }
            if (active_indices.empty()) {
                use_mask = false;
            } else {
                active_pos.assign(static_cast<size_t>(n), -1);
                for (int k = 0; k < static_cast<int>(active_indices.size()); ++k) {
                    active_pos[static_cast<size_t>(active_indices[static_cast<size_t>(k)])] = k;
                }
            }
        }

        auto is_active = [&](int idx) -> bool {
            if (!use_mask) {
                return true;
            }
            return (*active_mask)[static_cast<size_t>(idx)] != 0;
        };

        if (p.quantize_decimals < -1) {
            throw std::runtime_error("SARefiner: quantize_decimals inválido (use -1 para desligar).");
        }

        const int qdec = p.quantize_decimals;
        auto quantize_pose_inplace = [&](TreePose& pose) {
            if (qdec >= 0) {
                pose = quantize_pose_wrap_deg_exact(pose, qdec);
            } else {
                pose.deg = wrap_deg(pose.deg);
            }
        };

        std::vector<TreePose> poses = start;
        for (auto& pose : poses) {
            quantize_pose_inplace(pose);
        }
        std::vector<Polygon> polys;
        polys.reserve(static_cast<size_t>(n));
        std::vector<BoundingBox> bbs;
        bbs.reserve(static_cast<size_t>(n));
        for (const auto& pose : poses) {
            Polygon poly = transform_polygon(base_poly_, pose);
            polys.push_back(std::move(poly));
            bbs.push_back(bounding_box(polys.back()));
        }

        Extents e = compute_extents(bbs);
        double gmnx = e.min_x, gmxx = e.max_x, gmny = e.min_y, gmxy = e.max_y;
        double curr_side = side_from_extents(e);

        Result best;
        best.best_side = curr_side;
        best.best_poses = poses;
        double best_min_dim = std::min(gmxx - gmnx, gmxy - gmny);
        int last_best_iter = 0;

        auto better_than_best = [&](double side, double min_dim) -> bool {
            if (!std::isfinite(best.best_side)) {
                return std::isfinite(side);
            }
            if (side + 1e-15 < best.best_side) {
                return true;
            }
            if (std::abs(side - best.best_side) <= 1e-12 &&
                min_dim + 1e-15 < best_min_dim) {
                return true;
            }
            return false;
        };

        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::normal_distribution<double> normal(0.0, 1.0);
        std::uniform_real_distribution<double> uni_deg(-1.0, 1.0);

	        const double thr = 2.0 * radius_ + 1e-9;
	        const double thr_sq = thr * thr;
	        const double tol = 1e-12;
	        const double overlap_w0 =
	            (p.overlap_weight_start >= 0.0) ? p.overlap_weight_start : p.overlap_weight;
	        const double overlap_w1 =
	            (p.overlap_weight_end >= 0.0) ? p.overlap_weight_end : p.overlap_weight;
	        const bool soft_overlap = (std::max(overlap_w0, overlap_w1) > 0.0);
	        const bool use_mtv_metric = (p.overlap_metric == OverlapMetric::kMtv2);

	        auto overlap_weight_at = [&](int t) -> double {
	            const double w0 = overlap_w0;
	            const double w1 = overlap_w1;
	            if (w0 == w1) {
	                return w1;
	            }
	            double frac = (p.iters > 1) ? static_cast<double>(t) /
	                                              static_cast<double>(p.iters - 1)
	                                        : 1.0;
	            if (p.overlap_weight_power != 1.0) {
	                frac = std::pow(frac, p.overlap_weight_power);
	            }
	            return w0 + (w1 - w0) * frac;
	        };

	        double overlap_weight = soft_overlap ? overlap_weight_at(0) : 0.0;

	        auto clamp_overlap = [&](double metric) -> double {
	            return (metric > p.overlap_eps_area) ? metric : 0.0;
	        };
            auto plateau_term = [&](double width, double height) -> double {
                if (!(p.plateau_eps > 0.0)) {
                    return 0.0;
                }
                return p.plateau_eps * std::min(width, height);
            };
	        auto cost_from = [&](double width, double height, double overlap_value) -> double {
                const double side = std::max(width, height);
                double cost = side + plateau_term(width, height);
	            if (soft_overlap && overlap_weight > 0.0) {
	                cost += overlap_weight * overlap_value;
	            }
	            return cost;
	        };
	        auto overlap_metric = [&](const TreePose& a, const TreePose& b) -> double {
	            if (!use_mtv_metric) {
	                return overlap_info(a, b).area;
	            }
	            Point mtv{0.0, 0.0};
	            double area = 0.0;
	            if (!overlap_mtv(a, b, mtv, area)) {
	                return 0.0;
	            }
	            return mtv.x * mtv.x + mtv.y * mtv.y;
	        };

        // Broad-phase incremental: hash grid uniforme com célula ~ 2r.
        // Como o raio é um círculo envolvente do polígono, qualquer interseção
        // implica dist(centros) <= 2r, então basta olhar 3x3 células.
        struct UniformGrid {
            double cell_size = 1.0;
            double inv_cell_size = 1.0;
            int nx = 1;
            int ny = 1;
            std::vector<std::vector<int>> cells;
            std::vector<int> cell_of;
            std::vector<int> pos_in_cell;

            UniformGrid(int n, double cell_size_in)
                : cell_size(std::max(1e-12, cell_size_in)),
                  inv_cell_size(1.0 / std::max(1e-12, cell_size_in)) {
                const double span = 200.0;
                nx = static_cast<int>(std::ceil(span * inv_cell_size)) + 1;
                ny = nx;
                cells.resize(static_cast<size_t>(nx * ny));
                cell_of.assign(static_cast<size_t>(n), -1);
                pos_in_cell.assign(static_cast<size_t>(n), -1);
            }

            int cell_coord(double v) const {
                double f = (v + 100.0) * inv_cell_size;
                int c = static_cast<int>(std::floor(f));
                if (c < 0) {
                    c = 0;
                } else if (c >= nx) {
                    c = nx - 1;
                }
                return c;
            }

            int cell_id(double x, double y) const {
                const int cx = cell_coord(x);
                const int cy = cell_coord(y);
                return cx + cy * nx;
            }

            void clear() {
                for (auto& v : cells) {
                    v.clear();
                }
                std::fill(cell_of.begin(), cell_of.end(), -1);
                std::fill(pos_in_cell.begin(), pos_in_cell.end(), -1);
            }

            void insert(int idx, double x, double y) {
                const int cid = cell_id(x, y);
                auto& v = cells[static_cast<size_t>(cid)];
                cell_of[static_cast<size_t>(idx)] = cid;
                pos_in_cell[static_cast<size_t>(idx)] = static_cast<int>(v.size());
                v.push_back(idx);
            }

            void erase(int idx) {
                const int cid = cell_of[static_cast<size_t>(idx)];
                if (cid < 0) {
                    return;
                }
                auto& v = cells[static_cast<size_t>(cid)];
                const int pos = pos_in_cell[static_cast<size_t>(idx)];
                const int last = v.back();
                v[static_cast<size_t>(pos)] = last;
                v.pop_back();
                pos_in_cell[static_cast<size_t>(last)] = pos;
                cell_of[static_cast<size_t>(idx)] = -1;
                pos_in_cell[static_cast<size_t>(idx)] = -1;
            }

            void update_position(int idx, double x, double y) {
                const int new_cid = cell_id(x, y);
                const int old_cid = cell_of[static_cast<size_t>(idx)];
                if (old_cid == new_cid) {
                    return;
                }
                if (old_cid >= 0) {
                    erase(idx);
                }
                insert(idx, x, y);
            }

            void rebuild(const std::vector<TreePose>& poses) {
                clear();
                for (int i = 0; i < static_cast<int>(poses.size()); ++i) {
                    insert(i,
                           poses[static_cast<size_t>(i)].x,
                           poses[static_cast<size_t>(i)].y);
                }
            }

            void gather(double x, double y, std::vector<int>& out) const {
                out.clear();
                const int cx = cell_coord(x);
                const int cy = cell_coord(y);
                for (int dx = -1; dx <= 1; ++dx) {
                    int ix = cx + dx;
                    if (ix < 0 || ix >= nx) {
                        continue;
                    }
                    for (int dy = -1; dy <= 1; ++dy) {
                        int iy = cy + dy;
                        if (iy < 0 || iy >= ny) {
                            continue;
                        }
                        const int cid = ix + iy * nx;
                        const auto& v = cells[static_cast<size_t>(cid)];
                        out.insert(out.end(), v.begin(), v.end());
                    }
                }
            }
        };

        UniformGrid grid(n, thr);
        grid.rebuild(poses);

        std::vector<int> neigh;
        neigh.reserve(64);
        std::vector<int> neigh2;
        neigh2.reserve(64);
        std::vector<int> neigh_union;
        neigh_union.reserve(128);

        std::vector<int> mark(static_cast<size_t>(n), 0);
        int mark_stamp = 1;
        auto gather_union = [&](int skip1,
                                int skip2,
                                double ax,
                                double ay,
                                double bx,
                                double by,
                                std::vector<int>& out) {
            if (mark_stamp >= std::numeric_limits<int>::max() - 2) {
                std::fill(mark.begin(), mark.end(), 0);
                mark_stamp = 1;
            }
            const int stamp = mark_stamp++;
            out.clear();
            grid.gather(ax, ay, neigh);
            for (int j : neigh) {
                if (j == skip1 || j == skip2) {
                    continue;
                }
                if (mark[static_cast<size_t>(j)] == stamp) {
                    continue;
                }
                mark[static_cast<size_t>(j)] = stamp;
                out.push_back(j);
            }
            grid.gather(bx, by, neigh);
            for (int j : neigh) {
                if (j == skip1 || j == skip2) {
                    continue;
                }
                if (mark[static_cast<size_t>(j)] == stamp) {
                    continue;
                }
                mark[static_cast<size_t>(j)] = stamp;
                out.push_back(j);
            }
        };

	        double curr_overlap = 0.0;
	        if (soft_overlap) {
	            for (int i = 0; i < n; ++i) {
	                for (int j = i + 1; j < n; ++j) {
	                    double dx = poses[static_cast<size_t>(i)].x - poses[static_cast<size_t>(j)].x;
	                    double dy = poses[static_cast<size_t>(i)].y - poses[static_cast<size_t>(j)].y;
	                    if (dx * dx + dy * dy > thr_sq) {
	                        continue;
	                    }
	                    if (!aabb_overlap(bbs[static_cast<size_t>(i)], bbs[static_cast<size_t>(j)])) {
	                        continue;
	                    }
	                    if (!polygons_intersect(polys[static_cast<size_t>(i)],
	                                            polys[static_cast<size_t>(j)])) {
	                        continue;
	                    }
	                    curr_overlap +=
	                        overlap_metric(poses[static_cast<size_t>(i)],
	                                       poses[static_cast<size_t>(j)]);
	                }
	            }
	            curr_overlap = clamp_overlap(curr_overlap);
	        }
            const double curr_width = gmxx - gmnx;
            const double curr_height = gmxy - gmny;
	        double curr_cost = cost_from(curr_width, curr_height, curr_overlap);

	        if (curr_overlap > p.overlap_eps_area) {
	            best.best_side = std::numeric_limits<double>::infinity();
                best_min_dim = std::numeric_limits<double>::infinity();
	        }

	        enum Op : int {
	            kMicro = 0,
	            kSwapRot = 1,
	            kRelocate = 2,
	            kBlockTranslate = 3,
	            kBlockRotate = 4,
	            kLNS = 5,
	            kPushContact = 6,
	            kSqueeze = 7,
	            kResolveOverlap = 8,
	            kNumOps = 9,
	        };

	        std::array<double, kNumOps> weights = {std::max(0.0, p.w_micro),
	                                               std::max(0.0, p.w_swap_rot),
	                                               std::max(0.0, p.w_relocate),
	                                               std::max(0.0, p.w_block_translate),
	                                               std::max(0.0, p.w_block_rotate),
	                                               std::max(0.0, p.w_lns),
	                                               std::max(0.0, p.w_push_contact),
	                                               std::max(0.0, p.w_squeeze),
	                                               std::max(0.0, p.w_resolve_overlap)};
	        if (!soft_overlap) {
	            weights[kResolveOverlap] = 0.0;
	        } else {
	            if (!(p.push_overshoot_frac > 0.0)) {
	                weights[kPushContact] = 0.0;
	            }
	            weights[kSqueeze] = 0.0;
	        }
	        std::array<double, kNumOps> op_cost = {
	            1.0,
	            2.0,
	            static_cast<double>(std::max(1, p.relocate_attempts)),
	            static_cast<double>(std::max(1, p.block_size)),
	            static_cast<double>(std::max(1, p.block_size)),
	            static_cast<double>(std::max(1, p.lns_remove)) *
	                static_cast<double>(std::max(1, p.lns_attempts_per_tree)),
	            static_cast<double>(std::max(1, p.push_bisect_iters)),
	            static_cast<double>(std::max(1, p.squeeze_pushes)) *
	                static_cast<double>(std::max(1, p.push_bisect_iters)),
	            static_cast<double>(std::max(1, p.resolve_attempts)),
	        };
	        std::array<char, kNumOps> enabled = {
	            static_cast<char>(weights[kMicro] > 0.0),
	            static_cast<char>(weights[kSwapRot] > 0.0),
	            static_cast<char>(weights[kRelocate] > 0.0),
	            static_cast<char>(weights[kBlockTranslate] > 0.0),
	            static_cast<char>(weights[kBlockRotate] > 0.0),
	            static_cast<char>(weights[kLNS] > 0.0),
	            static_cast<char>(weights[kPushContact] > 0.0),
	            static_cast<char>(weights[kSqueeze] > 0.0 && p.squeeze_pushes > 0),
	            static_cast<char>(weights[kResolveOverlap] > 0.0),
	        };
	        if (!(enabled[kMicro] || enabled[kSwapRot] || enabled[kRelocate] ||
	              enabled[kBlockTranslate] || enabled[kBlockRotate] || enabled[kLNS] ||
	              enabled[kPushContact] || enabled[kSqueeze] || enabled[kResolveOverlap])) {
	            enabled[kMicro] = 1;
	            weights[kMicro] = 1.0;
	        }

	        std::array<double, kNumOps> op_score = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	        std::array<int, kNumOps> op_uses = {0, 0, 0, 0, 0, 0, 0, 0, 0};

        auto sum_weights = [&]() -> double {
            double s = 0.0;
            for (int k = 0; k < kNumOps; ++k) {
                if (enabled[static_cast<size_t>(k)]) {
                    s += weights[static_cast<size_t>(k)];
                }
            }
            return std::max(1e-12, s);
        };

        auto pick_op = [&]() -> int {
            double s = sum_weights();
            double r = uni(rng) * s;
            for (int k = 0; k < kNumOps; ++k) {
                if (!enabled[static_cast<size_t>(k)]) {
                    continue;
                }
                double w = weights[static_cast<size_t>(k)];
                if (r < w) {
                    return k;
                }
                r -= w;
            }
            return kMicro;
        };

        if (p.lns_p_uniform < 0.0 || p.lns_p_uniform > 1.0 ||
            p.lns_p_contact < 0.0 || p.lns_p_contact > 1.0 ||
            (p.lns_p_uniform + p.lns_p_contact) > 1.0 + 1e-12) {
            throw std::runtime_error("Parâmetros inválidos: lns_p_uniform/lns_p_contact (precisa somar <= 1).");
        }
        if (!(p.plateau_eps >= 0.0)) {
            throw std::runtime_error("Parâmetros inválidos: plateau_eps precisa ser >= 0.");
        }
        if (!(p.overlap_weight_power > 0.0)) {
            throw std::runtime_error("Parâmetros inválidos: overlap_weight_power precisa ser > 0.");
        }
        if (!(p.push_max_step_frac > 0.0)) {
            throw std::runtime_error("Parâmetros inválidos: push_max_step_frac precisa ser > 0.");
        }
        if (p.push_bisect_iters <= 0) {
            throw std::runtime_error("Parâmetros inválidos: push_bisect_iters precisa ser > 0.");
        }
        if (p.push_overshoot_frac < 0.0 || p.push_overshoot_frac > 1.0) {
            throw std::runtime_error("Parâmetros inválidos: push_overshoot_frac precisa estar em [0,1].");
        }
        if (p.squeeze_pushes < 0) {
            throw std::runtime_error("Parâmetros inválidos: squeeze_pushes precisa ser >= 0.");
        }
        if (p.reheat_iters < 0) {
            throw std::runtime_error("Parâmetros inválidos: reheat_iters precisa ser >= 0.");
        }
        if (p.reheat_mult < 1.0) {
            throw std::runtime_error("Parâmetros inválidos: reheat_mult precisa ser >= 1.0.");
        }
        if (p.reheat_step_mult < 1.0) {
            throw std::runtime_error("Parâmetros inválidos: reheat_step_mult precisa ser >= 1.0.");
        }
        if (p.reheat_max < 0) {
            throw std::runtime_error("Parâmetros inválidos: reheat_max precisa ser >= 0.");
        }

        auto build_boundary_pool = [&]() -> std::vector<int> {
            const double boundary_tol = 1e-9;
            std::vector<int> boundary;
            boundary.reserve(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) {
                if (!is_active(i)) {
                    continue;
                }
                const auto& bb = bbs[static_cast<size_t>(i)];
                if (bb.min_x <= gmnx + boundary_tol || bb.max_x >= gmxx - boundary_tol ||
                    bb.min_y <= gmny + boundary_tol || bb.max_y >= gmxy - boundary_tol) {
                    boundary.push_back(i);
                }
            }
            if (boundary.empty()) {
                boundary = build_extreme_pool(bbs, p.extreme_topk);
                if (use_mask) {
                    std::vector<int> filtered;
                    filtered.reserve(boundary.size());
                    for (int idx : boundary) {
                        if (is_active(idx)) {
                            filtered.push_back(idx);
                        }
                    }
                    boundary = std::move(filtered);
                }
            }
            return boundary;
        };

        std::vector<int> boundary_pool = build_boundary_pool();

        struct PickedIndex {
            int idx;
            bool extreme;
        };

        auto pick_index = [&](double prob_extreme) -> PickedIndex {
            if (!boundary_pool.empty() && uni(rng) < prob_extreme) {
                std::uniform_int_distribution<int> pick(
                    0, static_cast<int>(boundary_pool.size()) - 1);
                return PickedIndex{boundary_pool[static_cast<size_t>(pick(rng))], true};
            }
            if (use_mask) {
                std::uniform_int_distribution<int> pick(0,
                                                        static_cast<int>(active_indices.size()) -
                                                            1);
                return PickedIndex{active_indices[static_cast<size_t>(pick(rng))], false};
            }
            std::uniform_int_distribution<int> pick(0, n - 1);
            return PickedIndex{pick(rng), false};
        };

        auto pick_other_index = [&](int i) -> int {
            if (n <= 1) {
                return 0;
            }
            if (use_mask) {
                if (active_indices.size() <= 1) {
                    return i;
                }
                const int pos_i = active_pos[static_cast<size_t>(i)];
                if (pos_i < 0) {
                    std::uniform_int_distribution<int> pick(
                        0, static_cast<int>(active_indices.size()) - 1);
                    return active_indices[static_cast<size_t>(pick(rng))];
                }
                std::uniform_int_distribution<int> pick(
                    0, static_cast<int>(active_indices.size()) - 2);
                int jpos = pick(rng);
                if (jpos >= pos_i) {
                    ++jpos;
                }
                return active_indices[static_cast<size_t>(jpos)];
            }
            std::uniform_int_distribution<int> pick(0, n - 2);
            int j = pick(rng);
            if (j >= i) {
                ++j;
            }
            return j;
        };

	        auto add_reward = [&](int op,
	                              bool accepted,
	                              bool improved_best,
	                              bool improved_current,
	                              double old_cost,
	                              double new_cost) {
	            op_uses[static_cast<size_t>(op)] += 1;
	            if (!accepted) {
	                return;
	            }
	            const double denom = std::max(1e-12, old_cost);
	            const double rel_improve = (old_cost - new_cost) / denom;
	            if (!(rel_improve > 0.0)) {
	                return;
	            }
            double mult = p.hh_reward_accept;
            if (improved_best) {
                mult = p.hh_reward_best;
            } else if (improved_current) {
                mult = p.hh_reward_improve;
            }
            const double cost = std::max(1e-12, op_cost[static_cast<size_t>(op)]);
            const double reward = (p.hh_reward_scale * rel_improve / cost) * mult;
            if (reward > 0.0) {
                op_score[static_cast<size_t>(op)] += reward;
            }
        };

	        std::array<double, kNumOps> min_w = {p.hh_min_weight,
	                                             p.hh_min_weight,
	                                             p.hh_min_weight,
	                                             0.0,
	                                             0.0,
	                                             0.0,
	                                             0.0,
	                                             0.0,
	                                             0.0};
        const double max_block_w =
            (p.hh_max_block_weight > 0.0) ? p.hh_max_block_weight
                                          : std::numeric_limits<double>::infinity();
        const double max_lns_w =
            (p.hh_max_lns_weight > 0.0) ? p.hh_max_lns_weight
                                        : std::numeric_limits<double>::infinity();
	        std::array<double, kNumOps> max_w = {std::numeric_limits<double>::infinity(),
	                                             std::numeric_limits<double>::infinity(),
	                                             std::numeric_limits<double>::infinity(),
	                                             max_block_w,
	                                             max_block_w,
	                                             max_lns_w,
	                                             std::numeric_limits<double>::infinity(),
	                                             std::numeric_limits<double>::infinity(),
	                                             max_lns_w};

        auto maybe_update_controller = [&](int t) {
            if (p.hh_segment <= 0 || p.hh_reaction <= 0.0) {
                return;
            }
            if (((t + 1) % p.hh_segment) != 0) {
                return;
            }
            for (int k = 0; k < kNumOps; ++k) {
                if (!enabled[static_cast<size_t>(k)]) {
                    continue;
                }
                int uses = op_uses[static_cast<size_t>(k)];
                if (uses <= 0) {
                    continue;
                }
                double avg = op_score[static_cast<size_t>(k)] /
                             static_cast<double>(uses);
                double w = weights[static_cast<size_t>(k)];
                w = (1.0 - p.hh_reaction) * w + p.hh_reaction * avg;
                w = std::max(min_w[static_cast<size_t>(k)],
                             std::min(max_w[static_cast<size_t>(k)], w));
                weights[static_cast<size_t>(k)] = w;
                op_score[static_cast<size_t>(k)] = 0.0;
                op_uses[static_cast<size_t>(k)] = 0;
            }
        };

	        auto accept_move = [&](double old_cost, double new_cost, double T) -> bool {
	            double delta = new_cost - old_cost;
	            if (delta <= 0.0) {
	                return true;
	            }
	            if (!(T > 0.0)) {
	                return false;
	            }
	            double rel = delta / std::max(1e-12, old_cost);
	            double prob = std::exp(-rel / std::max(1e-12, T));
	            return (uni(rng) < prob);
	        };

		        // Pré-compaction determinístico: tenta puxar a casca pra dentro antes do SA estocástico.
		        // Mantém barato (poucos passos) e só aceita melhoria estrita.
		        // Em modo soft-overlap, este pré-passo fica desabilitado para manter `curr_cost`
		        // consistente (o custo depende do overlap).
		        if (!soft_overlap) {
		            int pre_iters = 0;
		            if (p.iters > 0) {
		                pre_iters = std::min(40, std::max(0, p.iters / 80));
		            }
	            if (pre_iters > 0) {
	                for (int it = 0; it < pre_iters; ++it) {
	                    boundary_pool = build_boundary_pool();
	                    if (boundary_pool.empty()) {
	                        break;
	                    }
	                    std::uniform_int_distribution<int> pick(0, static_cast<int>(boundary_pool.size()) - 1);
	                    int i = boundary_pool[static_cast<size_t>(pick(rng))];

	                    const double width = gmxx - gmnx;
	                    const double height = gmxy - gmny;
	                    const bool dom_x = (width >= height);
	                    const double boundary_tol = 1e-9;

	                    double dir_x = 0.0;
	                    double dir_y = 0.0;
	                    const auto& bb = bbs[static_cast<size_t>(i)];
	                    if (dom_x) {
	                        bool left = (bb.min_x <= gmnx + boundary_tol);
	                        bool right = (bb.max_x >= gmxx - boundary_tol);
	                        dir_x = left ? 1.0 : (right ? -1.0 : (0.5 * (gmnx + gmxx) - poses[static_cast<size_t>(i)].x >= 0.0 ? 1.0 : -1.0));
	                        dir_y = 0.0;
	                    } else {
	                        bool bottom = (bb.min_y <= gmny + boundary_tol);
	                        bool top = (bb.max_y >= gmxy - boundary_tol);
	                        dir_x = 0.0;
	                        dir_y = bottom ? 1.0 : (top ? -1.0 : (0.5 * (gmny + gmxy) - poses[static_cast<size_t>(i)].y >= 0.0 ? 1.0 : -1.0));
	                    }

	                    const double step = 0.08 * std::max(1e-9, curr_side);
	                    double dx = dir_x * step + normal(rng) * (0.10 * step);
	                    double dy = dir_y * step + normal(rng) * (0.10 * step);

	                    TreePose cand;
	                    Polygon cand_poly;
	                    BoundingBox cand_bb;
	                    bool ok = false;
	                    double scale = 1.0;
	                    for (int bt = 0; bt < 6; ++bt) {
	                        cand = poses[static_cast<size_t>(i)];
	                        cand.x += dx * scale;
	                        cand.y += dy * scale;
	                        quantize_pose_inplace(cand);
	                        if (cand.x < -100.0 || cand.x > 100.0 || cand.y < -100.0 || cand.y > 100.0) {
	                            scale *= 0.5;
	                            continue;
	                        }
	                        cand_poly = transform_polygon(base_poly_, cand);
	                        cand_bb = bounding_box(cand_poly);

	                        ok = true;
                            grid.gather(cand.x, cand.y, neigh);
                            for (int j : neigh) {
                                if (j == i) {
                                    continue;
                                }
                                double ddx = cand.x - poses[static_cast<size_t>(j)].x;
                                double ddy = cand.y - poses[static_cast<size_t>(j)].y;
                                if (ddx * ddx + ddy * ddy > thr_sq) {
                                    continue;
                                }
                                if (!aabb_overlap(cand_bb, bbs[static_cast<size_t>(j)])) {
                                    continue;
                                }
                                if (polygons_intersect(cand_poly, polys[static_cast<size_t>(j)])) {
                                    ok = false;
                                    break;
                                }
                            }
	                        if (ok) {
	                            break;
	                        }
	                        scale *= 0.5;
	                    }
	                    if (!ok) {
	                        continue;
	                    }

	                    const BoundingBox old_bb = bbs[static_cast<size_t>(i)];
	                    const double old_gmnx = gmnx, old_gmxx = gmxx, old_gmny = gmny, old_gmxy = gmxy;
	                    const double old_side = curr_side;

	                    bbs[static_cast<size_t>(i)] = cand_bb;
	                    gmnx = std::min(gmnx, cand_bb.min_x);
	                    gmxx = std::max(gmxx, cand_bb.max_x);
	                    gmny = std::min(gmny, cand_bb.min_y);
	                    gmxy = std::max(gmxy, cand_bb.max_y);

	                    bool need_full = (old_bb.min_x <= old_gmnx + tol) ||
	                                     (old_bb.max_x >= old_gmxx - tol) ||
	                                     (old_bb.min_y <= old_gmny + tol) ||
	                                     (old_bb.max_y >= old_gmxy - tol);
	                    if (need_full) {
	                        Extents e2 = compute_extents(bbs);
	                        gmnx = e2.min_x;
	                        gmxx = e2.max_x;
	                        gmny = e2.min_y;
	                        gmxy = e2.max_y;
	                    }

	                    double new_side = std::max(gmxx - gmnx, gmxy - gmny);
	                    if (new_side + 1e-15 < old_side) {
                            grid.update_position(i, cand.x, cand.y);
	                        poses[static_cast<size_t>(i)] = cand;
	                        polys[static_cast<size_t>(i)] = std::move(cand_poly);
		                        curr_side = new_side;
		                        if (curr_side + 1e-15 < best.best_side) {
		                            best.best_side = curr_side;
                                    best_min_dim = std::min(gmxx - gmnx, gmxy - gmny);
		                            best.best_poses = poses;
		                        }
	                    } else {
	                        bbs[static_cast<size_t>(i)] = old_bb;
	                        gmnx = old_gmnx;
	                        gmxx = old_gmxx;
	                        gmny = old_gmny;
	                        gmxy = old_gmxy;
	                        curr_side = old_side;
		                }
		            }
		        }
	        }

	        auto build_block = [&](int anchor, int want) -> std::vector<int> {
                const int max_can = use_mask ? static_cast<int>(active_indices.size()) : n;
	            want = std::max(1, std::min(want, max_can));
	            struct DistIdx {
	                double d2;
                int idx;
            };
            std::vector<DistIdx> d;
            d.reserve(static_cast<size_t>(max_can));
            const double ax = poses[static_cast<size_t>(anchor)].x;
            const double ay = poses[static_cast<size_t>(anchor)].y;
            for (int j = 0; j < n; ++j) {
                if (!is_active(j)) {
                    continue;
                }
                double dx = poses[static_cast<size_t>(j)].x - ax;
                double dy = poses[static_cast<size_t>(j)].y - ay;
                d.push_back(DistIdx{dx * dx + dy * dy, j});
            }
            if (d.empty()) {
                return std::vector<int>{anchor};
            }
            if (want > static_cast<int>(d.size())) {
                want = static_cast<int>(d.size());
            }
            std::nth_element(d.begin(),
                             d.begin() + (want - 1),
                             d.end(),
                             [](const DistIdx& a, const DistIdx& b) {
                                 return a.d2 < b.d2;
                             });
            d.resize(static_cast<size_t>(want));
            std::vector<int> out;
            out.reserve(static_cast<size_t>(want));
            for (const auto& it : d) {
                out.push_back(it.idx);
            }
            return out;
        };

        auto compute_extents_mixed =
            [&](const std::vector<BoundingBox>& base,
                const std::vector<int>& moved,
                const std::vector<BoundingBox>& moved_bb) -> Extents {
            std::vector<int> pos(static_cast<size_t>(n), -1);
            for (size_t k = 0; k < moved.size(); ++k) {
                pos[static_cast<size_t>(moved[k])] = static_cast<int>(k);
            }

            Extents e2;
            e2.min_x = std::numeric_limits<double>::infinity();
            e2.max_x = -std::numeric_limits<double>::infinity();
            e2.min_y = std::numeric_limits<double>::infinity();
            e2.max_y = -std::numeric_limits<double>::infinity();

            for (int i = 0; i < n; ++i) {
                int k = pos[static_cast<size_t>(i)];
                if (k >= 0) {
                    const auto& bb = moved_bb[static_cast<size_t>(k)];
                    e2.min_x = std::min(e2.min_x, bb.min_x);
                    e2.max_x = std::max(e2.max_x, bb.max_x);
                    e2.min_y = std::min(e2.min_y, bb.min_y);
                    e2.max_y = std::max(e2.max_y, bb.max_y);
                } else {
                    const auto& bb = base[static_cast<size_t>(i)];
                    e2.min_x = std::min(e2.min_x, bb.min_x);
                    e2.max_x = std::max(e2.max_x, bb.max_x);
                    e2.min_y = std::min(e2.min_y, bb.min_y);
                    e2.max_y = std::max(e2.max_y, bb.max_y);
                }
            }
            return e2;
        };

	        for (int t = 0; t < p.iters; ++t) {
	            double frac = static_cast<double>(t) / std::max(1, p.iters - 1);
	            double T = p.t0 * (1.0 - frac) + p.t1 * frac;

            double step_mult = 1.0;
            if (p.reheat_iters > 0 && p.reheat_max > 0 &&
                (p.reheat_mult > 1.0 || p.reheat_step_mult > 1.0)) {
                const int stall = t - last_best_iter;
                if (stall >= p.reheat_iters) {
                    const int level = std::min(p.reheat_max, stall / p.reheat_iters);
                    const double heat = std::pow(p.reheat_mult, level);
                    step_mult = std::pow(p.reheat_step_mult, level);
                    T *= heat;
                }
            }

            if (soft_overlap) {
                overlap_weight = overlap_weight_at(t);
                const double width = gmxx - gmnx;
                const double height = gmxy - gmny;
                curr_cost = cost_from(width, height, curr_overlap);
            }

            if (p.rebuild_extreme_every > 0 && (t % p.rebuild_extreme_every) == 0) {
                boundary_pool = build_boundary_pool();
            }

            double step =
                (p.step_frac_max * (1.0 - frac) + p.step_frac_min * frac) *
                std::max(1e-9, curr_side);
            double ddeg_rng = p.ddeg_max * (1.0 - frac) + p.ddeg_min * frac;
            step *= step_mult;
            ddeg_rng = std::min(180.0, ddeg_rng * step_mult);
            const double cx = 0.5 * (gmnx + gmxx);
            const double cy = 0.5 * (gmny + gmxy);

	            int op = pick_op();
	            bool accepted = false;
	            bool improved_best = false;
	            bool improved_curr = false;
	            double reward_old_cost = curr_cost;
	            double reward_new_cost = curr_cost;

	            if (op == kMicro) {
	                PickedIndex pick = pick_index(p.p_pick_extreme);
	                int i = pick.idx;

                double vxdir = cx - poses[static_cast<size_t>(i)].x;
                double vydir = cy - poses[static_cast<size_t>(i)].y;
                double vnorm = std::hypot(vxdir, vydir) + 1e-12;
                vxdir /= vnorm;
                vydir /= vnorm;

                double dx = 0.0;
                double dy = 0.0;
                if (uni(rng) < p.p_random_dir) {
                    dx = normal(rng) * step;
                    dy = normal(rng) * step;
                } else {
                    double mag = std::abs(normal(rng)) * step;
                    bool used_boundary_dir = false;
                    if (pick.extreme) {
                        const double width = gmxx - gmnx;
                        const double height = gmxy - gmny;
                        const double boundary_tol = 1e-9;
                        const auto& bb = bbs[static_cast<size_t>(i)];
                        if (width >= height) {
                            bool left = (bb.min_x <= gmnx + boundary_tol);
                            bool right = (bb.max_x >= gmxx - boundary_tol);
                            if (left || right) {
                                used_boundary_dir = true;
                                double dir = left ? 1.0 : -1.0;
                                dx = dir * mag + normal(rng) * (0.10 * step);
                                dy = normal(rng) * (0.35 * step);
                            }
                        } else {
                            bool bottom = (bb.min_y <= gmny + boundary_tol);
                            bool top = (bb.max_y >= gmxy - boundary_tol);
                            if (bottom || top) {
                                used_boundary_dir = true;
                                double dir = bottom ? 1.0 : -1.0;
                                dx = normal(rng) * (0.35 * step);
                                dy = dir * mag + normal(rng) * (0.10 * step);
                            }
                        }
                    }
                    if (!used_boundary_dir) {
                        dx = vxdir * mag + normal(rng) * (0.25 * step);
                        dy = vydir * mag + normal(rng) * (0.25 * step);
                    }
                }

	                if (p.kick_prob > 0.0 && uni(rng) < p.kick_prob) {
	                    dx *= p.kick_mult;
	                    dy *= p.kick_mult;
	                }

	                const bool do_rot = (uni(rng) < p.p_rot);
	                const double ddeg = do_rot ? (uni_deg(rng) * ddeg_rng) : 0.0;

	                TreePose cand;
	                Polygon cand_poly;
	                BoundingBox cand_bb;
	                bool ok = false;
	                double scale = 1.0;
		                for (int bt = 0; bt < 6; ++bt) {
		                    cand = poses[static_cast<size_t>(i)];
		                    cand.x += dx * scale;
		                    cand.y += dy * scale;
		                    if (do_rot) {
		                        cand.deg = wrap_deg(cand.deg + ddeg);
		                    }
		                    quantize_pose_inplace(cand);

		                    if (cand.x < -100.0 || cand.x > 100.0 || cand.y < -100.0 ||
		                        cand.y > 100.0) {
		                        scale *= 0.5;
		                        continue;
		                    }

	                    cand_poly = transform_polygon(base_poly_, cand);
	                    cand_bb = bounding_box(cand_poly);

		                    if (!soft_overlap) {
		                        ok = true;
                                grid.gather(cand.x, cand.y, neigh);
                                for (int j : neigh) {
                                    if (j == i) {
                                        continue;
                                    }
                                    double ddx = cand.x - poses[static_cast<size_t>(j)].x;
                                    double ddy = cand.y - poses[static_cast<size_t>(j)].y;
                                    if (ddx * ddx + ddy * ddy > thr_sq) {
                                        continue;
                                    }
                                    if (!aabb_overlap(cand_bb, bbs[static_cast<size_t>(j)])) {
                                        continue;
                                    }
                                    if (polygons_intersect(cand_poly,
                                                           polys[static_cast<size_t>(j)])) {
                                        ok = false;
                                        break;
                                    }
                                }
		                        if (ok) {
		                            break;
		                        }
		                        scale *= 0.5;
		                        continue;
		                    }
		                    ok = true;
		                    break;
		                }
		                if (!ok) {
		                    add_reward(op, false, false, false, curr_cost, curr_cost);
		                    maybe_update_controller(t);
		                    continue;
		                }

	                const BoundingBox old_bb = bbs[static_cast<size_t>(i)];
	                const double old_gmnx = gmnx, old_gmxx = gmxx, old_gmny = gmny,
	                             old_gmxy = gmxy;
	                const double old_side = curr_side;
	                const double old_cost = curr_cost;
	                const double old_overlap = curr_overlap;

	                bbs[static_cast<size_t>(i)] = cand_bb;
	                gmnx = std::min(gmnx, cand_bb.min_x);
	                gmxx = std::max(gmxx, cand_bb.max_x);
                gmny = std::min(gmny, cand_bb.min_y);
                gmxy = std::max(gmxy, cand_bb.max_y);

                bool need_full = (old_bb.min_x <= old_gmnx + tol) ||
                                 (old_bb.max_x >= old_gmxx - tol) ||
                                 (old_bb.min_y <= old_gmny + tol) ||
                                 (old_bb.max_y >= old_gmxy - tol);
                if (need_full) {
                    Extents e2 = compute_extents(bbs);
                    gmnx = e2.min_x;
                    gmxx = e2.max_x;
                    gmny = e2.min_y;
                    gmxy = e2.max_y;
                }

                    const double new_width = gmxx - gmnx;
                    const double new_height = gmxy - gmny;
	                double new_side = std::max(new_width, new_height);
                    const double new_min_dim = std::min(new_width, new_height);
	                double delta_overlap = 0.0;
	                if (soft_overlap) {
                        gather_union(i,
                                     -1,
                                     poses[static_cast<size_t>(i)].x,
                                     poses[static_cast<size_t>(i)].y,
                                     cand.x,
                                     cand.y,
                                     neigh_union);
	                    for (int j : neigh_union) {
	                        const double old_dx =
	                            poses[static_cast<size_t>(i)].x - poses[static_cast<size_t>(j)].x;
	                        const double old_dy =
	                            poses[static_cast<size_t>(i)].y - poses[static_cast<size_t>(j)].y;
	                        const double new_dx = cand.x - poses[static_cast<size_t>(j)].x;
	                        const double new_dy = cand.y - poses[static_cast<size_t>(j)].y;

	                        double old_a = 0.0;
	                        if ((old_dx * old_dx + old_dy * old_dy <= thr_sq) &&
	                            aabb_overlap(old_bb, bbs[static_cast<size_t>(j)]) &&
	                            polygons_intersect(polys[static_cast<size_t>(i)],
	                                               polys[static_cast<size_t>(j)])) {
	                            old_a = overlap_metric(poses[static_cast<size_t>(i)],
	                                                   poses[static_cast<size_t>(j)]);
	                        }
	                        double new_a = 0.0;
	                        if ((new_dx * new_dx + new_dy * new_dy <= thr_sq) &&
	                            aabb_overlap(cand_bb, bbs[static_cast<size_t>(j)]) &&
	                            polygons_intersect(cand_poly,
	                                               polys[static_cast<size_t>(j)])) {
	                            new_a = overlap_metric(cand,
	                                                   poses[static_cast<size_t>(j)]);
	                        }
	                        delta_overlap += clamp_overlap(new_a) - clamp_overlap(old_a);
	                    }
	                }
	                double new_overlap = std::max(0.0, old_overlap + delta_overlap);
	                new_overlap = clamp_overlap(new_overlap);
	                double new_cost = cost_from(new_width, new_height, new_overlap);
	                if (p.overlap_cost_cap > 0.0 && new_cost > p.overlap_cost_cap) {
	                    new_cost = std::numeric_limits<double>::infinity();
	                }

	                reward_old_cost = old_cost;
	                reward_new_cost = new_cost;
	                accepted = accept_move(old_cost, new_cost, T);
	                if (accepted) {
	                    improved_curr = (new_cost + 1e-15 < old_cost);
	                    const bool valid = (new_overlap <= p.overlap_eps_area);
	                    improved_best = valid && better_than_best(new_side, new_min_dim);
                        grid.update_position(i, cand.x, cand.y);
	                    poses[static_cast<size_t>(i)] = cand;
	                    polys[static_cast<size_t>(i)] = std::move(cand_poly);
	                    curr_side = new_side;
	                    curr_overlap = new_overlap;
	                    curr_cost = new_cost;
	                    if (improved_best) {
	                        best.best_side = curr_side;
                            best_min_dim = new_min_dim;
	                        best.best_poses = poses;
	                    }
	                } else {
	                    bbs[static_cast<size_t>(i)] = old_bb;
	                    gmnx = old_gmnx;
	                    gmxx = old_gmxx;
	                    gmny = old_gmny;
	                    gmxy = old_gmxy;
	                    curr_side = old_side;
	                    curr_overlap = old_overlap;
	                    curr_cost = old_cost;
	                }
            } else if (op == kPushContact) {
                const bool allow_soft_push = soft_overlap && (p.push_overshoot_frac > 0.0);
                if (n <= 1 || (soft_overlap && !allow_soft_push)) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

                const double width = gmxx - gmnx;
                const double height = gmxy - gmny;
                const bool axis_x = (width >= height);
                const double boundary_tol = 1e-9;

                const bool pick_min_side = (uni(rng) < 0.5);
                int i = -1;
                double dir_x = 0.0;
                double dir_y = 0.0;

                std::vector<int> candidates;
                candidates.reserve(static_cast<size_t>(n));
                if (axis_x) {
                    if (pick_min_side) {
                        for (int k = 0; k < n; ++k) {
                            if (!is_active(k)) {
                                continue;
                            }
                            if (bbs[static_cast<size_t>(k)].min_x <= gmnx + boundary_tol) {
                                candidates.push_back(k);
                            }
                        }
                        dir_x = 1.0;
                    } else {
                        for (int k = 0; k < n; ++k) {
                            if (!is_active(k)) {
                                continue;
                            }
                            if (bbs[static_cast<size_t>(k)].max_x >= gmxx - boundary_tol) {
                                candidates.push_back(k);
                            }
                        }
                        dir_x = -1.0;
                    }
                } else {
                    if (pick_min_side) {
                        for (int k = 0; k < n; ++k) {
                            if (!is_active(k)) {
                                continue;
                            }
                            if (bbs[static_cast<size_t>(k)].min_y <= gmny + boundary_tol) {
                                candidates.push_back(k);
                            }
                        }
                        dir_y = 1.0;
                    } else {
                        for (int k = 0; k < n; ++k) {
                            if (!is_active(k)) {
                                continue;
                            }
                            if (bbs[static_cast<size_t>(k)].max_y >= gmxy - boundary_tol) {
                                candidates.push_back(k);
                            }
                        }
                        dir_y = -1.0;
                    }
                }

                if (!candidates.empty()) {
                    std::uniform_int_distribution<int> pick(0,
                                                             static_cast<int>(candidates.size()) - 1);
                    i = candidates[static_cast<size_t>(pick(rng))];
                } else {
                    i = pick_index(p.p_pick_extreme).idx;
                    if (axis_x) {
                        dir_x = (uni(rng) < 0.5) ? 1.0 : -1.0;
                    } else {
                        dir_y = (uni(rng) < 0.5) ? 1.0 : -1.0;
                    }
                }

                double max_step =
                    p.push_max_step_frac * std::max(1e-9, curr_side);
                if (axis_x) {
                    if (dir_x > 0.0) {
                        max_step = std::min(max_step, 100.0 - poses[static_cast<size_t>(i)].x);
                    } else {
                        max_step = std::min(max_step, poses[static_cast<size_t>(i)].x + 100.0);
                    }
                } else {
                    if (dir_y > 0.0) {
                        max_step = std::min(max_step, 100.0 - poses[static_cast<size_t>(i)].y);
                    } else {
                        max_step = std::min(max_step, poses[static_cast<size_t>(i)].y + 100.0);
                    }
                }
                max_step = std::min(max_step, thr);
                if (!(max_step > 1e-12)) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

	                auto valid_at = [&](double delta,
	                                    TreePose& pose_out,
	                                    Polygon& poly_out,
	                                    BoundingBox& bb_out) -> bool {
	                    pose_out = poses[static_cast<size_t>(i)];
	                    pose_out.x += dir_x * delta;
	                    pose_out.y += dir_y * delta;
	                    quantize_pose_inplace(pose_out);
	                    if (pose_out.x < -100.0 || pose_out.x > 100.0 || pose_out.y < -100.0 ||
	                        pose_out.y > 100.0) {
	                        return false;
	                    }
	                    poly_out = transform_polygon(base_poly_, pose_out);
                    bb_out = bounding_box(poly_out);

                    grid.gather(pose_out.x, pose_out.y, neigh);
                    for (int j : neigh) {
                        if (j == i) {
                            continue;
                        }
                        double ddx = pose_out.x - poses[static_cast<size_t>(j)].x;
                        double ddy = pose_out.y - poses[static_cast<size_t>(j)].y;
                        if (ddx * ddx + ddy * ddy > thr_sq) {
                            continue;
                        }
                        if (!aabb_overlap(bb_out, bbs[static_cast<size_t>(j)])) {
                            continue;
                        }
                        if (polygons_intersect(poly_out, polys[static_cast<size_t>(j)])) {
                            return false;
                        }
                    }
                    return true;
                };

                TreePose best_pose;
                Polygon best_poly;
                BoundingBox best_bb;
                double best_delta = 0.0;
                {
                    TreePose pose_hi;
                    Polygon poly_hi;
                    BoundingBox bb_hi;
                    if (valid_at(max_step, pose_hi, poly_hi, bb_hi)) {
                        best_pose = pose_hi;
                        best_poly = std::move(poly_hi);
                        best_bb = bb_hi;
                        best_delta = max_step;
                    } else {
                        double hi_invalid = max_step;
                        double lo_valid = 0.0;
                        TreePose pose_lo;
                        Polygon poly_lo;
                        BoundingBox bb_lo;
                        bool found = false;
                        double step = max_step;
                        for (int bt = 0; bt < 14; ++bt) {
                            step *= 0.5;
                            if (!(step > 1e-12)) {
                                break;
                            }
                            if (valid_at(step, pose_lo, poly_lo, bb_lo)) {
                                lo_valid = step;
                                found = true;
                                break;
                            }
                            hi_invalid = step;
                        }
                        if (!found) {
                            add_reward(op, false, false, false, curr_cost, curr_cost);
                            maybe_update_controller(t);
                            continue;
                        }

                        double lo = lo_valid;
                        double hi = hi_invalid;
                        best_pose = pose_lo;
                        best_poly = std::move(poly_lo);
                        best_bb = bb_lo;
                        best_delta = lo;

                        for (int it = 0; it < p.push_bisect_iters; ++it) {
                            double mid = 0.5 * (lo + hi);
                            TreePose pose_mid;
                            Polygon poly_mid;
                            BoundingBox bb_mid;
                            if (valid_at(mid, pose_mid, poly_mid, bb_mid)) {
                                lo = mid;
                                best_pose = pose_mid;
                                best_poly = std::move(poly_mid);
                                best_bb = bb_mid;
                                best_delta = lo;
                            } else {
                                hi = mid;
                            }
                        }
                    }
                }

                if (!(best_delta > 1e-12)) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

                TreePose cand_pose = best_pose;
                Polygon cand_poly = best_poly;
                BoundingBox cand_bb = best_bb;
                double cand_delta = best_delta;
                if (soft_overlap) {
                    const double overshoot = p.push_overshoot_frac * max_step;
                    if (overshoot > 0.0) {
                        double target_delta = std::min(max_step, best_delta + overshoot);
                        if (target_delta > best_delta + 1e-12) {
                            double delta = target_delta;
                            bool ok = false;
                            for (int bt = 0; bt < 6; ++bt) {
                                TreePose pose_out = poses[static_cast<size_t>(i)];
                                pose_out.x += dir_x * delta;
                                pose_out.y += dir_y * delta;
                                quantize_pose_inplace(pose_out);
                                if (pose_out.x < -100.0 || pose_out.x > 100.0 ||
                                    pose_out.y < -100.0 || pose_out.y > 100.0) {
                                    delta = best_delta + 0.5 * (delta - best_delta);
                                    continue;
                                }
                                Polygon poly_out = transform_polygon(base_poly_, pose_out);
                                BoundingBox bb_out = bounding_box(poly_out);
                                cand_pose = pose_out;
                                cand_poly = std::move(poly_out);
                                cand_bb = bb_out;
                                cand_delta = delta;
                                ok = true;
                                break;
                            }
                            if (!ok) {
                                cand_pose = best_pose;
                                cand_poly = best_poly;
                                cand_bb = best_bb;
                                cand_delta = best_delta;
                            }
                        }
                    }
                }

                if (!(cand_delta > 1e-12)) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

                const BoundingBox old_bb = bbs[static_cast<size_t>(i)];
                const double old_gmnx = gmnx, old_gmxx = gmxx, old_gmny = gmny, old_gmxy = gmxy;
                const double old_side = curr_side;
                const double old_cost = curr_cost;
                const double old_overlap = curr_overlap;

                bbs[static_cast<size_t>(i)] = cand_bb;
                gmnx = std::min(gmnx, cand_bb.min_x);
                gmxx = std::max(gmxx, cand_bb.max_x);
                gmny = std::min(gmny, cand_bb.min_y);
                gmxy = std::max(gmxy, cand_bb.max_y);

                bool need_full = (old_bb.min_x <= old_gmnx + tol) ||
                                 (old_bb.max_x >= old_gmxx - tol) ||
                                 (old_bb.min_y <= old_gmny + tol) ||
                                 (old_bb.max_y >= old_gmxy - tol);
                if (need_full) {
                    Extents e2 = compute_extents(bbs);
                    gmnx = e2.min_x;
                    gmxx = e2.max_x;
                    gmny = e2.min_y;
                    gmxy = e2.max_y;
                }

                const double new_width = gmxx - gmnx;
                const double new_height = gmxy - gmny;
                double new_side = std::max(new_width, new_height);
                const double new_min_dim = std::min(new_width, new_height);
                double new_overlap = curr_overlap;
                if (soft_overlap) {
                    double delta_overlap = 0.0;
                    gather_union(i,
                                 -1,
                                 poses[static_cast<size_t>(i)].x,
                                 poses[static_cast<size_t>(i)].y,
                                 cand_pose.x,
                                 cand_pose.y,
                                 neigh_union);
                    for (int j : neigh_union) {
                        const double old_dx =
                            poses[static_cast<size_t>(i)].x - poses[static_cast<size_t>(j)].x;
                        const double old_dy =
                            poses[static_cast<size_t>(i)].y - poses[static_cast<size_t>(j)].y;
                        const double new_dx = cand_pose.x - poses[static_cast<size_t>(j)].x;
                        const double new_dy = cand_pose.y - poses[static_cast<size_t>(j)].y;

                        double old_a = 0.0;
                        if ((old_dx * old_dx + old_dy * old_dy <= thr_sq) &&
                            aabb_overlap(old_bb, bbs[static_cast<size_t>(j)]) &&
                            polygons_intersect(polys[static_cast<size_t>(i)],
                                               polys[static_cast<size_t>(j)])) {
                            old_a = overlap_metric(poses[static_cast<size_t>(i)],
                                                   poses[static_cast<size_t>(j)]);
                        }
                        double new_a = 0.0;
                        if ((new_dx * new_dx + new_dy * new_dy <= thr_sq) &&
                            aabb_overlap(cand_bb, bbs[static_cast<size_t>(j)]) &&
                            polygons_intersect(cand_poly,
                                               polys[static_cast<size_t>(j)])) {
                            new_a = overlap_metric(cand_pose,
                                                   poses[static_cast<size_t>(j)]);
                        }
                        delta_overlap += clamp_overlap(new_a) - clamp_overlap(old_a);
                    }
                    new_overlap = std::max(0.0, old_overlap + delta_overlap);
                    new_overlap = clamp_overlap(new_overlap);
                }

                double new_cost = cost_from(new_width, new_height, new_overlap);
                if (p.overlap_cost_cap > 0.0 && new_cost > p.overlap_cost_cap) {
                    new_cost = std::numeric_limits<double>::infinity();
                }

                reward_old_cost = old_cost;
                reward_new_cost = new_cost;
                if (soft_overlap) {
                    accepted = accept_move(old_cost, new_cost, T);
                } else {
                    accepted = (new_cost <= old_cost + 1e-15);
                }
                if (accepted) {
                    improved_curr = (new_cost + 1e-15 < old_cost);
                    const bool valid =
                        soft_overlap ? (new_overlap <= p.overlap_eps_area) : true;
                    improved_best = valid && better_than_best(new_side, new_min_dim);
                    grid.update_position(i, cand_pose.x, cand_pose.y);
                    poses[static_cast<size_t>(i)] = cand_pose;
                    polys[static_cast<size_t>(i)] = std::move(cand_poly);
                    curr_side = new_side;
                    if (soft_overlap) {
                        curr_overlap = new_overlap;
                    }
                    curr_cost = new_cost;
                    if (improved_best) {
                        best.best_side = curr_side;
                        best_min_dim = new_min_dim;
                        best.best_poses = poses;
                    }
                } else {
                    bbs[static_cast<size_t>(i)] = old_bb;
                    gmnx = old_gmnx;
                    gmxx = old_gmxx;
                    gmny = old_gmny;
                    gmxy = old_gmxy;
                    curr_side = old_side;
                    if (soft_overlap) {
                        curr_overlap = old_overlap;
                    }
                    curr_cost = old_cost;
                }
            } else if (op == kSqueeze) {
                if (soft_overlap || n <= 1 || p.squeeze_pushes <= 0) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

                const double old_cost = curr_cost;
                bool moved_any = false;

                auto attempt_push = [&](bool axis_x, bool pick_min_side) -> bool {
                    const double boundary_tol = 1e-9;

                    int i = -1;
                    double dir_x = 0.0;
                    double dir_y = 0.0;

                    std::vector<int> candidates;
                    candidates.reserve(static_cast<size_t>(n));
                    if (axis_x) {
                        if (pick_min_side) {
                            for (int k = 0; k < n; ++k) {
                                if (!is_active(k)) {
                                    continue;
                                }
                                if (bbs[static_cast<size_t>(k)].min_x <= gmnx + boundary_tol) {
                                    candidates.push_back(k);
                                }
                            }
                            dir_x = 1.0;
                        } else {
                            for (int k = 0; k < n; ++k) {
                                if (!is_active(k)) {
                                    continue;
                                }
                                if (bbs[static_cast<size_t>(k)].max_x >= gmxx - boundary_tol) {
                                    candidates.push_back(k);
                                }
                            }
                            dir_x = -1.0;
                        }
                    } else {
                        if (pick_min_side) {
                            for (int k = 0; k < n; ++k) {
                                if (!is_active(k)) {
                                    continue;
                                }
                                if (bbs[static_cast<size_t>(k)].min_y <= gmny + boundary_tol) {
                                    candidates.push_back(k);
                                }
                            }
                            dir_y = 1.0;
                        } else {
                            for (int k = 0; k < n; ++k) {
                                if (!is_active(k)) {
                                    continue;
                                }
                                if (bbs[static_cast<size_t>(k)].max_y >= gmxy - boundary_tol) {
                                    candidates.push_back(k);
                                }
                            }
                            dir_y = -1.0;
                        }
                    }

                    if (candidates.empty()) {
                        return false;
                    }
                    std::uniform_int_distribution<int> pick(0, static_cast<int>(candidates.size()) - 1);
                    i = candidates[static_cast<size_t>(pick(rng))];

                    double max_step =
                        p.push_max_step_frac * std::max(1e-9, curr_side);
                    if (axis_x) {
                        if (dir_x > 0.0) {
                            max_step = std::min(max_step, 100.0 - poses[static_cast<size_t>(i)].x);
                        } else {
                            max_step = std::min(max_step, poses[static_cast<size_t>(i)].x + 100.0);
                        }
                    } else {
                        if (dir_y > 0.0) {
                            max_step = std::min(max_step, 100.0 - poses[static_cast<size_t>(i)].y);
                        } else {
                            max_step = std::min(max_step, poses[static_cast<size_t>(i)].y + 100.0);
                        }
                    }
                    max_step = std::min(max_step, thr);
                    if (!(max_step > 1e-12)) {
                        return false;
                    }

	                    auto valid_at = [&](double delta,
	                                        TreePose& pose_out,
	                                        Polygon& poly_out,
	                                        BoundingBox& bb_out) -> bool {
	                        pose_out = poses[static_cast<size_t>(i)];
	                        pose_out.x += dir_x * delta;
	                        pose_out.y += dir_y * delta;
	                        quantize_pose_inplace(pose_out);
	                        if (pose_out.x < -100.0 || pose_out.x > 100.0 || pose_out.y < -100.0 ||
	                            pose_out.y > 100.0) {
	                            return false;
	                        }
	                        poly_out = transform_polygon(base_poly_, pose_out);
                        bb_out = bounding_box(poly_out);

                        grid.gather(pose_out.x, pose_out.y, neigh);
                        for (int j : neigh) {
                            if (j == i) {
                                continue;
                            }
                            double ddx = pose_out.x - poses[static_cast<size_t>(j)].x;
                            double ddy = pose_out.y - poses[static_cast<size_t>(j)].y;
                            if (ddx * ddx + ddy * ddy > thr_sq) {
                                continue;
                            }
                            if (!aabb_overlap(bb_out, bbs[static_cast<size_t>(j)])) {
                                continue;
                            }
                            if (polygons_intersect(poly_out, polys[static_cast<size_t>(j)])) {
                                return false;
                            }
                        }
                        return true;
                    };

                    TreePose best_pose;
                    Polygon best_poly;
                    BoundingBox best_bb;
                    double best_delta = 0.0;
                    {
                        TreePose pose_hi;
                        Polygon poly_hi;
                        BoundingBox bb_hi;
                        if (valid_at(max_step, pose_hi, poly_hi, bb_hi)) {
                            best_pose = pose_hi;
                            best_poly = std::move(poly_hi);
                            best_bb = bb_hi;
                            best_delta = max_step;
                        } else {
                            double hi_invalid = max_step;
                            double lo_valid = 0.0;
                            TreePose pose_lo;
                            Polygon poly_lo;
                            BoundingBox bb_lo;
                            bool found = false;
                            double step = max_step;
                            for (int bt = 0; bt < 14; ++bt) {
                                step *= 0.5;
                                if (!(step > 1e-12)) {
                                    break;
                                }
                                if (valid_at(step, pose_lo, poly_lo, bb_lo)) {
                                    lo_valid = step;
                                    found = true;
                                    break;
                                }
                                hi_invalid = step;
                            }
                            if (!found) {
                                return false;
                            }

                            double lo = lo_valid;
                            double hi = hi_invalid;
                            best_pose = pose_lo;
                            best_poly = std::move(poly_lo);
                            best_bb = bb_lo;
                            best_delta = lo;

                            for (int it = 0; it < p.push_bisect_iters; ++it) {
                                double mid = 0.5 * (lo + hi);
                                TreePose pose_mid;
                                Polygon poly_mid;
                                BoundingBox bb_mid;
                                if (valid_at(mid, pose_mid, poly_mid, bb_mid)) {
                                    lo = mid;
                                    best_pose = pose_mid;
                                    best_poly = std::move(poly_mid);
                                    best_bb = bb_mid;
                                    best_delta = lo;
                                } else {
                                    hi = mid;
                                }
                            }
                        }
                    }
                    if (!(best_delta > 1e-12)) {
                        return false;
                    }

                    const BoundingBox old_bb = bbs[static_cast<size_t>(i)];
                    const double old_gmnx = gmnx, old_gmxx = gmxx, old_gmny = gmny, old_gmxy = gmxy;
                    const double old_side = curr_side;
                    const double old_cost_local = curr_cost;

                    bbs[static_cast<size_t>(i)] = best_bb;
                    gmnx = std::min(gmnx, best_bb.min_x);
                    gmxx = std::max(gmxx, best_bb.max_x);
                    gmny = std::min(gmny, best_bb.min_y);
                    gmxy = std::max(gmxy, best_bb.max_y);

                    bool need_full = (old_bb.min_x <= old_gmnx + tol) ||
                                     (old_bb.max_x >= old_gmxx - tol) ||
                                     (old_bb.min_y <= old_gmny + tol) ||
                                     (old_bb.max_y >= old_gmxy - tol);
                    if (need_full) {
                        Extents e2 = compute_extents(bbs);
                        gmnx = e2.min_x;
                        gmxx = e2.max_x;
                        gmny = e2.min_y;
                        gmxy = e2.max_y;
                    }

                    const double new_width = gmxx - gmnx;
                    const double new_height = gmxy - gmny;
                    double new_side = std::max(new_width, new_height);
                    double new_cost = cost_from(new_width, new_height, curr_overlap);
                    if (!(new_cost <= old_cost_local + 1e-15)) {
                        bbs[static_cast<size_t>(i)] = old_bb;
                        gmnx = old_gmnx;
                        gmxx = old_gmxx;
                        gmny = old_gmny;
                        gmxy = old_gmxy;
                        curr_side = old_side;
                        curr_cost = old_cost_local;
                        return false;
                    }

                    grid.update_position(i, best_pose.x, best_pose.y);
                    poses[static_cast<size_t>(i)] = best_pose;
                    polys[static_cast<size_t>(i)] = std::move(best_poly);
                    curr_side = new_side;
                    curr_cost = new_cost;
                    return true;
                };

                for (int rep = 0; rep < p.squeeze_pushes; ++rep) {
                    const double width = gmxx - gmnx;
                    const double height = gmxy - gmny;
                    const bool axis_x = (width >= height);
                    const bool pick_min_side = (uni(rng) < 0.5);
                    if (attempt_push(axis_x, pick_min_side)) {
                        moved_any = true;
                    }
                }

                reward_old_cost = old_cost;
                reward_new_cost = curr_cost;
                accepted = moved_any;
                improved_curr = accepted && (curr_cost + 1e-15 < old_cost);
                const double curr_min_dim = std::min(gmxx - gmnx, gmxy - gmny);
                improved_best = accepted && better_than_best(curr_side, curr_min_dim);
                if (improved_best) {
                    best.best_side = curr_side;
                    best_min_dim = curr_min_dim;
                    best.best_poses = poses;
                }
            } else if (op == kSwapRot) {
                int i = pick_index(p.p_pick_extreme).idx;
                int j = pick_other_index(i);

	                TreePose cand_i = poses[static_cast<size_t>(i)];
	                TreePose cand_j = poses[static_cast<size_t>(j)];
	                std::swap(cand_i.deg, cand_j.deg);
	                quantize_pose_inplace(cand_i);
	                quantize_pose_inplace(cand_j);

		                Polygon poly_i = transform_polygon(base_poly_, cand_i);
		                Polygon poly_j = transform_polygon(base_poly_, cand_j);
		                BoundingBox bb_i = bounding_box(poly_i);
		                BoundingBox bb_j = bounding_box(poly_j);

	                bool ok = true;
	                if (!soft_overlap) {
	                    {
	                        double ddx = cand_i.x - cand_j.x;
	                        double ddy = cand_i.y - cand_j.y;
	                        if (ddx * ddx + ddy * ddy <= thr_sq) {
	                            if (aabb_overlap(bb_i, bb_j) &&
	                                polygons_intersect(poly_i, poly_j)) {
	                                ok = false;
	                            }
	                        }
	                    }

	                    if (ok) {
                            grid.gather(cand_i.x, cand_i.y, neigh);
                            for (int k : neigh) {
                                if (k == i || k == j) {
                                    continue;
                                }
                                double ddx = cand_i.x - poses[static_cast<size_t>(k)].x;
                                double ddy = cand_i.y - poses[static_cast<size_t>(k)].y;
                                if (ddx * ddx + ddy * ddy <= thr_sq) {
                                    if (aabb_overlap(bb_i, bbs[static_cast<size_t>(k)]) &&
                                        polygons_intersect(poly_i,
                                                           polys[static_cast<size_t>(k)])) {
                                        ok = false;
                                        break;
                                    }
                                }
                            }
                        }
                        if (ok) {
                            grid.gather(cand_j.x, cand_j.y, neigh2);
                            for (int k : neigh2) {
                                if (k == i || k == j) {
                                    continue;
                                }
                                double ddx = cand_j.x - poses[static_cast<size_t>(k)].x;
                                double ddy = cand_j.y - poses[static_cast<size_t>(k)].y;
                                if (ddx * ddx + ddy * ddy <= thr_sq) {
                                    if (aabb_overlap(bb_j, bbs[static_cast<size_t>(k)]) &&
                                        polygons_intersect(poly_j,
                                                           polys[static_cast<size_t>(k)])) {
                                        ok = false;
                                        break;
                                    }
                                }
                            }
	                    }
	                }

	                if (!ok) {
	                    add_reward(op, false, false, false, curr_cost, curr_cost);
	                    maybe_update_controller(t);
	                    continue;
	                }

	                const BoundingBox old_bb_i = bbs[static_cast<size_t>(i)];
	                const BoundingBox old_bb_j = bbs[static_cast<size_t>(j)];
	                const double old_gmnx = gmnx, old_gmxx = gmxx, old_gmny = gmny,
	                             old_gmxy = gmxy;
	                const double old_side = curr_side;
	                const double old_cost = curr_cost;
	                const double old_overlap = curr_overlap;

                bbs[static_cast<size_t>(i)] = bb_i;
                bbs[static_cast<size_t>(j)] = bb_j;
                gmnx = std::min(gmnx, std::min(bb_i.min_x, bb_j.min_x));
                gmxx = std::max(gmxx, std::max(bb_i.max_x, bb_j.max_x));
                gmny = std::min(gmny, std::min(bb_i.min_y, bb_j.min_y));
                gmxy = std::max(gmxy, std::max(bb_i.max_y, bb_j.max_y));

                bool need_full =
                    (old_bb_i.min_x <= old_gmnx + tol) ||
                    (old_bb_i.max_x >= old_gmxx - tol) ||
                    (old_bb_i.min_y <= old_gmny + tol) ||
                    (old_bb_i.max_y >= old_gmxy - tol) ||
                    (old_bb_j.min_x <= old_gmnx + tol) ||
                    (old_bb_j.max_x >= old_gmxx - tol) ||
                    (old_bb_j.min_y <= old_gmny + tol) ||
                    (old_bb_j.max_y >= old_gmxy - tol);
                if (need_full) {
                    Extents e2 = compute_extents(bbs);
                    gmnx = e2.min_x;
                    gmxx = e2.max_x;
                    gmny = e2.min_y;
                    gmxy = e2.max_y;
                }

                    const double new_width = gmxx - gmnx;
                    const double new_height = gmxy - gmny;
	                double new_side = std::max(new_width, new_height);
                    const double new_min_dim = std::min(new_width, new_height);
	                double delta_overlap = 0.0;
	                if (soft_overlap) {
	                    auto old_pair = [&](int a, int b) -> double {
	                        const double dx = poses[static_cast<size_t>(a)].x - poses[static_cast<size_t>(b)].x;
	                        const double dy = poses[static_cast<size_t>(a)].y - poses[static_cast<size_t>(b)].y;
	                        if (dx * dx + dy * dy > thr_sq) {
	                            return 0.0;
	                        }
	                        const BoundingBox& bb_a = (a == i) ? old_bb_i : (a == j ? old_bb_j : bbs[static_cast<size_t>(a)]);
	                        const BoundingBox& bb_b = (b == i) ? old_bb_i : (b == j ? old_bb_j : bbs[static_cast<size_t>(b)]);
	                        const Polygon& poly_a = polys[static_cast<size_t>(a)];
	                        const Polygon& poly_b = polys[static_cast<size_t>(b)];
	                        if (!aabb_overlap(bb_a, bb_b) || !polygons_intersect(poly_a, poly_b)) {
	                            return 0.0;
	                        }
	                        return clamp_overlap(
	                            overlap_metric(poses[static_cast<size_t>(a)],
	                                           poses[static_cast<size_t>(b)]));
	                    };
	                    auto new_pair = [&](int a, int b) -> double {
	                        const TreePose& pa = (a == i) ? cand_i : (a == j ? cand_j : poses[static_cast<size_t>(a)]);
	                        const TreePose& pb = (b == i) ? cand_i : (b == j ? cand_j : poses[static_cast<size_t>(b)]);
	                        const double dx = pa.x - pb.x;
	                        const double dy = pa.y - pb.y;
	                        if (dx * dx + dy * dy > thr_sq) {
	                            return 0.0;
	                        }
	                        const BoundingBox& bb_a = (a == i) ? bb_i : (a == j ? bb_j : bbs[static_cast<size_t>(a)]);
	                        const BoundingBox& bb_b = (b == i) ? bb_i : (b == j ? bb_j : bbs[static_cast<size_t>(b)]);
	                        const Polygon& poly_a = (a == i) ? poly_i : (a == j ? poly_j : polys[static_cast<size_t>(a)]);
	                        const Polygon& poly_b = (b == i) ? poly_i : (b == j ? poly_j : polys[static_cast<size_t>(b)]);
	                        if (!aabb_overlap(bb_a, bb_b) || !polygons_intersect(poly_a, poly_b)) {
	                            return 0.0;
	                        }
	                        return clamp_overlap(overlap_metric(pa, pb));
	                    };

	                    delta_overlap += new_pair(i, j) - old_pair(i, j);
                        gather_union(i,
                                     j,
                                     poses[static_cast<size_t>(i)].x,
                                     poses[static_cast<size_t>(i)].y,
                                     poses[static_cast<size_t>(j)].x,
                                     poses[static_cast<size_t>(j)].y,
                                     neigh_union);
	                    for (int k : neigh_union) {
	                        delta_overlap += new_pair(i, k) - old_pair(i, k);
	                        delta_overlap += new_pair(j, k) - old_pair(j, k);
	                    }
	                }
	                double new_overlap = std::max(0.0, old_overlap + delta_overlap);
	                new_overlap = clamp_overlap(new_overlap);
	                double new_cost = cost_from(new_width, new_height, new_overlap);
	                if (p.overlap_cost_cap > 0.0 && new_cost > p.overlap_cost_cap) {
	                    new_cost = std::numeric_limits<double>::infinity();
	                }

	                reward_old_cost = old_cost;
	                reward_new_cost = new_cost;
	                accepted = accept_move(old_cost, new_cost, T);
	                if (accepted) {
	                    improved_curr = (new_cost + 1e-15 < old_cost);
	                    const bool valid = (new_overlap <= p.overlap_eps_area);
	                    improved_best = valid && better_than_best(new_side, new_min_dim);
	                    poses[static_cast<size_t>(i)].deg = cand_i.deg;
	                    poses[static_cast<size_t>(j)].deg = cand_j.deg;
	                    polys[static_cast<size_t>(i)] = std::move(poly_i);
	                    polys[static_cast<size_t>(j)] = std::move(poly_j);
	                    curr_side = new_side;
	                    curr_overlap = new_overlap;
	                    curr_cost = new_cost;
	                    if (improved_best) {
	                        best.best_side = curr_side;
                            best_min_dim = new_min_dim;
	                        best.best_poses = poses;
	                    }
	                } else {
	                    bbs[static_cast<size_t>(i)] = old_bb_i;
	                    bbs[static_cast<size_t>(j)] = old_bb_j;
	                    gmnx = old_gmnx;
	                    gmxx = old_gmxx;
	                    gmny = old_gmny;
	                    gmxy = old_gmxy;
	                    curr_side = old_side;
	                    curr_overlap = old_overlap;
	                    curr_cost = old_cost;
	                }
	            } else if (op == kRelocate) {
	                int i = pick_index(p.p_pick_extreme).idx;

	                const BoundingBox old_bb = bbs[static_cast<size_t>(i)];
	                const double old_gmnx = gmnx, old_gmxx = gmxx, old_gmny = gmny,
	                             old_gmxy = gmxy;
	                const double old_side = curr_side;
	                const double old_overlap = curr_overlap;
	                const double old_cost = curr_cost;

	                const double noise =
                        p.relocate_noise_frac * std::max(1e-9, curr_side) * step_mult;

	                TreePose best_cand;
	                Polygon best_poly;
	                BoundingBox best_bb;
	                Extents best_extents;
	                double best_new_side = std::numeric_limits<double>::infinity();
                    double best_new_min_dim = std::numeric_limits<double>::infinity();
	                double best_new_overlap = old_overlap;
	                double best_new_cost = std::numeric_limits<double>::infinity();
	                bool found = false;

	                for (int attempt = 0; attempt < std::max(1, p.relocate_attempts); ++attempt) {
	                    double pull =
	                        p.relocate_pull_min +
	                        (p.relocate_pull_max - p.relocate_pull_min) * uni(rng);

	                    TreePose cand = poses[static_cast<size_t>(i)];
		                    cand.x += pull * (cx - cand.x) + normal(rng) * noise;
		                    cand.y += pull * (cy - cand.y) + normal(rng) * noise;
		                    if (uni(rng) < p.relocate_p_rot) {
		                        cand.deg = wrap_deg(cand.deg + uni_deg(rng) * ddeg_rng);
		                    }
		                    quantize_pose_inplace(cand);

		                    if (cand.x < -100.0 || cand.x > 100.0 || cand.y < -100.0 || cand.y > 100.0) {
		                        continue;
		                    }

	                    Polygon cand_poly = transform_polygon(base_poly_, cand);
	                    BoundingBox cand_bb = bounding_box(cand_poly);

		                    if (!soft_overlap) {
		                        bool ok = true;
                                grid.gather(cand.x, cand.y, neigh);
                                for (int k : neigh) {
                                    if (k == i) {
                                        continue;
                                    }
                                    double ddx = cand.x - poses[static_cast<size_t>(k)].x;
                                    double ddy = cand.y - poses[static_cast<size_t>(k)].y;
                                    if (ddx * ddx + ddy * ddy > thr_sq) {
                                        continue;
                                    }
                                    if (!aabb_overlap(cand_bb, bbs[static_cast<size_t>(k)])) {
                                        continue;
                                    }
                                    if (polygons_intersect(cand_poly, polys[static_cast<size_t>(k)])) {
                                        ok = false;
                                        break;
                                    }
                                }
		                        if (!ok) {
		                            continue;
		                        }

	                        // Hard mode: usa o caminho incremental original.
	                        bbs[static_cast<size_t>(i)] = cand_bb;
	                        gmnx = std::min(gmnx, cand_bb.min_x);
	                        gmxx = std::max(gmxx, cand_bb.max_x);
	                        gmny = std::min(gmny, cand_bb.min_y);
	                        gmxy = std::max(gmxy, cand_bb.max_y);

	                        bool need_full = (old_bb.min_x <= old_gmnx + tol) ||
	                                         (old_bb.max_x >= old_gmxx - tol) ||
	                                         (old_bb.min_y <= old_gmny + tol) ||
	                                         (old_bb.max_y >= old_gmxy - tol);
	                        if (need_full) {
	                            Extents e2 = compute_extents(bbs);
	                            gmnx = e2.min_x;
	                            gmxx = e2.max_x;
	                            gmny = e2.min_y;
	                            gmxy = e2.max_y;
	                        }

                            const double cand_width = gmxx - gmnx;
                            const double cand_height = gmxy - gmny;
	                        best_new_side = std::max(cand_width, cand_height);
                            best_new_min_dim = std::min(cand_width, cand_height);
	                        best_new_cost = cost_from(cand_width, cand_height, 0.0);
	                        best_new_overlap = 0.0;
	                        best_cand = cand;
	                        best_poly = std::move(cand_poly);
	                        best_bb = cand_bb;
	                        best_extents = Extents{gmnx, gmxx, gmny, gmxy};
	                        found = true;

	                        // restaura para avaliar acceptance fora.
	                        bbs[static_cast<size_t>(i)] = old_bb;
	                        gmnx = old_gmnx;
	                        gmxx = old_gmxx;
	                        gmny = old_gmny;
	                        gmxy = old_gmxy;
	                        break;
	                    }

		                    // Soft mode: escolhe o melhor candidato por custo.
		                    double delta_overlap = 0.0;
                            gather_union(i,
                                         -1,
                                         poses[static_cast<size_t>(i)].x,
                                         poses[static_cast<size_t>(i)].y,
                                         cand.x,
                                         cand.y,
                                         neigh_union);
		                    for (int k : neigh_union) {
		                        const double old_dx =
		                            poses[static_cast<size_t>(i)].x - poses[static_cast<size_t>(k)].x;
		                        const double old_dy =
		                            poses[static_cast<size_t>(i)].y - poses[static_cast<size_t>(k)].y;
	                        const double new_dx = cand.x - poses[static_cast<size_t>(k)].x;
	                        const double new_dy = cand.y - poses[static_cast<size_t>(k)].y;

	                        double old_a = 0.0;
	                        if ((old_dx * old_dx + old_dy * old_dy <= thr_sq) &&
	                            aabb_overlap(old_bb, bbs[static_cast<size_t>(k)]) &&
	                            polygons_intersect(polys[static_cast<size_t>(i)],
	                                               polys[static_cast<size_t>(k)])) {
	                            old_a = overlap_metric(poses[static_cast<size_t>(i)],
	                                                   poses[static_cast<size_t>(k)]);
	                        }
	                        double new_a = 0.0;
	                        if ((new_dx * new_dx + new_dy * new_dy <= thr_sq) &&
	                            aabb_overlap(cand_bb, bbs[static_cast<size_t>(k)]) &&
	                            polygons_intersect(cand_poly, polys[static_cast<size_t>(k)])) {
	                            new_a = overlap_metric(cand, poses[static_cast<size_t>(k)]);
	                        }
	                        delta_overlap += clamp_overlap(new_a) - clamp_overlap(old_a);
	                    }
	                    double new_overlap = std::max(0.0, old_overlap + delta_overlap);
	                    new_overlap = clamp_overlap(new_overlap);

	                    Extents e2;
	                    e2.min_x = std::numeric_limits<double>::infinity();
	                    e2.max_x = -std::numeric_limits<double>::infinity();
	                    e2.min_y = std::numeric_limits<double>::infinity();
	                    e2.max_y = -std::numeric_limits<double>::infinity();
	                    for (int k = 0; k < n; ++k) {
	                        const BoundingBox& bb = (k == i) ? cand_bb : bbs[static_cast<size_t>(k)];
	                        e2.min_x = std::min(e2.min_x, bb.min_x);
	                        e2.max_x = std::max(e2.max_x, bb.max_x);
	                        e2.min_y = std::min(e2.min_y, bb.min_y);
	                        e2.max_y = std::max(e2.max_y, bb.max_y);
	                    }
                        const double new_width = e2.max_x - e2.min_x;
                        const double new_height = e2.max_y - e2.min_y;
	                    double new_side = std::max(new_width, new_height);
                        const double new_min_dim = std::min(new_width, new_height);
	                    double new_cost = cost_from(new_width, new_height, new_overlap);
	                    if (p.overlap_cost_cap > 0.0 && new_cost > p.overlap_cost_cap) {
	                        continue;
	                    }
	                    if (new_cost + 1e-15 < best_new_cost) {
	                        found = true;
	                        best_new_cost = new_cost;
	                        best_new_side = new_side;
                            best_new_min_dim = new_min_dim;
	                        best_new_overlap = new_overlap;
	                        best_cand = cand;
	                        best_poly = std::move(cand_poly);
	                        best_bb = cand_bb;
	                        best_extents = e2;
	                    }
	                }

	                if (!found) {
	                    add_reward(op, false, false, false, curr_cost, curr_cost);
	                    maybe_update_controller(t);
	                    continue;
	                }

	                reward_old_cost = old_cost;
	                reward_new_cost = best_new_cost;
		                accepted = accept_move(old_cost, best_new_cost, T);
		                if (accepted) {
		                    improved_curr = (best_new_cost + 1e-15 < old_cost);
		                    const bool valid = (best_new_overlap <= p.overlap_eps_area);
		                    improved_best = valid && better_than_best(best_new_side, best_new_min_dim);

                            grid.update_position(i, best_cand.x, best_cand.y);
		                    poses[static_cast<size_t>(i)] = best_cand;
		                    polys[static_cast<size_t>(i)] = std::move(best_poly);
		                    bbs[static_cast<size_t>(i)] = best_bb;
	                    gmnx = best_extents.min_x;
	                    gmxx = best_extents.max_x;
	                    gmny = best_extents.min_y;
	                    gmxy = best_extents.max_y;
	                    curr_side = best_new_side;
	                    curr_overlap = best_new_overlap;
	                    curr_cost = best_new_cost;

	                    if (improved_best) {
	                        best.best_side = curr_side;
                            best_min_dim = best_new_min_dim;
	                        best.best_poses = poses;
	                    }
	                } else {
	                    bbs[static_cast<size_t>(i)] = old_bb;
	                    gmnx = old_gmnx;
	                    gmxx = old_gmxx;
	                    gmny = old_gmny;
	                    gmxy = old_gmxy;
	                    curr_side = old_side;
	                    curr_overlap = old_overlap;
	                    curr_cost = old_cost;
	                }
		            } else if (op == kBlockTranslate) {
                if (n <= 1) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }
                int anchor = pick_index(1.0).idx;
                std::vector<int> block = build_block(anchor, p.block_size);
                std::vector<char> in_block(static_cast<size_t>(n), 0);
                for (int idx : block) {
                    in_block[static_cast<size_t>(idx)] = 1;
                }

                double block_step =
                    (p.block_step_frac_max * (1.0 - frac) + p.block_step_frac_min * frac) *
                    std::max(1e-9, curr_side);
                block_step *= step_mult;
                double dx = 0.0;
                double dy = 0.0;
                if (uni(rng) < p.block_p_random_dir) {
                    dx = normal(rng) * block_step;
                    dy = normal(rng) * block_step;
                } else {
                    const double width = gmxx - gmnx;
                    const double height = gmxy - gmny;
                    const auto& bb = bbs[static_cast<size_t>(anchor)];
                    const double boundary_tol = 1e-9;
                    double mag = std::abs(normal(rng)) * block_step;
                    if (width >= height) {
                        bool left = (bb.min_x <= gmnx + boundary_tol);
                        bool right = (bb.max_x >= gmxx - boundary_tol);
                        double dir = left ? 1.0 : (right ? -1.0 : (cx - poses[static_cast<size_t>(anchor)].x >= 0.0 ? 1.0 : -1.0));
                        dx = dir * mag;
                        dy = normal(rng) * (0.35 * block_step);
                    } else {
                        bool bottom = (bb.min_y <= gmny + boundary_tol);
                        bool top = (bb.max_y >= gmxy - boundary_tol);
                        double dir = bottom ? 1.0 : (top ? -1.0 : (cy - poses[static_cast<size_t>(anchor)].y >= 0.0 ? 1.0 : -1.0));
                        dx = normal(rng) * (0.35 * block_step);
                        dy = dir * mag;
                    }
                }

	                std::vector<TreePose> moved_pose;
	                std::vector<Polygon> moved_poly;
	                std::vector<BoundingBox> moved_bb;
	                moved_pose.reserve(block.size());
	                moved_poly.reserve(block.size());
	                moved_bb.reserve(block.size());

	                bool ok = false;
	                double scale = 1.0;
	                for (int bt = 0; bt < 6; ++bt) {
	                    moved_pose.clear();
	                    moved_poly.clear();
	                    moved_bb.clear();

	                    bool ok_local = true;
		                    for (int idx : block) {
		                        TreePose cand = poses[static_cast<size_t>(idx)];
		                        cand.x += dx * scale;
		                        cand.y += dy * scale;
		                        quantize_pose_inplace(cand);
		                        if (cand.x < -100.0 || cand.x > 100.0 || cand.y < -100.0 ||
		                            cand.y > 100.0) {
		                            ok_local = false;
		                            break;
	                        }
	                        Polygon poly = transform_polygon(base_poly_, cand);
	                        BoundingBox bb = bounding_box(poly);
	                        moved_pose.push_back(cand);
	                        moved_poly.push_back(std::move(poly));
	                        moved_bb.push_back(bb);
	                    }

		                    if (ok_local && !soft_overlap) {
		                        for (size_t bi = 0; bi < block.size() && ok_local; ++bi) {
		                            const auto& cand = moved_pose[bi];
		                            const auto& cand_bb = moved_bb[bi];
		                            const auto& cand_poly = moved_poly[bi];
                                    grid.gather(cand.x, cand.y, neigh);
                                    for (int j : neigh) {
                                        if (in_block[static_cast<size_t>(j)]) {
                                            continue;
                                        }
                                        double ddx = cand.x - poses[static_cast<size_t>(j)].x;
                                        double ddy = cand.y - poses[static_cast<size_t>(j)].y;
                                        if (ddx * ddx + ddy * ddy > thr_sq) {
                                            continue;
                                        }
                                        if (!aabb_overlap(cand_bb, bbs[static_cast<size_t>(j)])) {
                                            continue;
                                        }
                                        if (polygons_intersect(cand_poly,
                                                               polys[static_cast<size_t>(j)])) {
                                            ok_local = false;
                                            break;
                                        }
                                    }
	                        }
	                    }

	                    if (ok_local) {
	                        ok = true;
	                        break;
	                    }

	                    scale *= 0.5;
	                }

		                if (!ok) {
		                    add_reward(op, false, false, false, curr_cost, curr_cost);
		                    maybe_update_controller(t);
		                    continue;
		                }

	                Extents e2 = compute_extents_mixed(bbs, block, moved_bb);
	                double new_side = side_from_extents(e2);
	                const double old_side = curr_side;
	                const double old_overlap = curr_overlap;
	                const double old_cost = curr_cost;

	                double delta_overlap = 0.0;
	                if (soft_overlap) {
	                    std::vector<int> pos(static_cast<size_t>(n), -1);
	                    for (size_t bi = 0; bi < block.size(); ++bi) {
	                        pos[static_cast<size_t>(block[bi])] = static_cast<int>(bi);
	                    }

	                    auto overlap_old = [&](int a, int b) -> double {
	                        const double dx = poses[static_cast<size_t>(a)].x - poses[static_cast<size_t>(b)].x;
	                        const double dy = poses[static_cast<size_t>(a)].y - poses[static_cast<size_t>(b)].y;
	                        if (dx * dx + dy * dy > thr_sq) {
	                            return 0.0;
	                        }
	                        if (!aabb_overlap(bbs[static_cast<size_t>(a)], bbs[static_cast<size_t>(b)]) ||
	                            !polygons_intersect(polys[static_cast<size_t>(a)],
	                                               polys[static_cast<size_t>(b)])) {
	                            return 0.0;
	                        }
	                        return clamp_overlap(
	                            overlap_metric(poses[static_cast<size_t>(a)],
	                                           poses[static_cast<size_t>(b)]));
	                    };
	                    auto overlap_new = [&](int a, int b) -> double {
	                        const int pa = pos[static_cast<size_t>(a)];
	                        const int pb = pos[static_cast<size_t>(b)];
	                        const TreePose& ta =
	                            (pa >= 0) ? moved_pose[static_cast<size_t>(pa)] : poses[static_cast<size_t>(a)];
	                        const TreePose& tb =
	                            (pb >= 0) ? moved_pose[static_cast<size_t>(pb)] : poses[static_cast<size_t>(b)];
	                        const double dx = ta.x - tb.x;
	                        const double dy = ta.y - tb.y;
	                        if (dx * dx + dy * dy > thr_sq) {
	                            return 0.0;
	                        }
	                        const BoundingBox& ba =
	                            (pa >= 0) ? moved_bb[static_cast<size_t>(pa)] : bbs[static_cast<size_t>(a)];
	                        const BoundingBox& bb =
	                            (pb >= 0) ? moved_bb[static_cast<size_t>(pb)] : bbs[static_cast<size_t>(b)];
	                        const Polygon& pa_poly =
	                            (pa >= 0) ? moved_poly[static_cast<size_t>(pa)] : polys[static_cast<size_t>(a)];
	                        const Polygon& pb_poly =
	                            (pb >= 0) ? moved_poly[static_cast<size_t>(pb)] : polys[static_cast<size_t>(b)];
	                        if (!aabb_overlap(ba, bb) || !polygons_intersect(pa_poly, pb_poly)) {
	                            return 0.0;
	                        }
	                        return clamp_overlap(overlap_metric(ta, tb));
	                    };

	                    for (int a : block) {
	                        for (int b = 0; b < n; ++b) {
	                            if (b == a) {
	                                continue;
	                            }
	                            if (pos[static_cast<size_t>(b)] >= 0 && b < a) {
	                                continue;
	                            }
	                            delta_overlap += overlap_new(a, b) - overlap_old(a, b);
	                        }
	                    }
	                }

	                double new_overlap = std::max(0.0, old_overlap + delta_overlap);
	                new_overlap = clamp_overlap(new_overlap);
                    const double new_width = e2.max_x - e2.min_x;
                    const double new_height = e2.max_y - e2.min_y;
                    const double new_min_dim = std::min(new_width, new_height);
	                double new_cost = cost_from(new_width, new_height, new_overlap);
	                if (p.overlap_cost_cap > 0.0 && new_cost > p.overlap_cost_cap) {
	                    new_cost = std::numeric_limits<double>::infinity();
	                }

	                reward_old_cost = old_cost;
	                reward_new_cost = new_cost;
	                accepted = accept_move(old_cost, new_cost, T);
	                if (accepted) {
		                    improved_curr = (new_cost + 1e-15 < old_cost);
		                    const bool valid = (new_overlap <= p.overlap_eps_area);
		                    improved_best = valid && better_than_best(new_side, new_min_dim);
		                    for (size_t bi = 0; bi < block.size(); ++bi) {
		                        int idx = block[bi];
                                grid.update_position(idx,
                                                     moved_pose[bi].x,
                                                     moved_pose[bi].y);
		                        poses[static_cast<size_t>(idx)] = moved_pose[bi];
		                        polys[static_cast<size_t>(idx)] = std::move(moved_poly[bi]);
		                        bbs[static_cast<size_t>(idx)] = moved_bb[bi];
		                    }
	                    gmnx = e2.min_x;
	                    gmxx = e2.max_x;
	                    gmny = e2.min_y;
	                    gmxy = e2.max_y;
	                    curr_side = new_side;
	                    curr_overlap = new_overlap;
	                    curr_cost = new_cost;
	                    if (improved_best) {
	                        best.best_side = curr_side;
                            best_min_dim = new_min_dim;
	                        best.best_poses = poses;
	                    }
	                }
            } else if (op == kBlockRotate) {
                if (n <= 1) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }
                int anchor = pick_index(1.0).idx;
                std::vector<int> block = build_block(anchor, p.block_size);
                std::vector<char> in_block(static_cast<size_t>(n), 0);
                for (int idx : block) {
                    in_block[static_cast<size_t>(idx)] = 1;
                }

                double px = 0.0;
                double py = 0.0;
                for (int idx : block) {
                    px += poses[static_cast<size_t>(idx)].x;
                    py += poses[static_cast<size_t>(idx)].y;
                }
                px /= static_cast<double>(block.size());
                py /= static_cast<double>(block.size());

                double rot_rng =
                    p.block_rot_deg_max * (1.0 - frac) + p.block_rot_deg_min * frac;
                rot_rng = std::min(180.0, rot_rng * step_mult);
                const double ang = uni_deg(rng) * rot_rng;
                const double rad = ang * 3.14159265358979323846 / 180.0;
                const double cA = std::cos(rad);
                const double sA = std::sin(rad);

                std::vector<TreePose> moved_pose;
                std::vector<Polygon> moved_poly;
                std::vector<BoundingBox> moved_bb;
                moved_pose.reserve(block.size());
                moved_poly.reserve(block.size());
                moved_bb.reserve(block.size());

                bool ok = true;
                for (int idx : block) {
                    const auto& src = poses[static_cast<size_t>(idx)];
                    double dx0 = src.x - px;
                    double dy0 = src.y - py;
	                    TreePose cand = src;
	                    cand.x = px + cA * dx0 - sA * dy0;
	                    cand.y = py + sA * dx0 + cA * dy0;
	                    cand.deg = wrap_deg(cand.deg + ang);
	                    quantize_pose_inplace(cand);
	                    if (cand.x < -100.0 || cand.x > 100.0 || cand.y < -100.0 || cand.y > 100.0) {
	                        ok = false;
	                        break;
	                    }
                    Polygon poly = transform_polygon(base_poly_, cand);
                    BoundingBox bb = bounding_box(poly);
                    moved_pose.push_back(cand);
                    moved_poly.push_back(std::move(poly));
                    moved_bb.push_back(bb);
                }
                if (!ok) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

                if (!soft_overlap) {
                    for (size_t bi = 0; bi < block.size() && ok; ++bi) {
                        const auto& cand = moved_pose[bi];
                        const auto& cand_bb = moved_bb[bi];
                        const auto& cand_poly = moved_poly[bi];
                        grid.gather(cand.x, cand.y, neigh);
                        for (int j : neigh) {
                            if (in_block[static_cast<size_t>(j)]) {
                                continue;
                            }
                            double ddx = cand.x - poses[static_cast<size_t>(j)].x;
                            double ddy = cand.y - poses[static_cast<size_t>(j)].y;
                            if (ddx * ddx + ddy * ddy > thr_sq) {
                                continue;
                            }
                            if (!aabb_overlap(cand_bb, bbs[static_cast<size_t>(j)])) {
                                continue;
                            }
                            if (polygons_intersect(cand_poly, polys[static_cast<size_t>(j)])) {
                                ok = false;
                                break;
                            }
                        }
                        if (!ok) {
                            break;
                        }
                    }
                    if (!ok) {
                        add_reward(op, false, false, false, curr_cost, curr_cost);
                        maybe_update_controller(t);
                        continue;
                    }
                }

                Extents e2 = compute_extents_mixed(bbs, block, moved_bb);
                double new_side = side_from_extents(e2);

                const double old_side = curr_side;
                const double old_overlap = curr_overlap;
                const double old_cost = curr_cost;

                double delta_overlap = 0.0;
                if (soft_overlap) {
                    std::vector<int> pos(static_cast<size_t>(n), -1);
                    for (size_t bi = 0; bi < block.size(); ++bi) {
                        pos[static_cast<size_t>(block[bi])] = static_cast<int>(bi);
                    }

                    auto overlap_old = [&](int a, int b) -> double {
                        const double dx =
                            poses[static_cast<size_t>(a)].x - poses[static_cast<size_t>(b)].x;
                        const double dy =
                            poses[static_cast<size_t>(a)].y - poses[static_cast<size_t>(b)].y;
                        if (dx * dx + dy * dy > thr_sq) {
                            return 0.0;
                        }
                        if (!aabb_overlap(bbs[static_cast<size_t>(a)],
                                          bbs[static_cast<size_t>(b)]) ||
                            !polygons_intersect(polys[static_cast<size_t>(a)],
                                               polys[static_cast<size_t>(b)])) {
                            return 0.0;
                        }
                        return clamp_overlap(
                            overlap_metric(poses[static_cast<size_t>(a)],
                                           poses[static_cast<size_t>(b)]));
                    };
                    auto overlap_new = [&](int a, int b) -> double {
                        const int pa = pos[static_cast<size_t>(a)];
                        const int pb = pos[static_cast<size_t>(b)];
                        const TreePose& ta =
                            (pa >= 0) ? moved_pose[static_cast<size_t>(pa)]
                                      : poses[static_cast<size_t>(a)];
                        const TreePose& tb =
                            (pb >= 0) ? moved_pose[static_cast<size_t>(pb)]
                                      : poses[static_cast<size_t>(b)];
                        const double dx = ta.x - tb.x;
                        const double dy = ta.y - tb.y;
                        if (dx * dx + dy * dy > thr_sq) {
                            return 0.0;
                        }
                        const BoundingBox& ba =
                            (pa >= 0) ? moved_bb[static_cast<size_t>(pa)]
                                      : bbs[static_cast<size_t>(a)];
                        const BoundingBox& bb =
                            (pb >= 0) ? moved_bb[static_cast<size_t>(pb)]
                                      : bbs[static_cast<size_t>(b)];
                        const Polygon& pa_poly =
                            (pa >= 0) ? moved_poly[static_cast<size_t>(pa)]
                                      : polys[static_cast<size_t>(a)];
                        const Polygon& pb_poly =
                            (pb >= 0) ? moved_poly[static_cast<size_t>(pb)]
                                      : polys[static_cast<size_t>(b)];
                        if (!aabb_overlap(ba, bb) || !polygons_intersect(pa_poly, pb_poly)) {
                            return 0.0;
                        }
                        return clamp_overlap(overlap_metric(ta, tb));
                    };

                    for (int a : block) {
                        for (int b = 0; b < n; ++b) {
                            if (b == a) {
                                continue;
                            }
                            if (pos[static_cast<size_t>(b)] >= 0 && b < a) {
                                continue;
                            }
                            delta_overlap += overlap_new(a, b) - overlap_old(a, b);
                        }
                    }
                }

                double new_overlap = std::max(0.0, old_overlap + delta_overlap);
                new_overlap = clamp_overlap(new_overlap);
                const double new_width = e2.max_x - e2.min_x;
                const double new_height = e2.max_y - e2.min_y;
                const double new_min_dim = std::min(new_width, new_height);
                double new_cost = cost_from(new_width, new_height, new_overlap);
                if (p.overlap_cost_cap > 0.0 && new_cost > p.overlap_cost_cap) {
                    new_cost = std::numeric_limits<double>::infinity();
                }

                reward_old_cost = old_cost;
                reward_new_cost = new_cost;
                accepted = accept_move(old_cost, new_cost, T);
	                if (accepted) {
	                    improved_curr = (new_cost + 1e-15 < old_cost);
	                    const bool valid = (new_overlap <= p.overlap_eps_area);
	                    improved_best = valid && better_than_best(new_side, new_min_dim);
	                    for (size_t bi = 0; bi < block.size(); ++bi) {
	                        int idx = block[bi];
                            grid.update_position(idx,
                                                 moved_pose[bi].x,
                                                 moved_pose[bi].y);
	                        poses[static_cast<size_t>(idx)] = moved_pose[bi];
	                        polys[static_cast<size_t>(idx)] = std::move(moved_poly[bi]);
	                        bbs[static_cast<size_t>(idx)] = moved_bb[bi];
	                    }
                    gmnx = e2.min_x;
                    gmxx = e2.max_x;
                    gmny = e2.min_y;
                    gmxy = e2.max_y;
                    curr_side = new_side;
	                    curr_overlap = new_overlap;
	                    curr_cost = new_cost;
	                    if (improved_best) {
	                        best.best_side = curr_side;
	                        best_min_dim = new_min_dim;
	                        best.best_poses = poses;
	                    }
	                } else {
	                    curr_side = old_side;
	                    curr_overlap = old_overlap;
                    curr_cost = old_cost;
                }
            } else if (op == kLNS) {
	                if (n <= 2 || p.lns_remove <= 0) {
	                    add_reward(op, false, false, false, curr_cost, curr_cost);
	                    maybe_update_controller(t);
	                    continue;
	                }

                const double boundary_tol = 1e-9;
                std::vector<int> boundary;
                boundary.reserve(static_cast<size_t>(n));
                for (int i = 0; i < n; ++i) {
                    if (!is_active(i)) {
                        continue;
                    }
                    const auto& bb = bbs[static_cast<size_t>(i)];
                    if (bb.min_x <= gmnx + boundary_tol || bb.max_x >= gmxx - boundary_tol ||
                        bb.min_y <= gmny + boundary_tol || bb.max_y >= gmxy - boundary_tol) {
                        boundary.push_back(i);
                    }
                }
                if (boundary.empty()) {
                    boundary = boundary_pool;
                }
                if (boundary.empty()) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

	                int m = std::min(std::max(1, p.lns_remove), n - 1);
	                if (static_cast<int>(boundary.size()) < m) {
	                    m = static_cast<int>(boundary.size());
	                }

	                auto pick_lns_remove_set = [&](const std::vector<int>& boundary_in,
	                                               int want_remove) -> std::vector<int> {
	                    if (static_cast<int>(boundary_in.size()) <= want_remove) {
	                        return boundary_in;
	                    }
	                    if (want_remove <= 0) {
	                        return {};
	                    }

	                    const double tol = 1e-9;
	                    auto two_smallest = [&](auto get) -> std::tuple<double, double, int> {
	                        double m1 = std::numeric_limits<double>::infinity();
	                        double m2 = std::numeric_limits<double>::infinity();
	                        int c1 = 0;
	                        for (const auto& bb : bbs) {
	                            double v = get(bb);
	                            if (v < m1 - tol) {
	                                m2 = m1;
	                                m1 = v;
	                                c1 = 1;
	                            } else if (std::abs(v - m1) <= tol) {
	                                c1 += 1;
	                            } else if (v < m2 - tol) {
	                                m2 = v;
	                            }
	                        }
	                        if (!std::isfinite(m2)) {
	                            m2 = m1;
	                        }
	                        return {m1, m2, c1};
	                    };
	                    auto two_largest = [&](auto get) -> std::tuple<double, double, int> {
	                        double m1 = -std::numeric_limits<double>::infinity();
	                        double m2 = -std::numeric_limits<double>::infinity();
	                        int c1 = 0;
	                        for (const auto& bb : bbs) {
	                            double v = get(bb);
	                            if (v > m1 + tol) {
	                                m2 = m1;
	                                m1 = v;
	                                c1 = 1;
	                            } else if (std::abs(v - m1) <= tol) {
	                                c1 += 1;
	                            } else if (v > m2 + tol) {
	                                m2 = v;
	                            }
	                        }
	                        if (!std::isfinite(m2)) {
	                            m2 = m1;
	                        }
	                        return {m1, m2, c1};
	                    };

	                    auto [minx1, minx2, minx_c] =
	                        two_smallest([&](const BoundingBox& bb) { return bb.min_x; });
	                    auto [maxx1, maxx2, maxx_c] =
	                        two_largest([&](const BoundingBox& bb) { return bb.max_x; });
	                    auto [miny1, miny2, miny_c] =
	                        two_smallest([&](const BoundingBox& bb) { return bb.min_y; });
	                    auto [maxy1, maxy2, maxy_c] =
	                        two_largest([&](const BoundingBox& bb) { return bb.max_y; });

	                    struct ScoredIdx {
	                        double score;
	                        int idx;
	                    };
	                    std::vector<ScoredIdx> scored;
	                    scored.reserve(boundary_in.size());
	                    double best_gain = 0.0;
	                    for (int idx : boundary_in) {
	                        const auto& bb = bbs[static_cast<size_t>(idx)];
	                        double new_min_x = minx1;
	                        if (minx_c == 1 && std::abs(bb.min_x - minx1) <= tol) {
	                            new_min_x = minx2;
	                        }
	                        double new_max_x = maxx1;
	                        if (maxx_c == 1 && std::abs(bb.max_x - maxx1) <= tol) {
	                            new_max_x = maxx2;
	                        }
	                        double new_min_y = miny1;
	                        if (miny_c == 1 && std::abs(bb.min_y - miny1) <= tol) {
	                            new_min_y = miny2;
	                        }
	                        double new_max_y = maxy1;
	                        if (maxy_c == 1 && std::abs(bb.max_y - maxy1) <= tol) {
	                            new_max_y = maxy2;
	                        }

	                        double new_width = new_max_x - new_min_x;
	                        double new_height = new_max_y - new_min_y;
	                        double new_side = std::max(new_width, new_height);
	                        double base_gain = curr_side - new_side;
	                        if (base_gain > best_gain) {
	                            best_gain = base_gain;
	                        }
	                        double gain = (base_gain > 0.0) ? (base_gain + uni(rng) * 1e-12) : 0.0;
	                        scored.push_back(ScoredIdx{gain, idx});
	                    }

	                    if (!(best_gain > 0.0)) {
	                        std::vector<int> out = boundary_in;
	                        std::shuffle(out.begin(), out.end(), rng);
	                        out.resize(static_cast<size_t>(want_remove));
	                        return out;
	                    }

	                    std::sort(scored.begin(), scored.end(), [](const ScoredIdx& a, const ScoredIdx& b) {
	                        if (a.score != b.score) {
	                            return a.score > b.score;
	                        }
	                        return a.idx < b.idx;
	                    });

	                    const double width0 = gmxx - gmnx;
	                    const double height0 = gmxy - gmny;
	                    const bool dom_x = (width0 >= height0);

	                    auto is_left = [&](int idx) -> bool {
	                        return bbs[static_cast<size_t>(idx)].min_x <= minx1 + tol;
	                    };
	                    auto is_right = [&](int idx) -> bool {
	                        return bbs[static_cast<size_t>(idx)].max_x >= maxx1 - tol;
	                    };
	                    auto is_bottom = [&](int idx) -> bool {
	                        return bbs[static_cast<size_t>(idx)].min_y <= miny1 + tol;
	                    };
	                    auto is_top = [&](int idx) -> bool {
	                        return bbs[static_cast<size_t>(idx)].max_y >= maxy1 - tol;
	                    };

	                    std::vector<char> used(static_cast<size_t>(n), 0);
	                    std::vector<int> out;
	                    out.reserve(static_cast<size_t>(want_remove));

	                    auto take_best = [&](auto pred) {
	                        if (static_cast<int>(out.size()) >= want_remove) {
	                            return;
	                        }
	                        for (const auto& si : scored) {
	                            int idx = si.idx;
	                            if (used[static_cast<size_t>(idx)]) {
	                                continue;
	                            }
	                            if (!pred(idx)) {
	                                continue;
	                            }
	                            used[static_cast<size_t>(idx)] = 1;
	                            out.push_back(idx);
	                            break;
	                        }
	                    };

	                    // Garante que o LNS mexa em ambos os lados do eixo dominante (casca).
	                    if (dom_x) {
	                        take_best(is_left);
	                        take_best(is_right);
	                    } else {
	                        take_best(is_bottom);
	                        take_best(is_top);
	                    }

	                    for (const auto& si : scored) {
	                        if (static_cast<int>(out.size()) >= want_remove) {
	                            break;
	                        }
	                        int idx = si.idx;
	                        if (used[static_cast<size_t>(idx)]) {
	                            continue;
	                        }
	                        used[static_cast<size_t>(idx)] = 1;
	                        out.push_back(idx);
	                    }

	                    if (static_cast<int>(out.size()) > want_remove) {
	                        out.resize(static_cast<size_t>(want_remove));
	                    }
	                    return out;
	                };

	                boundary = pick_lns_remove_set(boundary, m);

	                std::vector<char> active(static_cast<size_t>(n), 1);
	                for (int idx : boundary) {
	                    active[static_cast<size_t>(idx)] = 0;
                }

                double min_x = std::numeric_limits<double>::infinity();
                double max_x = -std::numeric_limits<double>::infinity();
                double min_y = std::numeric_limits<double>::infinity();
                double max_y = -std::numeric_limits<double>::infinity();
                for (int i = 0; i < n; ++i) {
                    if (!active[static_cast<size_t>(i)]) {
                        continue;
                    }
                    const auto& bb = bbs[static_cast<size_t>(i)];
                    min_x = std::min(min_x, bb.min_x);
                    max_x = std::max(max_x, bb.max_x);
                    min_y = std::min(min_y, bb.min_y);
                    max_y = std::max(max_y, bb.max_y);
                }
                if (!std::isfinite(min_x)) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

                double box_side =
                    std::max(max_x - min_x, max_y - min_y) * std::max(1e-6, p.lns_box_mult);
                const double ccx = 0.5 * (min_x + max_x);
                const double ccy = 0.5 * (min_y + max_y);
                const double half = 0.5 * box_side;
                const double noise =
                    p.lns_noise_frac * std::max(1e-9, curr_side) * step_mult;

                std::vector<TreePose> cand_poses = poses;
                std::vector<Polygon> cand_polys = polys;
                std::vector<BoundingBox> cand_bbs = bbs;

	                double active_overlap = 0.0;
	                if (soft_overlap) {
	                    for (int i = 0; i < n; ++i) {
	                        if (!active[static_cast<size_t>(i)]) {
	                            continue;
	                        }
                        for (int j = i + 1; j < n; ++j) {
                            if (!active[static_cast<size_t>(j)]) {
                                continue;
                            }
                            double dx = cand_poses[static_cast<size_t>(i)].x -
                                        cand_poses[static_cast<size_t>(j)].x;
                            double dy = cand_poses[static_cast<size_t>(i)].y -
                                        cand_poses[static_cast<size_t>(j)].y;
                            if (dx * dx + dy * dy > thr_sq) {
                                continue;
                            }
                            if (!aabb_overlap(cand_bbs[static_cast<size_t>(i)],
                                              cand_bbs[static_cast<size_t>(j)])) {
                                continue;
                            }
                            if (!polygons_intersect(cand_polys[static_cast<size_t>(i)],
                                                   cand_polys[static_cast<size_t>(j)])) {
                                continue;
                            }
                            active_overlap += clamp_overlap(
                                overlap_metric(cand_poses[static_cast<size_t>(i)],
                                               cand_poses[static_cast<size_t>(j)]));
                        }
                    }
	                    active_overlap = clamp_overlap(active_overlap);
	                }

                    UniformGrid lns_grid(n, thr);
                    for (int j = 0; j < n; ++j) {
                        if (!active[static_cast<size_t>(j)]) {
                            continue;
                        }
                        lns_grid.insert(j,
                                        cand_poses[static_cast<size_t>(j)].x,
                                        cand_poses[static_cast<size_t>(j)].y);
                    }

	                std::shuffle(boundary.begin(), boundary.end(), rng);
	                bool ok = true;
	                for (int idx : boundary) {
                    TreePose best_pose;
                    Polygon best_poly;
                    BoundingBox best_bb;
                    double best_metric = std::numeric_limits<double>::infinity();
                    double best_side = std::numeric_limits<double>::infinity();
                    double best_overlap_add = 0.0;
                    bool found = false;

                    for (int attempt = 0; attempt < std::max(1, p.lns_attempts_per_tree); ++attempt) {
	                        TreePose cand = poses[static_cast<size_t>(idx)];
	                        const double mode = uni(rng);

	                        const double width0 = gmxx - gmnx;
	                        const double height0 = gmxy - gmny;
	                        const bool dom_x = (width0 >= height0);
	                        const double boundary_tol = 1e-9;
	                        const auto& bb0 = bbs[static_cast<size_t>(idx)];

	                        double pull = p.lns_pull_min +
	                                      (p.lns_pull_max - p.lns_pull_min) * uni(rng);
	                        if (mode < p.lns_p_uniform) {
	                            // Uniform, mas com leve viés pra dentro (evita jogar a árvore de volta na casca).
	                            double ux = (2.0 * uni(rng) - 1.0);
	                            double uy = (2.0 * uni(rng) - 1.0);
	                            cand.x = ccx + ux * (0.85 * half) + normal(rng) * (0.25 * noise);
	                            cand.y = ccy + uy * (0.85 * half) + normal(rng) * (0.25 * noise);
	                        } else if (mode < p.lns_p_uniform + p.lns_p_contact) {
	                            // Contact: tenta ancorar em uma árvore ativa, mas orienta o ângulo pra "dentro".
	                            int other = -1;
	                            std::uniform_int_distribution<int> pick_any(0, n - 1);
	                            for (int tries = 0; tries < 12; ++tries) {
	                                int j = pick_any(rng);
	                                if (j == idx) {
	                                    continue;
	                                }
	                                if (!active[static_cast<size_t>(j)]) {
	                                    continue;
	                                }
	                                other = j;
	                                break;
	                            }
	                            if (other < 0) {
	                                cand.x += pull * (ccx - cand.x) + normal(rng) * noise;
	                                cand.y += pull * (ccy - cand.y) + normal(rng) * noise;
	                            } else {
	                                const double dist = thr * (0.88 + 0.14 * uni(rng));
	                                double base_ang = std::atan2(ccy - cand_poses[static_cast<size_t>(other)].y,
	                                                             ccx - cand_poses[static_cast<size_t>(other)].x);
	                                double jitter = (uni(rng) * 2.0 - 1.0) * (35.0 * 3.14159265358979323846 / 180.0);
	                                double ang = base_ang + jitter;
	                                cand.x = cand_poses[static_cast<size_t>(other)].x +
	                                         dist * std::cos(ang) + normal(rng) * (0.15 * noise);
	                                cand.y = cand_poses[static_cast<size_t>(other)].y +
	                                         dist * std::sin(ang) + normal(rng) * (0.15 * noise);
	                            }
	                        } else {
	                            // Pull: como relocate, mas se idx estava na casca, puxa mais no eixo dominante.
	                            cand.x += pull * (ccx - cand.x) + normal(rng) * noise;
	                            cand.y += pull * (ccy - cand.y) + normal(rng) * noise;
	                            if (dom_x) {
	                                if (bb0.min_x <= gmnx + boundary_tol) {
	                                    cand.x += std::abs(normal(rng)) * (0.25 * half);
	                                } else if (bb0.max_x >= gmxx - boundary_tol) {
	                                    cand.x -= std::abs(normal(rng)) * (0.25 * half);
	                                }
	                            } else {
	                                if (bb0.min_y <= gmny + boundary_tol) {
	                                    cand.y += std::abs(normal(rng)) * (0.25 * half);
	                                } else if (bb0.max_y >= gmxy - boundary_tol) {
	                                    cand.y -= std::abs(normal(rng)) * (0.25 * half);
	                                }
	                            }
	                        }
	                        if (uni(rng) < p.lns_p_rot) {
	                            cand.deg = wrap_deg(cand.deg + uni_deg(rng) * ddeg_rng);
	                        }
	                        quantize_pose_inplace(cand);

	                        if (cand.x < -100.0 || cand.x > 100.0 || cand.y < -100.0 || cand.y > 100.0) {
	                            continue;
	                        }

                        Polygon poly = transform_polygon(base_poly_, cand);
                        BoundingBox bb = bounding_box(poly);

                        double overlap_add = 0.0;
                        bool collide = false;
                        lns_grid.gather(cand.x, cand.y, neigh);
                        for (int j : neigh) {
                            if (j == idx) {
                                continue;
                            }
                            if (!active[static_cast<size_t>(j)]) {
                                continue;
                            }
                            double ddx = cand.x - cand_poses[static_cast<size_t>(j)].x;
                            double ddy = cand.y - cand_poses[static_cast<size_t>(j)].y;
                            if (ddx * ddx + ddy * ddy > thr_sq) {
                                continue;
                            }
                            if (!aabb_overlap(bb, cand_bbs[static_cast<size_t>(j)])) {
                                continue;
                            }
                            if (polygons_intersect(poly, cand_polys[static_cast<size_t>(j)])) {
                                if (!soft_overlap) {
                                    collide = true;
                                    break;
                                }
                                overlap_add += clamp_overlap(
                                    overlap_metric(cand, cand_poses[static_cast<size_t>(j)]));
                            }
                        }
                        if (collide) {
                            continue;
                        }

                        double nmin_x = std::min(min_x, bb.min_x);
                        double nmax_x = std::max(max_x, bb.max_x);
                        double nmin_y = std::min(min_y, bb.min_y);
                        double nmax_y = std::max(max_y, bb.max_y);
	                        double cand_width = nmax_x - nmin_x;
	                        double cand_height = nmax_y - nmin_y;
	                        double side = std::max(cand_width, cand_height);
	                        double total_overlap =
	                            soft_overlap ? clamp_overlap(active_overlap + overlap_add) : 0.0;
	                        double metric = cost_from(cand_width, cand_height, total_overlap);
	                        if (soft_overlap && p.overlap_cost_cap > 0.0 &&
	                            metric > p.overlap_cost_cap) {
	                            continue;
	                        }
                        if (metric + 1e-15 < best_metric) {
                            best_metric = metric;
                            best_side = side;
                            best_overlap_add = overlap_add;
                            best_pose = cand;
                            best_poly = std::move(poly);
                            best_bb = bb;
                            found = true;
                        }
                    }

                    if (!found) {
                        ok = false;
                        break;
                    }

                    cand_poses[static_cast<size_t>(idx)] = best_pose;
                    cand_polys[static_cast<size_t>(idx)] = std::move(best_poly);
                    cand_bbs[static_cast<size_t>(idx)] = best_bb;
                    active[static_cast<size_t>(idx)] = 1;
                    lns_grid.insert(idx, best_pose.x, best_pose.y);
                    if (soft_overlap) {
                        active_overlap = clamp_overlap(active_overlap + best_overlap_add);
                    }
                    min_x = std::min(min_x, best_bb.min_x);
                    max_x = std::max(max_x, best_bb.max_x);
                    min_y = std::min(min_y, best_bb.min_y);
                    max_y = std::max(max_y, best_bb.max_y);
                }
                if (!ok) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

                    const double new_width = max_x - min_x;
                    const double new_height = max_y - min_y;
                    const double new_min_dim = std::min(new_width, new_height);
	                double new_side = std::max(new_width, new_height);
	                double new_overlap = soft_overlap ? active_overlap : 0.0;
	                new_overlap = clamp_overlap(new_overlap);
	                double new_cost = cost_from(new_width, new_height, new_overlap);
                if (p.overlap_cost_cap > 0.0 && new_cost > p.overlap_cost_cap) {
                    new_cost = std::numeric_limits<double>::infinity();
                }

                reward_old_cost = curr_cost;
                reward_new_cost = new_cost;
                accepted = accept_move(curr_cost, new_cost, T);
		                if (accepted) {
		                    improved_curr = (new_cost + 1e-15 < curr_cost);
		                    const bool valid = (new_overlap <= p.overlap_eps_area);
		                    improved_best = valid && better_than_best(new_side, new_min_dim);
		                    poses = std::move(cand_poses);
		                    polys = std::move(cand_polys);
		                    bbs = std::move(cand_bbs);
                        grid.rebuild(poses);
	                    gmnx = min_x;
	                    gmxx = max_x;
	                    gmny = min_y;
	                    gmxy = max_y;
	                    curr_side = new_side;
                    curr_overlap = new_overlap;
                    curr_cost = new_cost;
                    if (improved_best) {
                        best.best_side = curr_side;
                        best_min_dim = new_min_dim;
                        best.best_poses = poses;
                    }
                }
            } else if (op == kResolveOverlap) {
                if (!soft_overlap || n <= 1 || curr_overlap <= p.overlap_eps_area) {
                    add_reward(op, false, false, false, curr_cost, curr_cost);
                    maybe_update_controller(t);
                    continue;
                }

                bool moved = false;
                double step =
                    (p.resolve_step_frac_max * (1.0 - frac) + p.resolve_step_frac_min * frac) *
                    std::max(1e-9, curr_side);
                step *= step_mult;

                for (int att = 0; att < std::max(1, p.resolve_attempts) && !moved; ++att) {
                    int i = -1;
                    Point rep{0.0, 0.0};

                    for (int tries = 0; tries < 8; ++tries) {
                        int cand_i = pick_index(p.p_pick_extreme).idx;

                        Point best_rep{0.0, 0.0};
                        double best_score = 0.0;
                            grid.gather(poses[static_cast<size_t>(cand_i)].x,
                                        poses[static_cast<size_t>(cand_i)].y,
                                        neigh);
                            for (int j : neigh) {
                                if (j == cand_i) {
                                    continue;
                                }
                                double ddx = poses[static_cast<size_t>(cand_i)].x -
                                             poses[static_cast<size_t>(j)].x;
                                double ddy = poses[static_cast<size_t>(cand_i)].y -
                                             poses[static_cast<size_t>(j)].y;
                                if (ddx * ddx + ddy * ddy > thr_sq) {
                                    continue;
                                }
                                if (!aabb_overlap(bbs[static_cast<size_t>(cand_i)],
                                                  bbs[static_cast<size_t>(j)])) {
                                    continue;
                                }
                                if (!polygons_intersect(polys[static_cast<size_t>(cand_i)],
                                                        polys[static_cast<size_t>(j)])) {
                                    continue;
                                }

                                OverlapSeparation os =
                                    overlap_separation(poses[static_cast<size_t>(cand_i)],
                                                       poses[static_cast<size_t>(j)]);
                                double a = clamp_overlap(os.area);
                                if (!(a > 0.0)) {
                                    continue;
                                }

                                Point avg_mtv = mul_point(os.mtv_sum, 1.0 / a);
                                double metric = use_mtv_metric
                                                    ? (avg_mtv.x * avg_mtv.x + avg_mtv.y * avg_mtv.y)
                                                    : a;
                                metric = clamp_overlap(metric);
                                if (metric > best_score) {
                                    best_score = metric;
                                    best_rep = avg_mtv;
                                }
                            }

                        if (best_score > 0.0) {
                            i = cand_i;
                            rep = best_rep;
                            break;
                        }
                    }

                    if (i < 0) {
                        break;
                    }

                    double vnorm = std::hypot(rep.x, rep.y);
                    double dir_x = 0.0;
                    double dir_y = 0.0;
                    double move_len = step;
                    if (vnorm > 1e-12) {
                        dir_x = rep.x / vnorm;
                        dir_y = rep.y / vnorm;
                        move_len = std::min(step, vnorm);
                    } else {
                        double a = uni(rng) * 2.0 * 3.14159265358979323846;
                        dir_x = std::cos(a);
                        dir_y = std::sin(a);
                        move_len = step;
                    }

                    double dx = dir_x * move_len + normal(rng) * (p.resolve_noise_frac * move_len);
                    double dy = dir_y * move_len + normal(rng) * (p.resolve_noise_frac * move_len);

                    double scale = 1.0;
	                    for (int bt = 0; bt < 6 && !moved; ++bt) {
	                        TreePose cand = poses[static_cast<size_t>(i)];
	                        cand.x += dx * scale;
	                        cand.y += dy * scale;
	                        quantize_pose_inplace(cand);
	                        if (cand.x < -100.0 || cand.x > 100.0 || cand.y < -100.0 || cand.y > 100.0) {
	                            scale *= 0.5;
	                            continue;
	                        }

                        Polygon cand_poly = transform_polygon(base_poly_, cand);
                        BoundingBox cand_bb = bounding_box(cand_poly);

                        double delta_overlap = 0.0;
                        gather_union(i,
                                     -1,
                                     poses[static_cast<size_t>(i)].x,
                                     poses[static_cast<size_t>(i)].y,
                                     cand.x,
                                     cand.y,
                                     neigh_union);
                        for (int k : neigh_union) {
                            const double old_dx =
                                poses[static_cast<size_t>(i)].x - poses[static_cast<size_t>(k)].x;
                            const double old_dy =
                                poses[static_cast<size_t>(i)].y - poses[static_cast<size_t>(k)].y;
                            const double new_dx = cand.x - poses[static_cast<size_t>(k)].x;
                            const double new_dy = cand.y - poses[static_cast<size_t>(k)].y;

                            double old_a = 0.0;
                            if ((old_dx * old_dx + old_dy * old_dy <= thr_sq) &&
                                aabb_overlap(bbs[static_cast<size_t>(i)],
                                             bbs[static_cast<size_t>(k)]) &&
                                polygons_intersect(polys[static_cast<size_t>(i)],
                                                   polys[static_cast<size_t>(k)])) {
                                old_a = overlap_metric(poses[static_cast<size_t>(i)],
                                                       poses[static_cast<size_t>(k)]);
                            }
                            double new_a = 0.0;
                            if ((new_dx * new_dx + new_dy * new_dy <= thr_sq) &&
                                aabb_overlap(cand_bb, bbs[static_cast<size_t>(k)]) &&
                                polygons_intersect(cand_poly, polys[static_cast<size_t>(k)])) {
                                new_a = overlap_metric(cand, poses[static_cast<size_t>(k)]);
                            }
                            delta_overlap += clamp_overlap(new_a) - clamp_overlap(old_a);
                        }

                        double new_overlap = std::max(0.0, curr_overlap + delta_overlap);
                        new_overlap = clamp_overlap(new_overlap);

                        Extents e2;
                        e2.min_x = std::numeric_limits<double>::infinity();
                        e2.max_x = -std::numeric_limits<double>::infinity();
                        e2.min_y = std::numeric_limits<double>::infinity();
                        e2.max_y = -std::numeric_limits<double>::infinity();
                        for (int k = 0; k < n; ++k) {
                            const BoundingBox& bb =
                                (k == i) ? cand_bb : bbs[static_cast<size_t>(k)];
                            e2.min_x = std::min(e2.min_x, bb.min_x);
                            e2.max_x = std::max(e2.max_x, bb.max_x);
                            e2.min_y = std::min(e2.min_y, bb.min_y);
                            e2.max_y = std::max(e2.max_y, bb.max_y);
                        }
                        const double new_width = e2.max_x - e2.min_x;
                        const double new_height = e2.max_y - e2.min_y;
                        const double new_min_dim = std::min(new_width, new_height);
                        double new_side = std::max(new_width, new_height);
                        double new_cost = cost_from(new_width, new_height, new_overlap);
                        if (p.overlap_cost_cap > 0.0 && new_cost > p.overlap_cost_cap) {
                            scale *= 0.5;
                            continue;
                        }

                        reward_old_cost = curr_cost;
                        reward_new_cost = new_cost;
                        bool acc = accept_move(curr_cost, new_cost, T);
                        if (!acc) {
                            scale *= 0.5;
                            continue;
                        }

                        accepted = true;
		                        improved_curr = (new_cost + 1e-15 < curr_cost);
		                        const bool valid = (new_overlap <= p.overlap_eps_area);
		                        improved_best = valid && better_than_best(new_side, new_min_dim);

                            grid.update_position(i, cand.x, cand.y);
	                        poses[static_cast<size_t>(i)] = cand;
	                        polys[static_cast<size_t>(i)] = std::move(cand_poly);
	                        bbs[static_cast<size_t>(i)] = cand_bb;
	                        gmnx = e2.min_x;
                        gmxx = e2.max_x;
                        gmny = e2.min_y;
                        gmxy = e2.max_y;
                        curr_side = new_side;
	                        curr_overlap = new_overlap;
	                        curr_cost = new_cost;
	                        if (improved_best) {
	                            best.best_side = curr_side;
                                best_min_dim = new_min_dim;
	                            best.best_poses = poses;
	                        }
                        moved = true;
                    }
                }
            } else {
                add_reward(op, false, false, false, curr_cost, curr_cost);
                maybe_update_controller(t);
                continue;
            }

            add_reward(op, accepted, improved_best, improved_curr, reward_old_cost, reward_new_cost);
            if (improved_best) {
                last_best_iter = t;
            }
            maybe_update_controller(t);
        }

        return best;
}
