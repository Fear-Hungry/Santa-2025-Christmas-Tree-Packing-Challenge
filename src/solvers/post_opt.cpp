#include "solvers/post_opt.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <utility>
#include <vector>

#include "geometry/collision.hpp"
#include "utils/score.hpp"
#include "utils/submission_io.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int kSubmissionDecimals = 9;

int omp_max_threads() {
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}

int omp_thread_id() {
#if defined(_OPENMP)
    return omp_get_thread_num();
#else
    return 0;
#endif
}

void omp_set_threads(int threads) {
#if defined(_OPENMP)
    if (threads > 0) {
        omp_set_num_threads(threads);
    }
#else
    (void)threads;
#endif
}

double wrap_deg360(double deg) {
    deg = std::fmod(deg, 360.0);
    if (deg < 0.0) {
        deg += 360.0;
    }
    return deg;
}

struct FastRng {
    uint64_t s[2];

    explicit FastRng(uint64_t seed = 42) {
        s[0] = seed ^ 0x853c49e6748fea9bULL;
        s[1] = (seed * 0x9e3779b97f4a7c15ULL) ^ 0xc4ceb9fe1a85ec53ULL;
    }

    static inline uint64_t rotl(uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    inline uint64_t next() {
        uint64_t s0 = s[0];
        uint64_t s1 = s[1];
        uint64_t r = s0 + s1;
        s1 ^= s0;
        s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
        s[1] = rotl(s1, 37);
        return r;
    }

    inline double rf() {
        return (next() >> 11) * 0x1.0p-53;
    }

    inline double rf2() {
        return rf() * 2.0 - 1.0;
    }

    inline int ri(int n) {
        return static_cast<int>(next() % static_cast<uint64_t>(n));
    }

    inline double gaussian() {
        double u1 = rf() + 1e-10;
        double u2 = rf();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * kPi * u2);
    }
};

struct PostCfg {
    int n = 0;
    std::vector<TreePose> poses;
    std::vector<Polygon> polys;
    std::vector<BoundingBox> bbs;
    double gx0 = 0.0;
    double gy0 = 0.0;
    double gx1 = 0.0;
    double gy1 = 0.0;

    void resize(int nn) {
        n = nn;
        poses.resize(static_cast<size_t>(n));
        polys.resize(static_cast<size_t>(n));
        bbs.resize(static_cast<size_t>(n));
    }

    void update_poly(int i, const Polygon& base_poly) {
        const TreePose& pose = poses[static_cast<size_t>(i)];
        Polygon& poly = polys[static_cast<size_t>(i)];
        if (poly.size() != base_poly.size()) {
            poly.assign(base_poly.size(), Point{0.0, 0.0});
        }
        const double r = pose.deg * kPi / 180.0;
        const double c = std::cos(r);
        const double s = std::sin(r);
        double min_x = std::numeric_limits<double>::infinity();
        double max_x = -std::numeric_limits<double>::infinity();
        double min_y = std::numeric_limits<double>::infinity();
        double max_y = -std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < base_poly.size(); ++k) {
            const Point& v = base_poly[k];
            const double x = c * v.x - s * v.y + pose.x;
            const double y = s * v.x + c * v.y + pose.y;
            poly[k] = Point{x, y};
            min_x = std::min(min_x, x);
            max_x = std::max(max_x, x);
            min_y = std::min(min_y, y);
            max_y = std::max(max_y, y);
        }
        bbs[static_cast<size_t>(i)] = BoundingBox{min_x, max_x, min_y, max_y};
    }

    void update_all(const Polygon& base_poly) {
        for (int i = 0; i < n; ++i) {
            update_poly(i, base_poly);
        }
        update_global();
    }

    void update_global() {
        if (n <= 0) {
            gx0 = gy0 = 0.0;
            gx1 = gy1 = 0.0;
            return;
        }
        gx0 = gy0 = std::numeric_limits<double>::infinity();
        gx1 = gy1 = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < n; ++i) {
            const BoundingBox& bb = bbs[static_cast<size_t>(i)];
            gx0 = std::min(gx0, bb.min_x);
            gx1 = std::max(gx1, bb.max_x);
            gy0 = std::min(gy0, bb.min_y);
            gy1 = std::max(gy1, bb.max_y);
        }
    }

    static inline bool aabb_overlap(const BoundingBox& a, const BoundingBox& b) {
        if (a.max_x < b.min_x || b.max_x < a.min_x) {
            return false;
        }
        if (a.max_y < b.min_y || b.max_y < a.min_y) {
            return false;
        }
        return true;
    }

    bool has_overlap(int i) const {
        const BoundingBox& bi = bbs[static_cast<size_t>(i)];
        const Polygon& pi = polys[static_cast<size_t>(i)];
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                continue;
            }
            const BoundingBox& bj = bbs[static_cast<size_t>(j)];
            if (!aabb_overlap(bi, bj)) {
                continue;
            }
            if (polygons_intersect(pi, polys[static_cast<size_t>(j)])) {
                return true;
            }
        }
        return false;
    }

    bool any_overlap() const {
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (!aabb_overlap(bbs[static_cast<size_t>(i)],
                                  bbs[static_cast<size_t>(j)])) {
                    continue;
                }
                if (polygons_intersect(polys[static_cast<size_t>(i)],
                                       polys[static_cast<size_t>(j)])) {
                    return true;
                }
            }
        }
        return false;
    }

    double side() const {
        return std::max(gx1 - gx0, gy1 - gy0);
    }

    double score() const {
        if (n <= 0) {
            return 0.0;
        }
        const double s = side();
        return (s * s) / static_cast<double>(n);
    }

    void get_boundary(std::vector<int>& out) const {
        out.clear();
        const double eps = 0.01;
        for (int i = 0; i < n; ++i) {
            const BoundingBox& bb = bbs[static_cast<size_t>(i)];
            if (bb.min_x - gx0 < eps || gx1 - bb.max_x < eps ||
                bb.min_y - gy0 < eps || gy1 - bb.max_y < eps) {
                out.push_back(i);
            }
        }
    }

    PostCfg remove_tree(int remove_idx, const Polygon& base_poly) const {
        PostCfg out;
        out.resize(n - 1);
        int j = 0;
        for (int i = 0; i < n; ++i) {
            if (i == remove_idx) {
                continue;
            }
            out.poses[static_cast<size_t>(j)] = poses[static_cast<size_t>(i)];
            ++j;
        }
        out.update_all(base_poly);
        return out;
    }
};

PostCfg make_cfg(const Polygon& base_poly, const std::vector<TreePose>& poses) {
    PostCfg c;
    c.resize(static_cast<int>(poses.size()));
    for (int i = 0; i < c.n; ++i) {
        c.poses[static_cast<size_t>(i)] = poses[static_cast<size_t>(i)];
    }
    c.update_all(base_poly);
    return c;
}

struct TermSchedule {
    std::vector<int> tier;
    std::vector<int> tier_a_ns;
    std::vector<int> tier_b_ns;
};

TermSchedule compute_term_schedule(const std::vector<PostCfg>& cfg,
                                  int n_max,
                                  int term_min_n,
                                  int tier_a,
                                  int tier_b) {
    TermSchedule out;
    out.tier.assign(static_cast<size_t>(n_max + 1), 3);

    if (n_max <= 0) {
        return out;
    }

    std::vector<std::pair<double, int>> order;
    order.reserve(static_cast<size_t>(n_max));

    const int min_n = std::min(std::max(1, term_min_n), n_max);
    for (int n = min_n; n <= n_max; ++n) {
        order.emplace_back(cfg[static_cast<size_t>(n)].score(), n);
    }

    std::sort(order.begin(),
              order.end(),
              [](const auto& a, const auto& b) {
                  if (a.first != b.first) {
                      return a.first > b.first;
                  }
                  return a.second < b.second;
              });

    const int max_rank = static_cast<int>(order.size());
    const int b = std::min(std::max(0, tier_b), max_rank);
    const int a = std::min(std::max(0, tier_a), b);

    out.tier_a_ns.reserve(static_cast<size_t>(a));
    out.tier_b_ns.reserve(static_cast<size_t>(b - a));

    for (int i = 0; i < a; ++i) {
        const int n = order[static_cast<size_t>(i)].second;
        out.tier[static_cast<size_t>(n)] = 1;
        out.tier_a_ns.push_back(n);
    }
    for (int i = a; i < b; ++i) {
        const int n = order[static_cast<size_t>(i)].second;
        out.tier[static_cast<size_t>(n)] = 2;
        out.tier_b_ns.push_back(n);
    }

    return out;
}

double it_mult_for_tier(const PostOptOptions& opt, int tier) {
    if (tier == 1) {
        return opt.tier_a_iters_mult;
    }
    if (tier == 2) {
        return opt.tier_b_iters_mult;
    }
    return opt.tier_c_iters_mult;
}

double rs_mult_for_tier(const PostOptOptions& opt, int tier) {
    if (tier == 1) {
        return opt.tier_a_restarts_mult;
    }
    if (tier == 2) {
        return opt.tier_b_restarts_mult;
    }
    return opt.tier_c_restarts_mult;
}

double tighten_mult_for_tier(const PostOptOptions& opt, int tier) {
    if (!opt.enable_term_scheduler) {
        return 1.0;
    }
    if (tier == 1) {
        return opt.tier_a_tighten_mult;
    }
    if (tier == 2) {
        return opt.tier_b_tighten_mult;
    }
    return opt.tier_c_tighten_mult;
}

int scaled_budget(int base, double mult) {
    return std::max(0, static_cast<int>(std::llround(static_cast<double>(base) * mult)));
}

PostCfg squeeze(PostCfg c, const Polygon& base_poly) {
    const double cx = (c.gx0 + c.gx1) * 0.5;
    const double cy = (c.gy0 + c.gy1) * 0.5;
    for (double scale = 0.9995; scale >= 0.98; scale -= 0.0005) {
        PostCfg trial = c;
        for (int i = 0; i < trial.n; ++i) {
            TreePose& p = trial.poses[static_cast<size_t>(i)];
            p.x = cx + (p.x - cx) * scale;
            p.y = cy + (p.y - cy) * scale;
        }
        trial.update_all(base_poly);
        if (!trial.any_overlap()) {
            c = std::move(trial);
        } else {
            break;
        }
    }
    return c;
}

PostCfg compaction(PostCfg c, const Polygon& base_poly, int iters) {
    double best_side = c.side();
    for (int it = 0; it < iters; ++it) {
        const double cx = (c.gx0 + c.gx1) * 0.5;
        const double cy = (c.gy0 + c.gy1) * 0.5;
        bool improved = false;
        for (int i = 0; i < c.n; ++i) {
            TreePose& pose = c.poses[static_cast<size_t>(i)];
            double ox = pose.x;
            double oy = pose.y;
            const double dx = cx - pose.x;
            const double dy = cy - pose.y;
            const double d = std::sqrt(dx * dx + dy * dy);
            if (d < 1e-6) {
                continue;
            }
            const double steps[] = {0.02, 0.008, 0.003, 0.001, 0.0004};
            for (double step : steps) {
                pose.x = ox + dx / d * step;
                pose.y = oy + dy / d * step;
                c.update_poly(i, base_poly);
                if (!c.has_overlap(i)) {
                    c.update_global();
                    const double side = c.side();
                    if (side < best_side - 1e-12) {
                        best_side = side;
                        improved = true;
                        ox = pose.x;
                        oy = pose.y;
                    } else {
                        pose.x = ox;
                        pose.y = oy;
                        c.update_poly(i, base_poly);
                    }
                } else {
                    pose.x = ox;
                    pose.y = oy;
                    c.update_poly(i, base_poly);
                }
            }
        }
        c.update_global();
        if (!improved) {
            break;
        }
    }
    return c;
}

PostCfg local_search(PostCfg c, const Polygon& base_poly, int max_iter) {
    double best_side = c.side();
    const double steps[] = {0.01, 0.004, 0.002, 0.001, 0.0005, 0.00025, 0.0001};
    const double rots[] = {4.0, 2.0, 1.0, 0.5, 0.25, 0.125};
    const int dxs[] = {1, -1, 0, 0, 1, 1, -1, -1};
    const int dys[] = {0, 0, 1, -1, 1, -1, 1, -1};

    for (int iter = 0; iter < max_iter; ++iter) {
        bool improved = false;
        for (int i = 0; i < c.n; ++i) {
            TreePose& pose = c.poses[static_cast<size_t>(i)];
            const double cx = (c.gx0 + c.gx1) * 0.5;
            const double cy = (c.gy0 + c.gy1) * 0.5;
            const double ddx = cx - pose.x;
            const double ddy = cy - pose.y;
            const double dist = std::sqrt(ddx * ddx + ddy * ddy);
            if (dist > 1e-6) {
                for (double st : steps) {
                    const double ox = pose.x;
                    const double oy = pose.y;
                    pose.x += ddx / dist * st;
                    pose.y += ddy / dist * st;
                    c.update_poly(i, base_poly);
                    if (!c.has_overlap(i)) {
                        c.update_global();
                        const double side = c.side();
                        if (side < best_side - 1e-12) {
                            best_side = side;
                            improved = true;
                        } else {
                            pose.x = ox;
                            pose.y = oy;
                            c.update_poly(i, base_poly);
                            c.update_global();
                        }
                    } else {
                        pose.x = ox;
                        pose.y = oy;
                        c.update_poly(i, base_poly);
                    }
                }
            }

            for (double st : steps) {
                for (int d = 0; d < 8; ++d) {
                    const double ox = pose.x;
                    const double oy = pose.y;
                    pose.x += static_cast<double>(dxs[d]) * st;
                    pose.y += static_cast<double>(dys[d]) * st;
                    c.update_poly(i, base_poly);
                    if (!c.has_overlap(i)) {
                        c.update_global();
                        const double side = c.side();
                        if (side < best_side - 1e-12) {
                            best_side = side;
                            improved = true;
                        } else {
                            pose.x = ox;
                            pose.y = oy;
                            c.update_poly(i, base_poly);
                            c.update_global();
                        }
                    } else {
                        pose.x = ox;
                        pose.y = oy;
                        c.update_poly(i, base_poly);
                    }
                }
            }

            for (double rt : rots) {
                for (double da : {rt, -rt}) {
                    const double oa = pose.deg;
                    pose.deg = wrap_deg360(pose.deg + da);
                    c.update_poly(i, base_poly);
                    if (!c.has_overlap(i)) {
                        c.update_global();
                        const double side = c.side();
                        if (side < best_side - 1e-12) {
                            best_side = side;
                            improved = true;
                        } else {
                            pose.deg = oa;
                            c.update_poly(i, base_poly);
                            c.update_global();
                        }
                    } else {
                        pose.deg = oa;
                        c.update_poly(i, base_poly);
                    }
                }
            }
        }
        if (!improved) {
            break;
        }
    }
    return c;
}

PostCfg edge_slide_compaction(PostCfg c, const Polygon& base_poly, int outer_iter) {
    double best_side = c.side();
    for (int it = 0; it < outer_iter; ++it) {
        bool improved = false;
        for (int i = 0; i < c.n; ++i) {
            const double gcx = (c.gx0 + c.gx1) * 0.5;
            const double gcy = (c.gy0 + c.gy1) * 0.5;
            const double dirs[5][2] = {
                {gcx - c.poses[static_cast<size_t>(i)].x,
                 gcy - c.poses[static_cast<size_t>(i)].y},
                {1.0, 0.0},
                {-1.0, 0.0},
                {0.0, 1.0},
                {0.0, -1.0},
            };

            for (int d = 0; d < 5; ++d) {
                double dx = dirs[d][0];
                double dy = dirs[d][1];
                double len = std::sqrt(dx * dx + dy * dy);
                if (len < 1e-9) {
                    continue;
                }
                dx /= len;
                dy /= len;

                const double max_step = 0.30;
                double lo = 0.0;
                double hi = max_step;
                double best_step = 0.0;

                const double ox = c.poses[static_cast<size_t>(i)].x;
                const double oy = c.poses[static_cast<size_t>(i)].y;

                for (int it2 = 0; it2 < 20; ++it2) {
                    const double mid = 0.5 * (lo + hi);

                    c.poses[static_cast<size_t>(i)].x =
                        quantize_value(ox + dx * mid, kSubmissionDecimals);
                    c.poses[static_cast<size_t>(i)].y =
                        quantize_value(oy + dy * mid, kSubmissionDecimals);
                    c.update_poly(i, base_poly);
                    c.update_global();

                    const bool ok_overlap = !c.has_overlap(i);
                    const bool ok_side = (c.side() <= best_side + 1e-9);

                    if (ok_overlap && ok_side) {
                        best_step = mid;
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }

                if (best_step > 1e-6) {
                    c.poses[static_cast<size_t>(i)].x =
                        quantize_value(ox + dx * best_step, kSubmissionDecimals);
                    c.poses[static_cast<size_t>(i)].y =
                        quantize_value(oy + dy * best_step, kSubmissionDecimals);
                    c.update_poly(i, base_poly);
                    c.update_global();

                    const double ns = c.side();
                    if (ns < best_side - 1e-12) {
                        best_side = ns;
                        improved = true;
                    }
                } else {
                    c.poses[static_cast<size_t>(i)].x = ox;
                    c.poses[static_cast<size_t>(i)].y = oy;
                    c.update_poly(i, base_poly);
                    c.update_global();
                }
            }
        }
        if (!improved) {
            break;
        }
    }
    return c;
}

PostCfg tighten_cfg(PostCfg c,
                   const Polygon& base_poly,
                   const PostOptOptions& opt,
                   int tier,
                   int compaction_base,
                   int edge_slide_base,
                   int local_search_base,
                   bool apply_squeeze = true) {
    if (apply_squeeze && opt.enable_squeeze) {
        c = squeeze(std::move(c), base_poly);
    }

    const double mult = tighten_mult_for_tier(opt, tier);
    const int compaction_iters = scaled_budget(compaction_base, mult);
    const int edge_slide_rounds = scaled_budget(edge_slide_base, mult);
    const int local_search_iters = scaled_budget(local_search_base, mult);

    if (opt.enable_compaction && compaction_iters > 0) {
        c = compaction(std::move(c), base_poly, compaction_iters);
    }
    if (opt.enable_edge_slide && edge_slide_rounds > 0) {
        c = edge_slide_compaction(std::move(c), base_poly, edge_slide_rounds);
    }
    if (opt.enable_local_search && local_search_iters > 0) {
        c = local_search(std::move(c), base_poly, local_search_iters);
    }
    return c;
}

PostCfg sa_opt(PostCfg c, const Polygon& base_poly, int iter, double t0, double tm, uint64_t seed) {
    FastRng rng(seed);
    PostCfg best = c;
    PostCfg cur = c;
    double best_side = best.side();
    double curr_side = best_side;
    double t = t0;
    double alpha = std::pow(tm / t0, 1.0 / std::max(1, iter));
    int no_imp = 0;

    for (int it = 0; it < iter; ++it) {
        const int mt = rng.ri(10);
        const double sc = t / t0;
        bool valid = true;

        if (mt == 0) {
            const int i = rng.ri(cur.n);
            const double ox = cur.poses[static_cast<size_t>(i)].x;
            const double oy = cur.poses[static_cast<size_t>(i)].y;
            cur.poses[static_cast<size_t>(i)].x += rng.gaussian() * 0.5 * sc;
            cur.poses[static_cast<size_t>(i)].y += rng.gaussian() * 0.5 * sc;
            cur.update_poly(i, base_poly);
            if (cur.has_overlap(i)) {
                cur.poses[static_cast<size_t>(i)].x = ox;
                cur.poses[static_cast<size_t>(i)].y = oy;
                cur.update_poly(i, base_poly);
                valid = false;
            }
        } else if (mt == 1) {
            const int i = rng.ri(cur.n);
            const double ox = cur.poses[static_cast<size_t>(i)].x;
            const double oy = cur.poses[static_cast<size_t>(i)].y;
            const double bcx = (cur.gx0 + cur.gx1) * 0.5;
            const double bcy = (cur.gy0 + cur.gy1) * 0.5;
            const double dx = bcx - cur.poses[static_cast<size_t>(i)].x;
            const double dy = bcy - cur.poses[static_cast<size_t>(i)].y;
            const double d = std::sqrt(dx * dx + dy * dy);
            if (d > 1e-6) {
                cur.poses[static_cast<size_t>(i)].x += dx / d * rng.rf() * 0.6 * sc;
                cur.poses[static_cast<size_t>(i)].y += dy / d * rng.rf() * 0.6 * sc;
            }
            cur.update_poly(i, base_poly);
            if (cur.has_overlap(i)) {
                cur.poses[static_cast<size_t>(i)].x = ox;
                cur.poses[static_cast<size_t>(i)].y = oy;
                cur.update_poly(i, base_poly);
                valid = false;
            }
        } else if (mt == 2) {
            const int i = rng.ri(cur.n);
            const double oa = cur.poses[static_cast<size_t>(i)].deg;
            cur.poses[static_cast<size_t>(i)].deg =
                wrap_deg360(cur.poses[static_cast<size_t>(i)].deg +
                            rng.gaussian() * 80.0 * sc);
            cur.update_poly(i, base_poly);
            if (cur.has_overlap(i)) {
                cur.poses[static_cast<size_t>(i)].deg = oa;
                cur.update_poly(i, base_poly);
                valid = false;
            }
        } else if (mt == 3) {
            const int i = rng.ri(cur.n);
            const double ox = cur.poses[static_cast<size_t>(i)].x;
            const double oy = cur.poses[static_cast<size_t>(i)].y;
            const double oa = cur.poses[static_cast<size_t>(i)].deg;
            cur.poses[static_cast<size_t>(i)].x += rng.rf2() * 0.5 * sc;
            cur.poses[static_cast<size_t>(i)].y += rng.rf2() * 0.5 * sc;
            cur.poses[static_cast<size_t>(i)].deg =
                wrap_deg360(cur.poses[static_cast<size_t>(i)].deg +
                            rng.rf2() * 60.0 * sc);
            cur.update_poly(i, base_poly);
            if (cur.has_overlap(i)) {
                cur.poses[static_cast<size_t>(i)].x = ox;
                cur.poses[static_cast<size_t>(i)].y = oy;
                cur.poses[static_cast<size_t>(i)].deg = oa;
                cur.update_poly(i, base_poly);
                valid = false;
            }
        } else if (mt == 4) {
            std::vector<int> boundary;
            cur.get_boundary(boundary);
            if (!boundary.empty()) {
                const int i = boundary[static_cast<size_t>(rng.ri(static_cast<int>(boundary.size())))];
                const double ox = cur.poses[static_cast<size_t>(i)].x;
                const double oy = cur.poses[static_cast<size_t>(i)].y;
                const double oa = cur.poses[static_cast<size_t>(i)].deg;
                const double bcx = (cur.gx0 + cur.gx1) * 0.5;
                const double bcy = (cur.gy0 + cur.gy1) * 0.5;
                const double dx = bcx - cur.poses[static_cast<size_t>(i)].x;
                const double dy = bcy - cur.poses[static_cast<size_t>(i)].y;
                const double d = std::sqrt(dx * dx + dy * dy);
                if (d > 1e-6) {
                    cur.poses[static_cast<size_t>(i)].x += dx / d * rng.rf() * 0.7 * sc;
                    cur.poses[static_cast<size_t>(i)].y += dy / d * rng.rf() * 0.7 * sc;
                }
                cur.poses[static_cast<size_t>(i)].deg =
                    wrap_deg360(cur.poses[static_cast<size_t>(i)].deg +
                                rng.rf2() * 50.0 * sc);
                cur.update_poly(i, base_poly);
                if (cur.has_overlap(i)) {
                    cur.poses[static_cast<size_t>(i)].x = ox;
                    cur.poses[static_cast<size_t>(i)].y = oy;
                    cur.poses[static_cast<size_t>(i)].deg = oa;
                    cur.update_poly(i, base_poly);
                    valid = false;
                }
            } else {
                valid = false;
            }
        } else if (mt == 5) {
            const double factor = 1.0 - rng.rf() * 0.004 * sc;
            const double cx = (cur.gx0 + cur.gx1) * 0.5;
            const double cy = (cur.gy0 + cur.gy1) * 0.5;
            PostCfg trial = cur;
            for (int i = 0; i < trial.n; ++i) {
                TreePose& pose = trial.poses[static_cast<size_t>(i)];
                pose.x = cx + (pose.x - cx) * factor;
                pose.y = cy + (pose.y - cy) * factor;
            }
            trial.update_all(base_poly);
            if (!trial.any_overlap()) {
                cur = std::move(trial);
            } else {
                valid = false;
            }
        } else if (mt == 6) {
            const int i = rng.ri(cur.n);
            const double ox = cur.poses[static_cast<size_t>(i)].x;
            const double oy = cur.poses[static_cast<size_t>(i)].y;
            const double levy = std::pow(rng.rf() + 0.001, -1.3) * 0.008;
            cur.poses[static_cast<size_t>(i)].x += rng.rf2() * levy;
            cur.poses[static_cast<size_t>(i)].y += rng.rf2() * levy;
            cur.update_poly(i, base_poly);
            if (cur.has_overlap(i)) {
                cur.poses[static_cast<size_t>(i)].x = ox;
                cur.poses[static_cast<size_t>(i)].y = oy;
                cur.update_poly(i, base_poly);
                valid = false;
            }
        } else if (mt == 7 && cur.n > 1) {
            const int i = rng.ri(cur.n);
            const int j = (i + 1) % cur.n;
            const double oxi = cur.poses[static_cast<size_t>(i)].x;
            const double oyi = cur.poses[static_cast<size_t>(i)].y;
            const double oxj = cur.poses[static_cast<size_t>(j)].x;
            const double oyj = cur.poses[static_cast<size_t>(j)].y;
            const double dx = rng.rf2() * 0.3 * sc;
            const double dy = rng.rf2() * 0.3 * sc;
            cur.poses[static_cast<size_t>(i)].x += dx;
            cur.poses[static_cast<size_t>(i)].y += dy;
            cur.poses[static_cast<size_t>(j)].x += dx;
            cur.poses[static_cast<size_t>(j)].y += dy;
            cur.update_poly(i, base_poly);
            cur.update_poly(j, base_poly);
            if (cur.has_overlap(i) || cur.has_overlap(j)) {
                cur.poses[static_cast<size_t>(i)].x = oxi;
                cur.poses[static_cast<size_t>(i)].y = oyi;
                cur.poses[static_cast<size_t>(j)].x = oxj;
                cur.poses[static_cast<size_t>(j)].y = oyj;
                cur.update_poly(i, base_poly);
                cur.update_poly(j, base_poly);
                valid = false;
            }
        } else {
            const int i = rng.ri(cur.n);
            const double ox = cur.poses[static_cast<size_t>(i)].x;
            const double oy = cur.poses[static_cast<size_t>(i)].y;
            cur.poses[static_cast<size_t>(i)].x += rng.rf2() * 0.002;
            cur.poses[static_cast<size_t>(i)].y += rng.rf2() * 0.002;
            cur.update_poly(i, base_poly);
            if (cur.has_overlap(i)) {
                cur.poses[static_cast<size_t>(i)].x = ox;
                cur.poses[static_cast<size_t>(i)].y = oy;
                cur.update_poly(i, base_poly);
                valid = false;
            }
        }

        if (!valid) {
            ++no_imp;
            t *= alpha;
            if (t < tm) {
                t = tm;
            }
            continue;
        }

        cur.update_global();
        const double ns = cur.side();
        const double delta = ns - curr_side;

        if (delta < 0.0 || rng.rf() < std::exp(-delta / t)) {
            curr_side = ns;
            if (ns < best_side) {
                best_side = ns;
                best = cur;
                no_imp = 0;
            } else {
                ++no_imp;
            }
        } else {
            cur = best;
            curr_side = best_side;
            ++no_imp;
        }

        if (no_imp > 200) {
            t = std::min(t * 5.0, t0);
            no_imp = 0;
        }

        t *= alpha;
        if (t < tm) {
            t = tm;
        }
    }
    return best;
}

PostCfg perturb(PostCfg c, const Polygon& base_poly, double strength, FastRng& rng) {
    PostCfg original = c;
    const int np = std::max(1, static_cast<int>(c.n * 0.08 + strength * 3.0));
    for (int k = 0; k < np; ++k) {
        const int i = rng.ri(c.n);
        c.poses[static_cast<size_t>(i)].x += rng.gaussian() * strength * 0.5;
        c.poses[static_cast<size_t>(i)].y += rng.gaussian() * strength * 0.5;
        c.poses[static_cast<size_t>(i)].deg =
            wrap_deg360(c.poses[static_cast<size_t>(i)].deg + rng.gaussian() * 30.0);
    }
    c.update_all(base_poly);

    for (int iter = 0; iter < 150; ++iter) {
        bool fixed = true;
        for (int i = 0; i < c.n; ++i) {
            if (c.has_overlap(i)) {
                fixed = false;
                const double cx = (c.gx0 + c.gx1) * 0.5;
                const double cy = (c.gy0 + c.gy1) * 0.5;
                const double dx = c.poses[static_cast<size_t>(i)].x - cx;
                const double dy = c.poses[static_cast<size_t>(i)].y - cy;
                const double d = std::sqrt(dx * dx + dy * dy);
                if (d > 1e-6) {
                    c.poses[static_cast<size_t>(i)].x += dx / d * 0.02;
                    c.poses[static_cast<size_t>(i)].y += dy / d * 0.02;
                }
                c.poses[static_cast<size_t>(i)].deg =
                    wrap_deg360(c.poses[static_cast<size_t>(i)].deg + rng.rf2() * 15.0);
                c.update_poly(i, base_poly);
            }
        }
        if (fixed) {
            break;
        }
    }

    c.update_global();
    if (c.any_overlap()) {
        return original;
    }
    return c;
}

PostCfg optimize_parallel(const PostCfg& c,
                          const Polygon& base_poly,
                          const PostOptOptions& opt,
                          int iters,
                          int restarts,
                          int tier) {
    PostCfg global_best = c;
    double global_best_side = c.side();

#pragma omp parallel
    {
        const int tid = omp_thread_id();
        FastRng rng(opt.seed + static_cast<uint64_t>(tid) * 1000ULL +
                    static_cast<uint64_t>(c.n) * 131ULL);
        PostCfg local_best = c;
        double local_best_side = c.side();

#pragma omp for schedule(dynamic)
        for (int r = 0; r < restarts; ++r) {
            PostCfg start;
            if (r == 0) {
                start = c;
            } else {
                start = perturb(c, base_poly, 0.02 + 0.02 * (r % 8), rng);
                if (start.any_overlap()) {
                    continue;
                }
            }

            const uint64_t seed =
                opt.seed + static_cast<uint64_t>(r) * 1000ULL +
                static_cast<uint64_t>(tid) * 100000ULL +
                static_cast<uint64_t>(c.n);
            PostCfg o = sa_opt(start, base_poly, iters, opt.t0, opt.tm, seed);
            o = tighten_cfg(std::move(o), base_poly, opt, tier, 50, 10, 80);

            if (!o.any_overlap() && o.side() < local_best_side) {
                local_best_side = o.side();
                local_best = std::move(o);
            }
        }

#pragma omp critical
        {
            if (!local_best.any_overlap() && local_best_side < global_best_side) {
                global_best_side = local_best_side;
                global_best = std::move(local_best);
            }
        }
    }

    global_best = tighten_cfg(std::move(global_best), base_poly, opt, tier, 80, 12, 150);

    if (global_best.any_overlap()) {
        return c;
    }
    return global_best;
}

struct TreeState {
    double x = 0.0;
    double y = 0.0;
    double deg = 0.0;
};

struct ReinsertParams {
    bool guided = false;
    int attempts = 200;
    int shell_anchors = 32;
    int core_anchors = 64;
    int jitter_attempts = 0;
    double angle_jitter_deg = 30.0;
    double early_stop_rel = 0.0;
};

BoundingBox local_bb_for_deg(const Polygon& base_poly, double deg) {
    const double r = deg * kPi / 180.0;
    const double c = std::cos(r);
    const double s = std::sin(r);
    double min_x = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    for (const auto& v : base_poly) {
        const double x = c * v.x - s * v.y;
        const double y = s * v.x + c * v.y;
        min_x = std::min(min_x, x);
        max_x = std::max(max_x, x);
        min_y = std::min(min_y, y);
        max_y = std::max(max_y, y);
    }
    return BoundingBox{min_x, max_x, min_y, max_y};
}

ReinsertParams reinsert_params_for_tier(const PostOptOptions& opt, int tier) {
    ReinsertParams p;
    p.guided = opt.enable_guided_reinsert;
    p.shell_anchors = opt.reinsert_shell_anchors;
    p.core_anchors = opt.reinsert_core_anchors;
    p.jitter_attempts = opt.reinsert_jitter_attempts;
    p.angle_jitter_deg = opt.reinsert_angle_jitter_deg;
    p.early_stop_rel = opt.reinsert_early_stop_rel;

    if (tier == 1) {
        p.attempts = opt.reinsert_attempts_tier_a;
    } else if (tier == 2) {
        p.attempts = opt.reinsert_attempts_tier_b;
    } else {
        p.attempts = opt.reinsert_attempts_tier_c;
    }
    p.attempts = std::max(0, p.attempts);
    p.shell_anchors = std::max(0, p.shell_anchors);
    p.core_anchors = std::max(0, p.core_anchors);
    p.jitter_attempts = std::max(0, p.jitter_attempts);
    p.angle_jitter_deg = std::max(0.0, p.angle_jitter_deg);
    p.early_stop_rel = std::max(0.0, std::min(1.0, p.early_stop_rel));
    return p;
}

void compute_free_area(const PostCfg& c, std::vector<double>& free_area) {
    const int n = c.n;
    free_area.assign(n, 0.0);
    std::vector<double> area(static_cast<size_t>(n), 0.0);
    std::vector<double> overlap_sum(static_cast<size_t>(n), 0.0);

    for (int i = 0; i < n; ++i) {
        const BoundingBox& bb = c.bbs[static_cast<size_t>(i)];
        const double w = std::max(0.0, bb.max_x - bb.min_x);
        const double h = std::max(0.0, bb.max_y - bb.min_y);
        area[static_cast<size_t>(i)] = w * h;
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                continue;
            }
            const BoundingBox& bi = c.bbs[static_cast<size_t>(i)];
            const BoundingBox& bj = c.bbs[static_cast<size_t>(j)];
            const double ix0 = std::max(bi.min_x, bj.min_x);
            const double iy0 = std::max(bi.min_y, bj.min_y);
            const double ix1 = std::min(bi.max_x, bj.max_x);
            const double iy1 = std::min(bi.max_y, bj.max_y);
            const double dx = ix1 - ix0;
            const double dy = iy1 - iy0;
            if (dx > 0.0 && dy > 0.0) {
                overlap_sum[static_cast<size_t>(i)] += dx * dy;
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        const double occ = std::min(overlap_sum[static_cast<size_t>(i)],
                                    area[static_cast<size_t>(i)]);
        free_area[static_cast<size_t>(i)] =
            std::max(0.0, area[static_cast<size_t>(i)] - occ);
    }
}

void compute_protrude_score(const PostCfg& c, std::vector<double>& protrude) {
    const int n = c.n;
    protrude.assign(n, 0.0);
    const double cx = (c.gx0 + c.gx1) * 0.5;
    const double cy = (c.gy0 + c.gy1) * 0.5;
    const double side = c.side();
    const double eps = side * 0.02;

    for (int i = 0; i < n; ++i) {
        const BoundingBox& bb = c.bbs[static_cast<size_t>(i)];
        const bool on_boundary =
            (bb.min_x - c.gx0 < eps) || (c.gx1 - bb.max_x < eps) ||
            (bb.min_y - c.gy0 < eps) || (c.gy1 - bb.max_y < eps);

        if (!on_boundary) {
            protrude[static_cast<size_t>(i)] = 0.0;
            continue;
        }

        const double tx = 0.5 * (bb.min_x + bb.max_x);
        const double ty = 0.5 * (bb.min_y + bb.max_y);
        const double d = std::hypot(tx - cx, ty - cy);
        protrude[static_cast<size_t>(i)] = d;
    }
}

PostCfg reinsert_trees(const PostCfg& base,
                       const Polygon& base_poly,
                       const std::vector<TreeState>& removed,
                       uint64_t seed,
                       const ReinsertParams& params) {
    PostCfg cur = base;
    FastRng rng(seed);

    for (const auto& t : removed) {
        const double base_cx = (cur.gx0 + cur.gx1) * 0.5;
        const double base_cy = (cur.gy0 + cur.gy1) * 0.5;
        const double base_side = cur.side();

        std::vector<int> shell;
        std::vector<int> core;
        if (params.guided && cur.n >= 1) {
            int idx_min_x = 0;
            int idx_max_x = 0;
            int idx_min_y = 0;
            int idx_max_y = 0;
            for (int i = 1; i < cur.n; ++i) {
                if (cur.bbs[static_cast<size_t>(i)].min_x < cur.bbs[static_cast<size_t>(idx_min_x)].min_x) {
                    idx_min_x = i;
                }
                if (cur.bbs[static_cast<size_t>(i)].max_x > cur.bbs[static_cast<size_t>(idx_max_x)].max_x) {
                    idx_max_x = i;
                }
                if (cur.bbs[static_cast<size_t>(i)].min_y < cur.bbs[static_cast<size_t>(idx_min_y)].min_y) {
                    idx_min_y = i;
                }
                if (cur.bbs[static_cast<size_t>(i)].max_y > cur.bbs[static_cast<size_t>(idx_max_y)].max_y) {
                    idx_max_y = i;
                }
            }

            auto add_shell = [&](int idx) {
                if (params.shell_anchors <= 0) {
                    return;
                }
                if (static_cast<int>(shell.size()) >= params.shell_anchors) {
                    return;
                }
                if (idx < 0 || idx >= cur.n) {
                    return;
                }
                if (std::find(shell.begin(), shell.end(), idx) != shell.end()) {
                    return;
                }
                shell.push_back(idx);
            };

            add_shell(idx_min_x);
            add_shell(idx_max_x);
            add_shell(idx_min_y);
            add_shell(idx_max_y);
            const int fixed_shell = static_cast<int>(shell.size());

            std::vector<double> protrude;
            compute_protrude_score(cur, protrude);
            std::vector<std::pair<double, int>> prot_list;
            prot_list.reserve(static_cast<size_t>(cur.n));
            for (int i = 0; i < cur.n; ++i) {
                if (protrude[static_cast<size_t>(i)] > 0.0) {
                    prot_list.emplace_back(protrude[static_cast<size_t>(i)], i);
                }
            }
            std::sort(prot_list.begin(), prot_list.end(),
                      [](const auto& a, const auto& b) {
                          if (a.first != b.first) {
                              return a.first > b.first;
                          }
                          return a.second < b.second;
                      });
            shell.reserve(static_cast<size_t>(std::max(params.shell_anchors, fixed_shell)));
            for (int i = 0; i < static_cast<int>(prot_list.size()); ++i) {
                add_shell(prot_list[static_cast<size_t>(i)].second);
            }

            std::vector<char> used(static_cast<size_t>(cur.n), 0);
            for (int idx : shell) {
                used[static_cast<size_t>(idx)] = 1;
            }

            std::vector<std::pair<double, int>> center_dist;
            center_dist.reserve(static_cast<size_t>(cur.n));
            for (int i = 0; i < cur.n; ++i) {
                const double dx = cur.poses[static_cast<size_t>(i)].x - base_cx;
                const double dy = cur.poses[static_cast<size_t>(i)].y - base_cy;
                center_dist.emplace_back(dx * dx + dy * dy, i);
            }
            std::sort(center_dist.begin(),
                      center_dist.end(),
                      [](const auto& a, const auto& b) {
                          if (a.first != b.first) {
                              return a.first < b.first;
                          }
                          return a.second < b.second;
                      });

            const int core_k = std::min(cur.n, params.core_anchors);
            const int core_central = std::min(static_cast<int>(center_dist.size()),
                                              core_k / 2);
            core.reserve(static_cast<size_t>(core_k));
            for (int i = 0; i < core_central && static_cast<int>(core.size()) < core_k; ++i) {
                const int idx = center_dist[static_cast<size_t>(i)].second;
                if (!used[static_cast<size_t>(idx)]) {
                    used[static_cast<size_t>(idx)] = 1;
                    core.push_back(idx);
                }
            }
            while (static_cast<int>(core.size()) < core_k) {
                const int idx = rng.ri(cur.n);
                if (!used[static_cast<size_t>(idx)]) {
                    used[static_cast<size_t>(idx)] = 1;
                    core.push_back(idx);
                }
                if (static_cast<int>(core.size()) >= cur.n) {
                    break;
                }
            }

            for (int i = static_cast<int>(shell.size()) - 1; i > fixed_shell; --i) {
                const int j = fixed_shell + rng.ri(i - fixed_shell + 1);
                std::swap(shell[static_cast<size_t>(i)], shell[static_cast<size_t>(j)]);
            }
            for (int i = static_cast<int>(core.size()) - 1; i > 0; --i) {
                const int j = rng.ri(i + 1);
                std::swap(core[static_cast<size_t>(i)], core[static_cast<size_t>(j)]);
            }
        }

        cur.resize(cur.n + 1);
        const int idx = cur.n - 1;
        if (!params.guided && params.jitter_attempts <= 0) {
            cur.poses[static_cast<size_t>(idx)] = TreePose{t.x, t.y, t.deg};
        } else {
            cur.poses[static_cast<size_t>(idx)] =
                quantize_pose_wrap_deg(TreePose{t.x, t.y, t.deg}, kSubmissionDecimals);
        }
        cur.update_poly(idx, base_poly);

        if (!params.guided && params.jitter_attempts <= 0) {
            cur.update_global();
            bool placed = false;
            for (int attempt = 0; attempt < params.attempts; ++attempt) {
                if (!cur.has_overlap(idx)) {
                    placed = true;
                    break;
                }
                const double cx = (cur.gx0 + cur.gx1) * 0.5;
                const double cy = (cur.gy0 + cur.gy1) * 0.5;
                const double radius = 0.1 + 0.6 * rng.rf();
                const double ang = 2.0 * kPi * rng.rf();
                cur.poses[static_cast<size_t>(idx)].x = cx + radius * std::cos(ang);
                cur.poses[static_cast<size_t>(idx)].y = cy + radius * std::sin(ang);
                cur.poses[static_cast<size_t>(idx)].deg =
                    wrap_deg360(t.deg + rng.rf2() * 120.0);
                cur.update_poly(idx, base_poly);
                cur.update_global();
            }

            if (!placed) {
                return base;
            }
            continue;
        }

        bool placed = false;
        TreePose best_pose = cur.poses[static_cast<size_t>(idx)];
        double best_side = std::numeric_limits<double>::infinity();
        double baseline_side = std::numeric_limits<double>::infinity();
        double early_stop_target = -1.0;
        bool early_stop = false;
        int attempts_main = 0;

        auto eval_pose = [&](TreePose cand) {
            if (early_stop) {
                return;
            }
            cand = quantize_pose_wrap_deg(cand, kSubmissionDecimals);
            if (cand.x < -100.0 || cand.x > 100.0 || cand.y < -100.0 || cand.y > 100.0) {
                return;
            }
            cur.poses[static_cast<size_t>(idx)] = cand;
            cur.update_poly(idx, base_poly);
            if (cur.has_overlap(idx)) {
                return;
            }
            cur.update_global();
            const double side = cur.side();
            if (early_stop_target < 0.0) {
                baseline_side = side;
                early_stop_target = baseline_side * (1.0 - params.early_stop_rel);
            }
            if (side < best_side - 1e-12) {
                best_side = side;
                best_pose = cand;
                placed = true;
                if (params.early_stop_rel > 0.0 &&
                    early_stop_target > 0.0 &&
                    best_side <= early_stop_target - 1e-12) {
                    early_stop = true;
                }
            }
        };

        auto try_pose = [&](TreePose cand) {
            if (early_stop || attempts_main >= params.attempts) {
                return;
            }
            ++attempts_main;
            eval_pose(cand);
        };

        try_pose(TreePose{t.x, t.y, t.deg});

        std::vector<double> rot_list;
        rot_list.reserve(5);
        rot_list.push_back(t.deg);
        if (params.angle_jitter_deg > 0.0) {
            rot_list.push_back(t.deg + params.angle_jitter_deg);
            rot_list.push_back(t.deg - params.angle_jitter_deg);
            rot_list.push_back(t.deg + 2.0 * params.angle_jitter_deg);
            rot_list.push_back(t.deg - 2.0 * params.angle_jitter_deg);
        }

        auto contact_on_anchor = [&](int aidx, double deg, bool inward_only) {
            const BoundingBox& abb = cur.bbs[static_cast<size_t>(aidx)];
            const TreePose& ap = cur.poses[static_cast<size_t>(aidx)];
            deg = wrap_deg360(quantize_value(deg, kSubmissionDecimals));
            const BoundingBox lbb = local_bb_for_deg(base_poly, deg);
            constexpr double kGap = 1e-6;

            const double y_center = ap.y;
            const double y_bottom = abb.min_y - lbb.min_y;
            const double y_top = abb.max_y - lbb.max_y;
            const double x_center = ap.x;
            const double x_left = abb.min_x - lbb.min_x;
            const double x_right = abb.max_x - lbb.max_x;

            auto contact_right = [&]() {
                const double xr = abb.max_x + kGap - lbb.min_x;
                try_pose(TreePose{xr, y_center, deg});
                try_pose(TreePose{xr, y_bottom, deg});
                try_pose(TreePose{xr, y_top, deg});
            };
            auto contact_left = [&]() {
                const double xl = abb.min_x - kGap - lbb.max_x;
                try_pose(TreePose{xl, y_center, deg});
                try_pose(TreePose{xl, y_bottom, deg});
                try_pose(TreePose{xl, y_top, deg});
            };
            auto contact_up = [&]() {
                const double yu = abb.max_y + kGap - lbb.min_y;
                try_pose(TreePose{x_center, yu, deg});
                try_pose(TreePose{x_left, yu, deg});
                try_pose(TreePose{x_right, yu, deg});
            };
            auto contact_down = [&]() {
                const double yd = abb.min_y - kGap - lbb.max_y;
                try_pose(TreePose{x_center, yd, deg});
                try_pose(TreePose{x_left, yd, deg});
                try_pose(TreePose{x_right, yd, deg});
            };

            if (!inward_only) {
                contact_right();
                contact_left();
                contact_up();
                contact_down();
                return;
            }

            const double acx = 0.5 * (abb.min_x + abb.max_x);
            const double acy = 0.5 * (abb.min_y + abb.max_y);
            if (acx >= base_cx) {
                contact_left();
            } else {
                contact_right();
            }
            if (acy >= base_cy) {
                contact_down();
            } else {
                contact_up();
            }
        };

        if (params.guided) {
            for (int aidx : shell) {
                if (early_stop || attempts_main >= params.attempts) {
                    break;
                }
                for (double deg : rot_list) {
                    if (early_stop || attempts_main >= params.attempts) {
                        break;
                    }
                    contact_on_anchor(aidx, deg, true);
                }
            }
            for (int aidx : core) {
                if (early_stop || attempts_main >= params.attempts) {
                    break;
                }
                for (double deg : rot_list) {
                    if (early_stop || attempts_main >= params.attempts) {
                        break;
                    }
                    contact_on_anchor(aidx, deg, false);
                }
            }
        }

        // Random fallback around the current center.
        while (!early_stop && attempts_main < params.attempts) {
            const double radius = std::max(0.02, base_side * (0.15 + 0.85 * rng.rf()));
            const double ang = 2.0 * kPi * rng.rf();
            const double deg = t.deg + rng.rf2() * std::max(10.0, params.angle_jitter_deg * 4.0);
            try_pose(TreePose{base_cx + radius * std::cos(ang),
                              base_cy + radius * std::sin(ang),
                              deg});
        }

        // Local jitter around the best pose (extra attempts).
        if (placed && params.jitter_attempts > 0 && !early_stop) {
            for (int j = 0; j < params.jitter_attempts; ++j) {
                const double dx = rng.gaussian() * 0.02;
                const double dy = rng.gaussian() * 0.02;
                const double ddeg = rng.gaussian() * std::max(1.0, params.angle_jitter_deg * 0.25);
                eval_pose(TreePose{best_pose.x + dx, best_pose.y + dy, best_pose.deg + ddeg});
                if (early_stop) {
                    break;
                }
            }
        }

        if (!placed) {
            return base;
        }

        cur.poses[static_cast<size_t>(idx)] = best_pose;
        cur.update_poly(idx, base_poly);
        cur.update_global();
    }

    if (cur.any_overlap()) {
        return base;
    }
    return cur;
}

PostCfg free_area_heuristic(const PostCfg& c,
                            const Polygon& base_poly,
                            const PostOptOptions& opt,
                            double remove_ratio,
                            uint64_t seed,
                            int tier = 3) {
    PostCfg best = c;
    const int n = c.n;
    if (n <= opt.free_area_min_n) {
        return best;
    }

    int k = static_cast<int>(std::floor(n * remove_ratio + 1e-9));
    k = std::max(1, std::min(k, n - 1));

    std::vector<double> free_area;
    std::vector<double> protrude;
    compute_free_area(c, free_area);
    compute_protrude_score(c, protrude);

    std::vector<std::pair<double, int>> free_list;
    free_list.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        free_list.emplace_back(free_area[static_cast<size_t>(i)], i);
    }
    std::sort(free_list.begin(), free_list.end(),
              [](const auto& a, const auto& b) {
                  if (a.first != b.first) {
                      return a.first > b.first;
                  }
                  return a.second < b.second;
              });

    std::vector<std::pair<double, int>> prot_list;
    prot_list.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        if (protrude[static_cast<size_t>(i)] > 0.0) {
            prot_list.emplace_back(protrude[static_cast<size_t>(i)], i);
        }
    }
    std::sort(prot_list.begin(), prot_list.end(),
              [](const auto& a, const auto& b) {
                  if (a.first != b.first) {
                      return a.first > b.first;
                  }
                  return a.second < b.second;
              });

    int k_prot = std::min(static_cast<int>(prot_list.size()), (k * 2) / 3);
    int k_free = k - k_prot;
    if (k_free < 0) {
        k_free = 0;
    }

    std::vector<bool> remove_flag(static_cast<size_t>(n), false);
    std::vector<TreeState> removed;
    removed.reserve(static_cast<size_t>(k));

    int removed_cnt = 0;
    for (int i = 0; i < static_cast<int>(prot_list.size()) && removed_cnt < k_prot; ++i) {
        const int idx = prot_list[static_cast<size_t>(i)].second;
        if (remove_flag[static_cast<size_t>(idx)]) {
            continue;
        }
        remove_flag[static_cast<size_t>(idx)] = true;
        const TreePose& p = c.poses[static_cast<size_t>(idx)];
        removed.push_back(TreeState{p.x, p.y, p.deg});
        ++removed_cnt;
    }

    for (int i = 0; i < static_cast<int>(free_list.size()) && removed_cnt < k; ++i) {
        const int idx = free_list[static_cast<size_t>(i)].second;
        if (remove_flag[static_cast<size_t>(idx)]) {
            continue;
        }
        remove_flag[static_cast<size_t>(idx)] = true;
        const TreePose& p = c.poses[static_cast<size_t>(idx)];
        removed.push_back(TreeState{p.x, p.y, p.deg});
        ++removed_cnt;
    }

    if (removed.empty()) {
        return best;
    }

    PostCfg reduced;
    reduced.resize(n - static_cast<int>(removed.size()));
    int ptr = 0;
    for (int i = 0; i < n; ++i) {
        if (!remove_flag[static_cast<size_t>(i)]) {
            reduced.poses[static_cast<size_t>(ptr)] = c.poses[static_cast<size_t>(i)];
            ++ptr;
        }
    }
    reduced.update_all(base_poly);
    if (reduced.any_overlap()) {
        return best;
    }

    int reduced_iters = 8000;
    int reduced_restarts = 8;
    if (opt.enable_term_scheduler) {
        reduced_iters = scaled_budget(reduced_iters, it_mult_for_tier(opt, tier));
        reduced_restarts = scaled_budget(reduced_restarts, rs_mult_for_tier(opt, tier));
    }

    PostCfg reduced_opt = reduced;
    if (reduced_iters > 0 && reduced_restarts > 0) {
        reduced_opt =
            optimize_parallel(reduced, base_poly, opt, reduced_iters, reduced_restarts, tier);
    }

    ReinsertParams rp = reinsert_params_for_tier(opt, tier);
    PostCfg with_inserted = reinsert_trees(reduced_opt, base_poly, removed, seed, rp);
    if (with_inserted.n != n || with_inserted.any_overlap()) {
        return best;
    }

    with_inserted = tighten_cfg(std::move(with_inserted), base_poly, opt, tier, 40, 10, 80);

    if (!with_inserted.any_overlap() &&
        with_inserted.side() < best.side() - 1e-12) {
        return with_inserted;
    }
    return best;
}

}  // namespace

bool post_optimize_submission(const Polygon& base_poly,
                              std::vector<std::vector<TreePose>>& solutions_by_n,
                              const PostOptOptions& opt,
                              PostOptStats* stats_out) {
    if (!opt.enabled) {
        return false;
    }
    if (solutions_by_n.size() <= 1) {
        return false;
    }

    const int n_max = static_cast<int>(solutions_by_n.size()) - 1;
    std::vector<std::vector<TreePose>> original = solutions_by_n;

    omp_set_threads(opt.threads > 0 ? opt.threads : omp_max_threads());

    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<PostCfg> base_cfg(static_cast<size_t>(n_max + 1));
    for (int n = 1; n <= n_max; ++n) {
        base_cfg[static_cast<size_t>(n)] = make_cfg(base_poly, solutions_by_n[static_cast<size_t>(n)]);
    }

    double init_score = 0.0;
    for (int n = 1; n <= n_max; ++n) {
        init_score += base_cfg[static_cast<size_t>(n)].score();
    }

    std::vector<PostCfg> res(static_cast<size_t>(n_max + 1));
    std::vector<bool> has_res(static_cast<size_t>(n_max + 1), false);
    int phase1_improved = 0;
    TermSchedule initial_term_sched;
    bool have_initial_term_sched = false;

    if (opt.enable_term_scheduler) {
        res = base_cfg;
        for (int n = 1; n <= n_max; ++n) {
            has_res[static_cast<size_t>(n)] = true;
        }

        const int epochs = std::max(1, opt.term_epochs);
        const double eps_term = std::max(1e-12, opt.accept_term_eps);

        std::vector<PostOptTermEpochStats> term_epochs;
        const bool want_term_stats = (stats_out != nullptr);
        if (want_term_stats) {
            term_epochs.reserve(static_cast<size_t>(epochs));
            initial_term_sched = compute_term_schedule(
                res, n_max, opt.term_min_n, opt.term_tier_a, opt.term_tier_b);
            have_initial_term_sched = true;
            stats_out->term_summary = PostOptTermSummary{};
            stats_out->term_summary.tier_a_count =
                static_cast<int>(initial_term_sched.tier_a_ns.size());
            stats_out->term_summary.tier_b_count =
                static_cast<int>(initial_term_sched.tier_b_ns.size());
            stats_out->term_summary.tier_a_ns = initial_term_sched.tier_a_ns;
            for (int n : initial_term_sched.tier_a_ns) {
                stats_out->term_summary.tier_a_term_before +=
                    base_cfg[static_cast<size_t>(n)].score();
            }
            for (int n : initial_term_sched.tier_b_ns) {
                stats_out->term_summary.tier_b_term_before +=
                    base_cfg[static_cast<size_t>(n)].score();
            }
            stats_out->term_epochs.clear();
        }

        for (int epoch = 0; epoch < epochs; ++epoch) {
            const TermSchedule sched = compute_term_schedule(
                res, n_max, opt.term_min_n, opt.term_tier_a, opt.term_tier_b);

            PostOptTermEpochStats epoch_stats;
            if (want_term_stats) {
                epoch_stats.epoch = epoch;
                epoch_stats.tier_a_count = static_cast<int>(sched.tier_a_ns.size());
                epoch_stats.tier_b_count = static_cast<int>(sched.tier_b_ns.size());
                if (epoch == 0) {
                    epoch_stats.tier_a_ns = sched.tier_a_ns;
                }
                for (int n : sched.tier_a_ns) {
                    epoch_stats.tier_a_term_before += res[static_cast<size_t>(n)].score();
                }
                for (int n : sched.tier_b_ns) {
                    epoch_stats.tier_b_term_before += res[static_cast<size_t>(n)].score();
                }
            }

            for (int n = n_max; n >= 1; --n) {
                PostCfg c = res[static_cast<size_t>(n)];
                const double os = c.score();

                const int t = sched.tier[static_cast<size_t>(n)];
                const double it_mult = (t == 1)   ? opt.tier_a_iters_mult
                                       : (t == 2) ? opt.tier_b_iters_mult
                                                  : opt.tier_c_iters_mult;
                const double rs_mult = (t == 1)   ? opt.tier_a_restarts_mult
                                       : (t == 2) ? opt.tier_b_restarts_mult
                                                  : opt.tier_c_restarts_mult;

                int iters = static_cast<int>(std::llround(static_cast<double>(opt.iters) * it_mult));
                int restarts =
                    static_cast<int>(std::llround(static_cast<double>(opt.restarts) * rs_mult));
                iters = std::max(0, iters);
                restarts = std::max(0, restarts);

                PostCfg o = c;
                if (iters > 0 && restarts > 0) {
                    o = optimize_parallel(c, base_poly, opt, iters, restarts, t);
                }

                for (int m = n + 1; m <= std::min(n_max, n + 2); ++m) {
                    const PostCfg& pc = res[static_cast<size_t>(m)];
                    if (pc.n < n) {
                        continue;
                    }
                    PostCfg ad;
                    ad.resize(n);
                    for (int i = 0; i < n; ++i) {
                        ad.poses[static_cast<size_t>(i)] = pc.poses[static_cast<size_t>(i)];
                    }
                    ad.update_all(base_poly);
                    if (!ad.any_overlap()) {
                        ad = tighten_cfg(std::move(ad), base_poly, opt, t, 40, 8, 60, false);
                        if (!ad.any_overlap() && ad.score() < o.score() - eps_term) {
                            o = std::move(ad);
                        }
                    }
                }

                if (o.any_overlap() || o.side() > c.side() + 1e-14) {
                    o = c;
                }

                if (opt.enable_free_area && n >= opt.free_area_min_n && t <= 2) {
                    PostCfg oh =
                        free_area_heuristic(o,
                                            base_poly,
                                            opt,
                                            opt.remove_ratio,
                                            opt.seed + static_cast<uint64_t>(n) * 101ULL,
                                            t);
                    if (!oh.any_overlap() && oh.score() < o.score() - eps_term) {
                        o = std::move(oh);
                    }
                }

                if (o.score() < os - eps_term) {
                    res[static_cast<size_t>(n)] = std::move(o);
                } else {
                    res[static_cast<size_t>(n)] = std::move(c);
                }
            }

            if (want_term_stats) {
                for (int n : sched.tier_a_ns) {
                    epoch_stats.tier_a_term_after += res[static_cast<size_t>(n)].score();
                }
                for (int n : sched.tier_b_ns) {
                    epoch_stats.tier_b_term_after += res[static_cast<size_t>(n)].score();
                }
                term_epochs.push_back(std::move(epoch_stats));
            }
        }

        for (int n = 1; n <= n_max; ++n) {
            if (res[static_cast<size_t>(n)].score() < base_cfg[static_cast<size_t>(n)].score() - 1e-10) {
                ++phase1_improved;
            }
        }
        if (want_term_stats) {
            stats_out->term_epochs = std::move(term_epochs);
        }
    } else {
        for (int n = n_max; n >= 1; --n) {
            PostCfg c = base_cfg[static_cast<size_t>(n)];
            const double os = c.score();

            int iters = opt.iters;
            int restarts = opt.restarts;
            if (n <= 10) {
                iters = static_cast<int>(iters * 2.5);
                restarts = restarts * 2;
            } else if (n <= 30) {
                iters = static_cast<int>(iters * 1.8);
                restarts = static_cast<int>(restarts * 1.5);
            } else if (n <= 60) {
                iters = static_cast<int>(iters * 1.3);
            } else if (n > 150) {
                iters = static_cast<int>(iters * 0.7);
                restarts = std::max(1, static_cast<int>(restarts * 0.8));
            }
            restarts = std::max(4, restarts);

            PostCfg o = optimize_parallel(c, base_poly, opt, iters, restarts, 2);

            for (int m = n + 1; m <= std::min(n_max, n + 2); ++m) {
                if (!has_res[static_cast<size_t>(m)]) {
                    continue;
                }
                const PostCfg& pc = res[static_cast<size_t>(m)];
                if (pc.n < n) {
                    continue;
                }
                PostCfg ad;
                ad.resize(n);
                for (int i = 0; i < n; ++i) {
                    ad.poses[static_cast<size_t>(i)] = pc.poses[static_cast<size_t>(i)];
                }
                ad.update_all(base_poly);
                if (!ad.any_overlap()) {
                    if (opt.enable_compaction) {
                        ad = compaction(std::move(ad), base_poly, 40);
                    }
                    if (opt.enable_edge_slide) {
                        ad = edge_slide_compaction(std::move(ad), base_poly, 8);
                    }
                    if (opt.enable_local_search) {
                        ad = local_search(std::move(ad), base_poly, 60);
                    }
                    if (!ad.any_overlap() && ad.side() < o.side()) {
                        o = std::move(ad);
                    }
                }
            }

            if (o.any_overlap() || o.side() > c.side() + 1e-14) {
                o = c;
            }

            if (opt.enable_free_area && n >= opt.free_area_min_n) {
                PostCfg oh =
                    free_area_heuristic(o,
                                        base_poly,
                                        opt,
                                        opt.remove_ratio,
                                        opt.seed + static_cast<uint64_t>(n) * 101ULL);
                if (!oh.any_overlap() && oh.side() < o.side() - 1e-12) {
                    o = std::move(oh);
                }
            }

            res[static_cast<size_t>(n)] = std::move(o);
            has_res[static_cast<size_t>(n)] = true;
            const double ns = res[static_cast<size_t>(n)].score();
            if (ns < os - 1e-10) {
                ++phase1_improved;
            }
        }
    }

    int backprop_improved = 0;
    if (opt.enable_backprop) {
        bool changed = true;
        int pass_num = 0;

        while (changed && pass_num < opt.backprop_passes) {
            changed = false;
            ++pass_num;

            std::vector<int> tier_sched(static_cast<size_t>(n_max + 1), 2);
            if (opt.enable_term_scheduler) {
                tier_sched = compute_term_schedule(res,
                                                   n_max,
                                                   opt.term_min_n,
                                                   opt.term_tier_a,
                                                   opt.term_tier_b)
                                 .tier;
            }

            auto span_for_tier = [&](int t) -> int {
                if (t == 1) {
                    return opt.backprop_span_tier_a;
                }
                if (t == 2) {
                    return opt.backprop_span_tier_b;
                }
                return opt.backprop_span_tier_c;
            };
            auto max_combos_for_tier = [&](int t) -> int {
                if (t == 1) {
                    return opt.backprop_max_combos_tier_a;
                }
                if (t == 2) {
                    return opt.backprop_max_combos_tier_b;
                }
                return opt.backprop_max_combos_tier_c;
            };

            for (int k = n_max; k >= 2; --k) {
                if (!has_res[static_cast<size_t>(k)] ||
                    !has_res[static_cast<size_t>(k - 1)]) {
                    continue;
                }

                const int tier_k1 = tier_sched[static_cast<size_t>(k - 1)];
                if (opt.enable_term_scheduler && tier_k1 > 2) {
                    continue;
                }
                const double side_k = res[static_cast<size_t>(k)].side();
                const double side_k1 = res[static_cast<size_t>(k - 1)].side();

                if (side_k < side_k1 - 1e-12) {
                    const PostCfg& cfg_k = res[static_cast<size_t>(k)];
                    double best_side = side_k1;
                    PostCfg best_cfg = res[static_cast<size_t>(k - 1)];

#pragma omp parallel
                    {
                        double local_best_side = best_side;
                        PostCfg local_best_cfg = best_cfg;

#pragma omp for schedule(dynamic)
                        for (int remove_idx = 0; remove_idx < k; ++remove_idx) {
                            PostCfg reduced = cfg_k.remove_tree(remove_idx, base_poly);
                            if (!reduced.any_overlap()) {
                                reduced = tighten_cfg(std::move(reduced),
                                                      base_poly,
                                                      opt,
                                                      tier_k1,
                                                      60,
                                                      10,
                                                      100);
                                if (!reduced.any_overlap() &&
                                    reduced.side() < local_best_side) {
                                    local_best_side = reduced.side();
                                    local_best_cfg = std::move(reduced);
                                }
                            }
                        }

#pragma omp critical
                        {
                            if (local_best_side < best_side) {
                                best_side = local_best_side;
                                best_cfg = std::move(local_best_cfg);
                            }
                        }
                    }

                    if (best_side < side_k1 - 1e-12) {
                        res[static_cast<size_t>(k - 1)] = std::move(best_cfg);
                        ++backprop_improved;
                        changed = true;
                    }
                }
            }

            if (!opt.enable_backprop_explore) {
                for (int k = n_max; k >= 3; --k) {
                    const int tier_k = tier_sched[static_cast<size_t>(k)];
                    if (opt.enable_term_scheduler && tier_k > 2) {
                        continue;
                    }

                    int span = opt.backprop_span;
                    if (opt.enable_term_scheduler) {
                        span = span_for_tier(tier_k);
                    }
                    span = std::min(n_max - k, std::max(0, span));
                    if (span <= 0) {
                        continue;
                    }

                    for (int src = k + 1; src <= std::min(n_max, k + span); ++src) {
                        if (!has_res[static_cast<size_t>(src)] ||
                            !has_res[static_cast<size_t>(k)]) {
                            continue;
                        }
                        const double side_src = res[static_cast<size_t>(src)].side();
                        const double side_k = res[static_cast<size_t>(k)].side();
                        if (side_src >= side_k - 1e-12) {
                            continue;
                        }

                        const int to_remove = src - k;
                        const PostCfg cfg_src = res[static_cast<size_t>(src)];
                        std::vector<std::vector<int>> combos;

                        if (to_remove == 1) {
                            combos.reserve(static_cast<size_t>(src));
                            for (int i = 0; i < src; ++i) {
                                combos.push_back({i});
                            }
                        } else if (to_remove == 2 && src <= 50) {
                            combos.reserve(static_cast<size_t>(src * (src - 1) / 2));
                            for (int i = 0; i < src; ++i) {
                                for (int j = i + 1; j < src; ++j) {
                                    combos.push_back({i, j});
                                }
                            }
                        } else {
                            int max_combos_limit = opt.backprop_max_combos;
                            if (opt.enable_term_scheduler) {
                                max_combos_limit = max_combos_for_tier(tier_k);
                            }
                            const int max_combos =
                                std::min(std::max(1, max_combos_limit), src * 3);
                            FastRng rng(static_cast<uint64_t>(k) * 1000ULL +
                                        static_cast<uint64_t>(src));
                            combos.reserve(static_cast<size_t>(max_combos));
                            for (int t = 0; t < max_combos; ++t) {
                                std::unordered_set<int> used;
                                std::vector<int> combo;
                                combo.reserve(static_cast<size_t>(to_remove));
                                for (int r = 0; r < to_remove; ++r) {
                                    int idx = 0;
                                    do {
                                        idx = rng.ri(src);
                                    } while (used.count(idx));
                                    used.insert(idx);
                                    combo.push_back(idx);
                                }
                                std::sort(combo.begin(), combo.end());
                                combos.push_back(std::move(combo));
                            }
                        }

                        double best_side = side_k;
                        PostCfg best_cfg = res[static_cast<size_t>(k)];

#pragma omp parallel
                        {
                            double local_best_side = best_side;
                            PostCfg local_best_cfg = best_cfg;

#pragma omp for schedule(dynamic)
                            for (int ci = 0; ci < static_cast<int>(combos.size()); ++ci) {
                                PostCfg reduced = cfg_src;
                                std::vector<int> to_rem = combos[static_cast<size_t>(ci)];
                                std::sort(to_rem.rbegin(), to_rem.rend());
                                for (int idx : to_rem) {
                                    reduced = reduced.remove_tree(idx, base_poly);
                                }

                                if (!reduced.any_overlap()) {
                                    reduced = tighten_cfg(std::move(reduced),
                                                          base_poly,
                                                          opt,
                                                          tier_k,
                                                          50,
                                                          10,
                                                          80);

                                    if (!reduced.any_overlap() &&
                                        reduced.side() < local_best_side) {
                                        local_best_side = reduced.side();
                                        local_best_cfg = std::move(reduced);
                                    }
                                }
                            }

#pragma omp critical
                            {
                                if (local_best_side < best_side) {
                                    best_side = local_best_side;
                                    best_cfg = std::move(local_best_cfg);
                                }
                            }
                        }

                        if (best_side < side_k - 1e-12) {
                            res[static_cast<size_t>(k)] = std::move(best_cfg);
                            ++backprop_improved;
                            changed = true;
                        }
                    }
                }
            } else {
                const double eps_term = std::max(1e-12, opt.accept_term_eps);

                for (int k = n_max; k >= 3; --k) {
                    if (!has_res[static_cast<size_t>(k)]) {
                        continue;
                    }
                    const int tier_k = tier_sched[static_cast<size_t>(k)];
                    if (opt.enable_term_scheduler && tier_k > 2) {
                        continue;
                    }

                    const int span_k = std::min(n_max - k, std::max(0, span_for_tier(tier_k)));
                    const int max_combos = std::max(1, max_combos_for_tier(tier_k));
                    if (span_k <= 0) {
                        continue;
                    }

                    const double rel_slack = (tier_k == 1) ? 0.05 : 0.03;
                    double cur_best_score = res[static_cast<size_t>(k)].score();
                    PostCfg cur_best_cfg = res[static_cast<size_t>(k)];

                    for (int src = k + 1; src <= std::min(n_max, k + span_k); ++src) {
                        if (!has_res[static_cast<size_t>(src)]) {
                            continue;
                        }

                        const double side_k = cur_best_cfg.side();
                        const double side_src = res[static_cast<size_t>(src)].side();
                        if (side_src > side_k * (1.0 + rel_slack) + 1e-12) {
                            continue;
                        }

                        const int to_remove = src - k;
                        if (to_remove <= 0) {
                            continue;
                        }

                        const PostCfg cfg_src = res[static_cast<size_t>(src)];

                        std::vector<double> protrude;
                        compute_protrude_score(cfg_src, protrude);
                        std::vector<int> idx_order(static_cast<size_t>(src));
                        for (int i = 0; i < src; ++i) {
                            idx_order[static_cast<size_t>(i)] = i;
                        }
                        std::sort(idx_order.begin(),
                                  idx_order.end(),
                                  [&](int a, int b) {
                                      const double pa = protrude[static_cast<size_t>(a)];
                                      const double pb = protrude[static_cast<size_t>(b)];
                                      if (pa != pb) {
                                          return pa > pb;
                                      }
                                      return a < b;
                                  });

                        const int det_pool =
                            std::min(src, std::max(24, to_remove * 12));

                        std::vector<std::vector<int>> combos;
                        combos.reserve(static_cast<size_t>(max_combos));

                        auto add_combo = [&](std::vector<int> combo) {
                            if (static_cast<int>(combo.size()) != to_remove) {
                                return;
                            }
                            std::sort(combo.begin(), combo.end());
                            combo.erase(std::unique(combo.begin(), combo.end()), combo.end());
                            if (static_cast<int>(combo.size()) != to_remove) {
                                return;
                            }
                            if (static_cast<int>(combos.size()) >= max_combos) {
                                return;
                            }
                            combos.push_back(std::move(combo));
                        };

                        if (to_remove == 1) {
                            const int take = std::min(src, max_combos);
                            for (int i = 0; i < take; ++i) {
                                add_combo({idx_order[static_cast<size_t>(i)]});
                            }
                        } else if (det_pool >= to_remove) {
                            std::vector<int> base_combo;
                            base_combo.reserve(static_cast<size_t>(to_remove));
                            for (int i = 0; i < to_remove; ++i) {
                                base_combo.push_back(idx_order[static_cast<size_t>(i)]);
                            }
                            add_combo(base_combo);

                            const int max_shift = std::min(6, det_pool - to_remove);
                            for (int off = 1; off <= max_shift; ++off) {
                                std::vector<int> combo;
                                combo.reserve(static_cast<size_t>(to_remove));
                                for (int i = 0; i < to_remove; ++i) {
                                    combo.push_back(idx_order[static_cast<size_t>(off + i)]);
                                }
                                add_combo(combo);
                            }

                            const int extra = std::min(12, det_pool);
                            for (int j = 1; j < extra && static_cast<int>(combos.size()) < max_combos; ++j) {
                                std::vector<int> combo;
                                combo.reserve(static_cast<size_t>(to_remove));
                                combo.push_back(idx_order[0]);
                                for (int r = 1; r < to_remove; ++r) {
                                    combo.push_back(idx_order[static_cast<size_t>((j + r) % det_pool)]);
                                }
                                add_combo(combo);
                            }
                        }

                        FastRng rng_src(opt.seed +
                                        static_cast<uint64_t>(pass_num) * 100000ULL +
                                        static_cast<uint64_t>(k) * 1000ULL +
                                        static_cast<uint64_t>(src) * 17ULL);

                        while (static_cast<int>(combos.size()) < max_combos) {
                            std::vector<int> combo;
                            combo.reserve(static_cast<size_t>(to_remove));
                            for (int r = 0; r < to_remove; ++r) {
                                int idx = 0;
                                int guard = 0;
                                do {
                                    if (det_pool > 0 && rng_src.rf() < 0.70) {
                                        idx = idx_order[static_cast<size_t>(rng_src.ri(det_pool))];
                                    } else {
                                        idx = rng_src.ri(src);
                                    }
                                    ++guard;
                                } while (std::find(combo.begin(), combo.end(), idx) != combo.end() &&
                                         guard < 32);
                                if (guard >= 32) {
                                    break;
                                }
                                combo.push_back(idx);
                            }
                            if (static_cast<int>(combo.size()) != to_remove) {
                                continue;
                            }
                            std::sort(combo.begin(), combo.end());
                            combo.erase(std::unique(combo.begin(), combo.end()), combo.end());
                            if (static_cast<int>(combo.size()) != to_remove) {
                                continue;
                            }
                            combos.push_back(std::move(combo));
                        }

                        double best_score = cur_best_score;
                        PostCfg best_cfg = cur_best_cfg;

#pragma omp parallel
                        {
                            double local_best_score = best_score;
                            PostCfg local_best_cfg = best_cfg;

#pragma omp for schedule(dynamic)
                            for (int ci = 0; ci < static_cast<int>(combos.size()); ++ci) {
                                PostCfg reduced = cfg_src;
                                std::vector<int> to_rem = combos[static_cast<size_t>(ci)];
                                std::sort(to_rem.rbegin(), to_rem.rend());
                                for (int idx : to_rem) {
                                    reduced = reduced.remove_tree(idx, base_poly);
                                }

                                if (!reduced.any_overlap()) {
                                    reduced = tighten_cfg(std::move(reduced),
                                                          base_poly,
                                                          opt,
                                                          tier_k,
                                                          50,
                                                          10,
                                                          80);

                                    const double sc = reduced.score();
                                    if (!reduced.any_overlap() &&
                                        sc < local_best_score - eps_term) {
                                        local_best_score = sc;
                                        local_best_cfg = std::move(reduced);
                                    }
                                }
                            }

#pragma omp critical
                            {
                                if (local_best_score < best_score - eps_term) {
                                    best_score = local_best_score;
                                    best_cfg = std::move(local_best_cfg);
                                }
                            }
                        }

                        if (best_score < cur_best_score - eps_term) {
                            cur_best_score = best_score;
                            cur_best_cfg = std::move(best_cfg);
                            ++backprop_improved;
                            changed = true;
                        }
                    }

                    if (cur_best_score < res[static_cast<size_t>(k)].score() - eps_term) {
                        res[static_cast<size_t>(k)] = std::move(cur_best_cfg);
                    }
                }
            }
        }
    }

    double final_score = 0.0;
    for (int n = 1; n <= n_max; ++n) {
        PostCfg& c = res[static_cast<size_t>(n)];
        if (!has_res[static_cast<size_t>(n)] || c.n != n) {
            solutions_by_n[static_cast<size_t>(n)] = original[static_cast<size_t>(n)];
            const double term = score_instance(base_poly, solutions_by_n[static_cast<size_t>(n)]);
            final_score += term;
            if (stats_out && have_initial_term_sched) {
                const int t = initial_term_sched.tier[static_cast<size_t>(n)];
                if (t == 1) {
                    stats_out->term_summary.tier_a_term_after += term;
                } else if (t == 2) {
                    stats_out->term_summary.tier_b_term_after += term;
                }
            }
            continue;
        }

        std::vector<TreePose> quantized =
            quantize_poses_wrap_deg(c.poses, kSubmissionDecimals);
        const double base_term = base_cfg[static_cast<size_t>(n)].score();
        const double q_term = score_instance(base_poly, quantized);

        if (any_overlap(base_poly, quantized) ||
            q_term > base_term + 1e-12) {
            solutions_by_n[static_cast<size_t>(n)] = original[static_cast<size_t>(n)];
            const double term = score_instance(base_poly, solutions_by_n[static_cast<size_t>(n)]);
            final_score += term;
            if (stats_out && have_initial_term_sched) {
                const int t = initial_term_sched.tier[static_cast<size_t>(n)];
                if (t == 1) {
                    stats_out->term_summary.tier_a_term_after += term;
                } else if (t == 2) {
                    stats_out->term_summary.tier_b_term_after += term;
                }
            }
            continue;
        }

        solutions_by_n[static_cast<size_t>(n)] = std::move(quantized);
        final_score += q_term;
        if (stats_out && have_initial_term_sched) {
            const int t = initial_term_sched.tier[static_cast<size_t>(n)];
            if (t == 1) {
                stats_out->term_summary.tier_a_term_after += q_term;
            } else if (t == 2) {
                stats_out->term_summary.tier_b_term_after += q_term;
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    const double elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() /
        1000.0;

    if (stats_out) {
        stats_out->initial_score = init_score;
        stats_out->final_score = final_score;
        stats_out->phase1_improved = phase1_improved;
        stats_out->backprop_improved = backprop_improved;
        stats_out->elapsed_sec = elapsed;
    }

    return true;
}
