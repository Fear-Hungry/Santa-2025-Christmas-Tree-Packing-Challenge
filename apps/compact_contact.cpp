#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "collision.hpp"
#include "geom.hpp"
#include "spatial_grid.hpp"
#include "submission_io.hpp"

namespace {

struct Extents {
    double min_x;
    double max_x;
    double min_y;
    double max_y;
};

struct Options {
    std::string base_path;
    std::string output_path = "runs/tmp/compact_contact.csv";
    int n_min = 1;
    int n_max = 200;
    int target_top = 0;
    std::vector<int> target_ns;
    int target_range_min = 1;
    int target_range_max = 200;
    bool target_range_set = false;
    uint64_t seed = 1;
    int passes = 30;
    int attempts_per_pass = 48;
    int patience = 4;
    int boundary_topk = 12;
    int push_bisect_iters = 10;
    double push_max_step_frac = 1.0;
    double plateau_eps = 0.0;
    bool alt_axis = true;
    double diag_frac = 0.0;
    double diag_rand = 0.0;
    double center_bias = 0.0;
    double interior_prob = 0.0;
    bool final_rigid = true;
    int quantize_decimals = 9;
};

Extents compute_extents(const std::vector<BoundingBox>& bbs) {
    Extents e{std::numeric_limits<double>::infinity(),
              -std::numeric_limits<double>::infinity(),
              std::numeric_limits<double>::infinity(),
              -std::numeric_limits<double>::infinity()};
    for (const auto& bb : bbs) {
        e.min_x = std::min(e.min_x, bb.min_x);
        e.max_x = std::max(e.max_x, bb.max_x);
        e.min_y = std::min(e.min_y, bb.min_y);
        e.max_y = std::max(e.max_y, bb.max_y);
    }
    return e;
}

double side_from_extents(const Extents& e) {
    return std::max(e.max_x - e.min_x, e.max_y - e.min_y);
}

double min_dim_from_extents(const Extents& e) {
    return std::min(e.max_x - e.min_x, e.max_y - e.min_y);
}

bool aabb_overlap(const BoundingBox& a, const BoundingBox& b) {
    if (a.max_x < b.min_x || b.max_x < a.min_x) {
        return false;
    }
    if (a.max_y < b.min_y || b.max_y < a.min_y) {
        return false;
    }
    return true;
}

std::vector<BoundingBox> bounding_boxes_for_poses(const Polygon& base_poly,
                                                  const std::vector<TreePose>& poses) {
    std::vector<BoundingBox> bbs;
    bbs.reserve(poses.size());
    for (const auto& pose : poses) {
        bbs.push_back(bounding_box(transform_polygon(base_poly, pose)));
    }
    return bbs;
}

std::vector<int> build_extreme_pool(const std::vector<BoundingBox>& bbs, int topk) {
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

double score_term_for_n(const Polygon& base_poly, const std::vector<TreePose>& poses) {
    if (poses.empty()) {
        return 0.0;
    }
    std::vector<Polygon> polys = transformed_polygons(base_poly, poses);
    const double side = bounding_square_side(polys);
    return (side * side) / static_cast<double>(poses.size());
}

Point normalize_dir(const Point& p) {
    const double norm = std::hypot(p.x, p.y);
    if (!(norm > 1e-12)) {
        return Point{0.0, 0.0};
    }
    return Point{p.x / norm, p.y / norm};
}

bool push_to_contact_dir(const Polygon& base_poly,
                         const std::vector<TreePose>& poses,
                         const std::vector<Polygon>& polys,
                         const std::vector<BoundingBox>& bbs,
                         const UniformGridIndex& grid,
                         int idx,
                         const Point& dir_in,
                         double max_step,
                         int bisect_iters,
                         double thr_sq,
                         int quantize_decimals,
                         TreePose& pose_out,
                         Polygon& poly_out,
                         BoundingBox& bb_out,
                         double& delta_out) {
    Point dir = normalize_dir(dir_in);
    if (!(max_step > 1e-12) || (std::abs(dir.x) < 1e-12 && std::abs(dir.y) < 1e-12)) {
        return false;
    }

    const TreePose& base_pose = poses[static_cast<size_t>(idx)];
    double t_max = max_step;
    if (dir.x > 1e-12) {
        t_max = std::min(t_max, (100.0 - base_pose.x) / dir.x);
    } else if (dir.x < -1e-12) {
        t_max = std::min(t_max, (-100.0 - base_pose.x) / dir.x);
    }
    if (dir.y > 1e-12) {
        t_max = std::min(t_max, (100.0 - base_pose.y) / dir.y);
    } else if (dir.y < -1e-12) {
        t_max = std::min(t_max, (-100.0 - base_pose.y) / dir.y);
    }

    if (!(t_max > 1e-12)) {
        return false;
    }

    std::vector<int> neigh;
    neigh.reserve(32);

    auto valid_at = [&](double delta,
                        TreePose& cand_pose,
                        Polygon& cand_poly,
                        BoundingBox& cand_bb) -> bool {
        cand_pose = base_pose;
        cand_pose.x += dir.x * delta;
        cand_pose.y += dir.y * delta;
        cand_pose = quantize_pose(cand_pose, quantize_decimals);
        if (cand_pose.x < -100.0 || cand_pose.x > 100.0 ||
            cand_pose.y < -100.0 || cand_pose.y > 100.0) {
            return false;
        }

        cand_poly = transform_polygon(base_poly, cand_pose);
        cand_bb = bounding_box(cand_poly);

        grid.gather(cand_pose.x, cand_pose.y, neigh);
        for (int j : neigh) {
            if (j == idx) {
                continue;
            }
            const double dx = cand_pose.x - poses[static_cast<size_t>(j)].x;
            const double dy = cand_pose.y - poses[static_cast<size_t>(j)].y;
            if (dx * dx + dy * dy > thr_sq) {
                continue;
            }
            if (!aabb_overlap(cand_bb, bbs[static_cast<size_t>(j)])) {
                continue;
            }
            if (polygons_intersect(cand_poly, polys[static_cast<size_t>(j)])) {
                return false;
            }
        }
        return true;
    };

    TreePose best_pose;
    Polygon best_poly;
    BoundingBox best_bb;
    double best_delta = 0.0;

    TreePose pose_hi;
    Polygon poly_hi;
    BoundingBox bb_hi;
    if (valid_at(t_max, pose_hi, poly_hi, bb_hi)) {
        best_pose = pose_hi;
        best_poly = std::move(poly_hi);
        best_bb = bb_hi;
        best_delta = t_max;
    } else {
        double hi_invalid = t_max;
        double lo_valid = 0.0;
        TreePose pose_lo;
        Polygon poly_lo;
        BoundingBox bb_lo;
        bool found = false;
        double step = t_max;
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

        for (int it = 0; it < bisect_iters; ++it) {
            const double mid = 0.5 * (lo + hi);
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

    if (!(best_delta > 1e-12)) {
        return false;
    }

    pose_out = best_pose;
    poly_out = std::move(best_poly);
    bb_out = best_bb;
    delta_out = best_delta;
    return true;
}

Options parse_args(int argc, char** argv) {
    Options opt;
    auto parse_int = [](const std::string& s) -> int {
        size_t pos = 0;
        int v = std::stoi(s, &pos);
        if (pos != s.size()) {
            throw std::runtime_error("Inteiro invalido: " + s);
        }
        return v;
    };
    auto parse_u64 = [](const std::string& s) -> uint64_t {
        size_t pos = 0;
        uint64_t v = std::stoull(s, &pos);
        if (pos != s.size()) {
            throw std::runtime_error("uint64 invalido: " + s);
        }
        return v;
    };
    auto parse_double = [](const std::string& s) -> double {
        size_t pos = 0;
        double v = std::stod(s, &pos);
        if (pos != s.size()) {
            throw std::runtime_error("Double invalido: " + s);
        }
        return v;
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Faltou valor para " + name);
            }
            return argv[++i];
        };

        if (arg == "--base") {
            opt.base_path = need(arg);
        } else if (arg == "--out" || arg == "--output") {
            opt.output_path = need(arg);
        } else if (arg == "--target-top") {
            opt.target_top = parse_int(need(arg));
        } else if (arg == "--target-n") {
            std::string s = need(arg);
            std::stringstream ss(s);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (item.empty()) {
                    continue;
                }
                opt.target_ns.push_back(parse_int(item));
            }
        } else if (arg == "--target-range") {
            std::string s = need(arg);
            std::stringstream ss(s);
            std::string a, b;
            if (!std::getline(ss, a, ',') || !std::getline(ss, b, ',')) {
                throw std::runtime_error("--target-range precisa ser 'a,b'.");
            }
            opt.target_range_min = parse_int(a);
            opt.target_range_max = parse_int(b);
            opt.target_range_set = true;
        } else if (arg == "--n-min") {
            opt.n_min = parse_int(need(arg));
        } else if (arg == "--n-max") {
            opt.n_max = parse_int(need(arg));
        } else if (arg == "--seed") {
            opt.seed = parse_u64(need(arg));
        } else if (arg == "--passes") {
            opt.passes = parse_int(need(arg));
        } else if (arg == "--attempts-per-pass") {
            opt.attempts_per_pass = parse_int(need(arg));
        } else if (arg == "--patience") {
            opt.patience = parse_int(need(arg));
        } else if (arg == "--boundary-topk") {
            opt.boundary_topk = parse_int(need(arg));
        } else if (arg == "--push-bisect-iters") {
            opt.push_bisect_iters = parse_int(need(arg));
        } else if (arg == "--push-max-step-frac") {
            opt.push_max_step_frac = parse_double(need(arg));
        } else if (arg == "--plateau-eps") {
            opt.plateau_eps = parse_double(need(arg));
        } else if (arg == "--diag-frac") {
            opt.diag_frac = parse_double(need(arg));
        } else if (arg == "--diag-rand") {
            opt.diag_rand = parse_double(need(arg));
        } else if (arg == "--center-bias") {
            opt.center_bias = parse_double(need(arg));
        } else if (arg == "--interior-prob") {
            opt.interior_prob = parse_double(need(arg));
        } else if (arg == "--no-alt-axis") {
            opt.alt_axis = false;
        } else if (arg == "--no-final-rigid") {
            opt.final_rigid = false;
        } else if (arg == "--quantize-decimals") {
            opt.quantize_decimals = parse_int(need(arg));
        } else {
            throw std::runtime_error("Flag desconhecida: " + arg);
        }
    }

    if (opt.base_path.empty()) {
        throw std::runtime_error("Use --base <csv>.");
    }
    if (opt.n_min < 1 || opt.n_min > opt.n_max || opt.n_max > 200) {
        throw std::runtime_error("--n-min/--n-max invalidos.");
    }
    if (opt.target_top < 0) {
        throw std::runtime_error("--target-top precisa ser >= 0.");
    }
    for (int n : opt.target_ns) {
        if (n < 1 || n > 200) {
            throw std::runtime_error("--target-n fora de [1,200].");
        }
    }
    if (opt.target_range_set) {
        if (opt.target_range_min < 1 || opt.target_range_max > 200 ||
            opt.target_range_min > opt.target_range_max) {
            throw std::runtime_error("--target-range invalido.");
        }
    }
    if (opt.passes < 0 || opt.attempts_per_pass < 0 || opt.patience < 0) {
        throw std::runtime_error("--passes/--attempts-per-pass/--patience invalidos.");
    }
    if (opt.boundary_topk <= 0) {
        throw std::runtime_error("--boundary-topk precisa ser > 0.");
    }
    if (opt.push_bisect_iters <= 0) {
        throw std::runtime_error("--push-bisect-iters precisa ser > 0.");
    }
    if (!(opt.push_max_step_frac > 0.0)) {
        throw std::runtime_error("--push-max-step-frac precisa ser > 0.");
    }
    if (opt.plateau_eps < 0.0) {
        throw std::runtime_error("--plateau-eps precisa ser >= 0.");
    }
    if (opt.diag_frac < 0.0) {
        throw std::runtime_error("--diag-frac precisa ser >= 0.");
    }
    if (opt.diag_rand < 0.0) {
        throw std::runtime_error("--diag-rand precisa ser >= 0.");
    }
    if (opt.center_bias < 0.0) {
        throw std::runtime_error("--center-bias precisa ser >= 0.");
    }
    if (opt.interior_prob < 0.0 || opt.interior_prob > 1.0) {
        throw std::runtime_error("--interior-prob precisa estar em [0,1].");
    }
    if (opt.quantize_decimals < 0) {
        throw std::runtime_error("--quantize-decimals precisa ser >= 0.");
    }

    bool has_target = (opt.target_top > 0) || !opt.target_ns.empty() || opt.target_range_set;
    if (!has_target) {
        throw std::runtime_error("Com --base, especifique --target-top/--target-n/--target-range.");
    }

    return opt;
}

std::vector<char> build_target_mask(const Options& opt,
                                    const Polygon& base_poly,
                                    const SubmissionPoses& base) {
    std::vector<char> target(static_cast<size_t>(opt.n_max + 1), 0);

    if (opt.target_top > 0) {
        struct TermRow {
            int n;
            double term;
        };
        std::vector<TermRow> rows;
        rows.reserve(static_cast<size_t>(opt.n_max));
        for (int n = opt.n_min; n <= opt.n_max; ++n) {
            const auto& poses = base.by_n[static_cast<size_t>(n)];
            if (static_cast<int>(poses.size()) != n) {
                continue;
            }
            const double term = score_term_for_n(base_poly, poses);
            rows.push_back(TermRow{n, term});
        }
        std::sort(rows.begin(), rows.end(), [](const TermRow& a, const TermRow& b) {
            return a.term > b.term;
        });
        const int topk = std::min(opt.target_top, static_cast<int>(rows.size()));
        for (int i = 0; i < topk; ++i) {
            target[static_cast<size_t>(rows[static_cast<size_t>(i)].n)] = 1;
        }
    }

    if (opt.target_range_set) {
        for (int n = opt.target_range_min; n <= opt.target_range_max; ++n) {
            if (n >= opt.n_min && n <= opt.n_max) {
                target[static_cast<size_t>(n)] = 1;
            }
        }
    }

    for (int n : opt.target_ns) {
        if (n >= opt.n_min && n <= opt.n_max) {
            target[static_cast<size_t>(n)] = 1;
        }
    }

    return target;
}

struct CompactResult {
    std::vector<TreePose> poses;
    double side_before = 0.0;
    double side_after = 0.0;
    bool ok = false;
};

CompactResult compact_contact(const Polygon& base_poly,
                              const std::vector<TreePose>& input,
                              const Options& opt,
                              std::mt19937_64& rng) {
    CompactResult result;
    if (input.empty()) {
        result.poses = input;
        result.ok = true;
        return result;
    }

    std::vector<TreePose> poses = quantize_poses(input, opt.quantize_decimals);
    std::vector<Polygon> polys = transformed_polygons(base_poly, poses);
    std::vector<BoundingBox> bbs = bounding_boxes_for_poses(base_poly, poses);

    Extents curr_ext = compute_extents(bbs);
    result.side_before = side_from_extents(curr_ext);

    const double radius = enclosing_circle_radius(base_poly);
    const double thr = 2.0 * radius + 1e-9;
    const double thr_sq = thr * thr;
    UniformGridIndex grid(static_cast<int>(poses.size()), thr);
    grid.rebuild(poses);

    auto cost_from_extents = [&](const Extents& e) -> double {
        const double side = side_from_extents(e);
        const double min_dim = min_dim_from_extents(e);
        return side + opt.plateau_eps * min_dim;
    };

    double curr_cost = cost_from_extents(curr_ext);
    int no_improve_passes = 0;

    std::uniform_int_distribution<int> coin(0, 1);
    std::uniform_real_distribution<double> uni01(0.0, 1.0);

    for (int pass = 0; pass < opt.passes; ++pass) {
        const double width = curr_ext.max_x - curr_ext.min_x;
        const double height = curr_ext.max_y - curr_ext.min_y;
        bool axis_x = (width >= height);
        if (opt.alt_axis && (pass % 2 == 1)) {
            axis_x = !axis_x;
        }
        const double center_x = 0.5 * (curr_ext.min_x + curr_ext.max_x);
        const double center_y = 0.5 * (curr_ext.min_y + curr_ext.max_y);

        std::vector<int> pool = build_extreme_pool(bbs, opt.boundary_topk);
        std::vector<int> candidates;
        candidates.reserve(pool.size());
        const double tol = 1e-9;
        for (int idx : pool) {
            if (axis_x) {
                if (bbs[static_cast<size_t>(idx)].min_x <= curr_ext.min_x + tol ||
                    bbs[static_cast<size_t>(idx)].max_x >= curr_ext.max_x - tol) {
                    candidates.push_back(idx);
                }
            } else {
                if (bbs[static_cast<size_t>(idx)].min_y <= curr_ext.min_y + tol ||
                    bbs[static_cast<size_t>(idx)].max_y >= curr_ext.max_y - tol) {
                    candidates.push_back(idx);
                }
            }
        }
        if (candidates.empty()) {
            candidates = pool;
        }

        if (candidates.empty()) {
            break;
        }

        std::uniform_int_distribution<int> pick(0, static_cast<int>(candidates.size()) - 1);
        std::uniform_int_distribution<int> pick_all(
            0, static_cast<int>(poses.size()) - 1);
        bool moved_any = false;

        const double max_step = opt.push_max_step_frac * std::max(1e-9, side_from_extents(curr_ext));

        for (int attempt = 0; attempt < opt.attempts_per_pass; ++attempt) {
            int i = candidates[static_cast<size_t>(pick(rng))];
            if (opt.interior_prob > 0.0 && uni01(rng) < opt.interior_prob) {
                i = pick_all(rng);
            }
            const BoundingBox& bb = bbs[static_cast<size_t>(i)];

            Point dir{0.0, 0.0};
            if (axis_x) {
                const bool at_min = bb.min_x <= curr_ext.min_x + tol;
                const bool at_max = bb.max_x >= curr_ext.max_x - tol;
                if (at_min && at_max) {
                    dir.x = (coin(rng) == 0) ? 1.0 : -1.0;
                } else if (at_min) {
                    dir.x = 1.0;
                } else if (at_max) {
                    dir.x = -1.0;
                } else {
                    dir.x = (poses[static_cast<size_t>(i)].x >= center_x) ? -1.0 : 1.0;
                }
                double cross = 0.0;
                if (opt.center_bias > 0.0) {
                    cross += opt.center_bias *
                             (center_y - poses[static_cast<size_t>(i)].y) /
                             std::max(1e-9, height);
                }
                if (opt.diag_frac > 0.0) {
                    cross += (coin(rng) == 0 ? -1.0 : 1.0) * opt.diag_frac;
                }
                if (opt.diag_rand > 0.0) {
                    cross += (2.0 * uni01(rng) - 1.0) * opt.diag_rand;
                }
                dir.y = cross;
            } else {
                const bool at_min = bb.min_y <= curr_ext.min_y + tol;
                const bool at_max = bb.max_y >= curr_ext.max_y - tol;
                if (at_min && at_max) {
                    dir.y = (coin(rng) == 0) ? 1.0 : -1.0;
                } else if (at_min) {
                    dir.y = 1.0;
                } else if (at_max) {
                    dir.y = -1.0;
                } else {
                    dir.y = (poses[static_cast<size_t>(i)].y >= center_y) ? -1.0 : 1.0;
                }
                double cross = 0.0;
                if (opt.center_bias > 0.0) {
                    cross += opt.center_bias *
                             (center_x - poses[static_cast<size_t>(i)].x) /
                             std::max(1e-9, width);
                }
                if (opt.diag_frac > 0.0) {
                    cross += (coin(rng) == 0 ? -1.0 : 1.0) * opt.diag_frac;
                }
                if (opt.diag_rand > 0.0) {
                    cross += (2.0 * uni01(rng) - 1.0) * opt.diag_rand;
                }
                dir.x = cross;
            }

            TreePose cand_pose;
            Polygon cand_poly;
            BoundingBox cand_bb;
            double cand_delta = 0.0;
            if (!push_to_contact_dir(base_poly, poses, polys, bbs, grid, i, dir, max_step,
                                     opt.push_bisect_iters, thr_sq, opt.quantize_decimals,
                                     cand_pose, cand_poly, cand_bb, cand_delta)) {
                continue;
            }

            const TreePose old_pose = poses[static_cast<size_t>(i)];
            Polygon old_poly = polys[static_cast<size_t>(i)];
            const BoundingBox old_bb = bbs[static_cast<size_t>(i)];

            poses[static_cast<size_t>(i)] = cand_pose;
            polys[static_cast<size_t>(i)] = std::move(cand_poly);
            bbs[static_cast<size_t>(i)] = cand_bb;

            Extents new_ext = compute_extents(bbs);
            const double new_cost = cost_from_extents(new_ext);

            if (new_cost <= curr_cost + 1e-15) {
                grid.update_position(i, cand_pose.x, cand_pose.y);
                curr_ext = new_ext;
                curr_cost = new_cost;
                moved_any = true;
            } else {
                poses[static_cast<size_t>(i)] = old_pose;
                polys[static_cast<size_t>(i)] = std::move(old_poly);
                bbs[static_cast<size_t>(i)] = old_bb;
            }
        }

        if (!moved_any) {
            no_improve_passes++;
            if (no_improve_passes >= opt.patience) {
                break;
            }
        } else {
            no_improve_passes = 0;
        }
    }

    if (opt.final_rigid) {
        std::vector<TreePose> rotated = poses;
        optimize_rigid_rotation(base_poly, rotated);
        rotated = quantize_poses_wrap_deg(rotated, opt.quantize_decimals);
        if (!any_overlap(base_poly, rotated)) {
            poses = std::move(rotated);
        }
    }

    poses = quantize_poses_wrap_deg(poses, opt.quantize_decimals);

    std::vector<Polygon> final_polys = transformed_polygons(base_poly, poses);
    result.side_after = bounding_square_side(final_polys);
    result.poses = std::move(poses);
    result.ok = true;
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);

        SubmissionPoses base = load_submission_poses(opt.base_path, opt.n_max);
        if (static_cast<int>(base.by_n.size()) <= opt.n_max) {
            throw std::runtime_error("Submission incompleta para n_max.");
        }

        Polygon base_poly = get_tree_polygon();
        std::vector<char> target = build_target_mask(opt, base_poly, base);

        std::vector<std::vector<TreePose>> out = base.by_n;
        std::mt19937_64 rng(opt.seed);

        for (int n = opt.n_min; n <= opt.n_max; ++n) {
            if (!target[static_cast<size_t>(n)]) {
                continue;
            }
            auto& poses = out[static_cast<size_t>(n)];
            if (static_cast<int>(poses.size()) != n) {
                continue;
            }

            CompactResult res = compact_contact(base_poly, poses, opt, rng);
            if (!res.ok) {
                continue;
            }
            const double before_term = (res.side_before * res.side_before) / static_cast<double>(n);
            const double after_term = (res.side_after * res.side_after) / static_cast<double>(n);
            std::cout << "n=" << n
                      << " side " << res.side_before << " -> " << res.side_after
                      << " term " << before_term << " -> " << after_term << "\n";
            poses = std::move(res.poses);
        }

        std::ofstream out_file(opt.output_path);
        if (!out_file) {
            throw std::runtime_error("Erro ao abrir arquivo de saida: " + opt.output_path);
        }
        out_file << "id,x,y,deg\n";
        for (int n = 1; n <= opt.n_max; ++n) {
            const auto& sol = out[static_cast<size_t>(n)];
            if (static_cast<int>(sol.size()) != n) {
                throw std::runtime_error("Solucao invalida para n=" + std::to_string(n));
            }
            for (int i = 0; i < n; ++i) {
                const auto& pose = sol[static_cast<size_t>(i)];
                out_file << fmt_submission_id(n, i) << ","
                         << fmt_submission_value(pose.x) << ","
                         << fmt_submission_value(pose.y) << ","
                         << fmt_submission_value(pose.deg) << "\n";
            }
        }

        std::cout << "Submission gerada em " << opt.output_path << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Erro no compact_contact: " << ex.what() << "\n";
        return 1;
    }
}
