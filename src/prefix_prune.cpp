#include "prefix_prune.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "collision.hpp"
#include "sa.hpp"
#include "submission_io.hpp"

namespace {

Extents compute_extents(const std::vector<BoundingBox>& bbs) {
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

double side_from_extents(const Extents& e) {
    return std::max(e.max_x - e.min_x, e.max_y - e.min_y);
}

std::vector<int> boundary_indices(const std::vector<BoundingBox>& bbs,
                                  const Extents& e,
                                  double tol) {
    std::vector<int> out;
    out.reserve(bbs.size());
    for (size_t i = 0; i < bbs.size(); ++i) {
        const auto& bb = bbs[i];
        if (bb.min_x <= e.min_x + tol || bb.max_x >= e.max_x - tol ||
            bb.min_y <= e.min_y + tol || bb.max_y >= e.max_y - tol) {
            out.push_back(static_cast<int>(i));
        }
    }
    return out;
}

ILSResult ils_basin_hop_compact_impl(const Polygon& base_poly,
                                     double radius,
                                     std::vector<TreePose>& poses,
                                     uint64_t seed,
                                     const Options& opt) {
    ILSResult out;
    if (opt.ils_iters <= 0 || poses.empty()) {
        return out;
    }

    const int n = static_cast<int>(poses.size());

    auto side_from_poses = [&](const std::vector<TreePose>& p) -> double {
        std::vector<BoundingBox> bbs = bounding_boxes_for_poses(base_poly, p);
        return side_from_extents(compute_extents(bbs));
    };

    poses = quantize_poses_wrap_deg(poses);
    if (any_overlap(base_poly, poses, radius)) {
        throw std::runtime_error("ILS: start contém overlap (precisa ser solução válida).");
    }

    const double start_side = side_from_poses(poses);
    out.start_side = start_side;
    out.best_side = start_side;

    std::vector<TreePose> best = poses;
    std::vector<TreePose> curr = poses;
    double best_side = start_side;
    double curr_side = start_side;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni01(0.0, 1.0);
    std::uniform_real_distribution<double> uni_alpha(opt.ils_alpha_min, opt.ils_alpha_max);
    std::uniform_real_distribution<double> uni_shear(-opt.ils_shear_max, opt.ils_shear_max);
    std::uniform_real_distribution<double> uni_rot(-opt.ils_rot_deg_max, opt.ils_rot_deg_max);
    std::normal_distribution<double> normal(0.0, 1.0);

    auto ils_temp_at = [&](int it) -> double {
        if (!opt.ils_accept_sa) {
            return 0.0;
        }
        const double t0 = opt.ils_t0;
        const double t1 = opt.ils_t1;
        if (!(t0 > 0.0) || !(t1 > 0.0)) {
            return 0.0;
        }
        if (opt.ils_iters <= 1) {
            return t1;
        }
        double frac = static_cast<double>(it) / static_cast<double>(opt.ils_iters - 1);
        return t0 * std::pow(t1 / t0, frac);
    };

    auto accept_worse = [&](double old_cost, double new_cost, double T) -> bool {
        double delta = new_cost - old_cost;
        if (delta <= 0.0) {
            return true;
        }
        if (!(T > 0.0)) {
            return false;
        }
        double rel = delta / std::max(1e-12, old_cost);
        double prob = std::exp(-rel / std::max(1e-12, T));
        return (uni01(rng) < prob);
    };

    auto apply_shake = [&](std::vector<TreePose>& p) -> bool {
        std::vector<BoundingBox> bbs = bounding_boxes_for_poses(base_poly, p);
        Extents e = compute_extents(bbs);
        if (!std::isfinite(e.min_x) || !std::isfinite(e.max_x) || !std::isfinite(e.min_y) ||
            !std::isfinite(e.max_y)) {
            return false;
        }
        const double cx = 0.5 * (e.min_x + e.max_x);
        const double cy = 0.5 * (e.min_y + e.max_y);
        const double width = e.max_x - e.min_x;
        const double height = e.max_y - e.min_y;
        const double base_side = std::max(width, height);
        const bool dom_x = (width >= height);

        double alpha = uni_alpha(rng);
        double ax = alpha;
        double ay = alpha;
        if (uni01(rng) < opt.ils_p_aniso) {
            if (dom_x) {
                const double ratio = (width > 1e-12) ? (height / width) : 1.0;
                ax = alpha;
                ay = 1.0 - (1.0 - alpha) * ratio;
            } else {
                const double ratio = (height > 1e-12) ? (width / height) : 1.0;
                ay = alpha;
                ax = 1.0 - (1.0 - alpha) * ratio;
            }
        }

        const double shx = (opt.ils_shear_max > 0.0) ? uni_shear(rng) : 0.0;
        const double shy = (opt.ils_shear_max > 0.0) ? uni_shear(rng) : 0.0;

        const double jitter = opt.ils_jitter_frac * std::max(1e-9, base_side);

        std::vector<int> idx(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            idx[static_cast<size_t>(i)] = i;
        }
        std::shuffle(idx.begin(), idx.end(), rng);
        int subset = static_cast<int>(std::round(opt.ils_subset_frac * static_cast<double>(n)));
        subset = std::max(0, std::min(n, subset));

        for (int i = 0; i < n; ++i) {
            TreePose& pose = p[static_cast<size_t>(i)];
            const double dx = pose.x - cx;
            const double dy = pose.y - cy;
            const double sx = ax * dx;
            const double sy = ay * dy;
            pose.x = cx + sx + shx * sy;
            pose.y = cy + sy + shy * sx;
        }

        for (int k = 0; k < subset; ++k) {
            const int i = idx[static_cast<size_t>(k)];
            TreePose& pose = p[static_cast<size_t>(i)];
            pose.x += normal(rng) * jitter;
            pose.y += normal(rng) * jitter;
            if (opt.ils_rot_deg_max > 0.0 && uni01(rng) < opt.ils_rot_prob) {
                pose.deg += uni_rot(rng);
            }
        }

        p = quantize_poses_wrap_deg(p);
        for (const auto& pose : p) {
            if (pose.x < -100.0 || pose.x > 100.0 || pose.y < -100.0 || pose.y > 100.0) {
                return false;
            }
        }
        return true;
    };

    auto aabb_overlap = [](const BoundingBox& a, const BoundingBox& b) -> bool {
        if (a.max_x < b.min_x || b.max_x < a.min_x) {
            return false;
        }
        if (a.max_y < b.min_y || b.max_y < a.min_y) {
            return false;
        }
        return true;
    };

    auto repair_mtv = [&](std::vector<TreePose>& p, uint64_t rseed) -> bool {
        if (opt.ils_repair_mtv_passes <= 0) {
            return !any_overlap(base_poly, p, radius);
        }

        const double thr = 2.0 * radius + 1e-9;
        const double limit_sq = thr * thr;

        std::vector<Point> centers;
        centers.reserve(p.size());
        std::vector<Polygon> polys;
        polys.reserve(p.size());
        std::vector<BoundingBox> bbs;
        bbs.reserve(p.size());
        for (const auto& pose : p) {
            centers.push_back(Point{pose.x, pose.y});
            Polygon poly = transform_polygon(base_poly, pose);
            polys.push_back(std::move(poly));
            bbs.push_back(bounding_box(polys.back()));
        }

        auto find_first = [&]() -> std::pair<int, int> {
            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    const double dx =
                        centers[static_cast<size_t>(i)].x - centers[static_cast<size_t>(j)].x;
                    const double dy =
                        centers[static_cast<size_t>(i)].y - centers[static_cast<size_t>(j)].y;
                    if (dx * dx + dy * dy > limit_sq) {
                        continue;
                    }
                    if (!aabb_overlap(bbs[static_cast<size_t>(i)],
                                      bbs[static_cast<size_t>(j)])) {
                        continue;
                    }
                    if (polygons_intersect(polys[static_cast<size_t>(i)],
                                           polys[static_cast<size_t>(j)])) {
                        return {i, j};
                    }
                }
            }
            return {-1, -1};
        };

        auto centrality_key = [&](const TreePose& pose) -> double {
            return std::max(std::abs(pose.x), std::abs(pose.y));
        };

        auto pick_idx = [&](int a, int b) -> int {
            double ka = centrality_key(p[static_cast<size_t>(a)]);
            double kb = centrality_key(p[static_cast<size_t>(b)]);
            return (ka < kb) ? a : b;
        };

        SARefiner sep(base_poly, radius);
        std::mt19937_64 rrng(rseed);
        std::uniform_real_distribution<double> u01(0.0, 1.0);

        const bool split = (opt.ils_repair_mtv_split > 0.0 && opt.ils_repair_mtv_split < 1.0);
        for (int pass = 0; pass < opt.ils_repair_mtv_passes; ++pass) {
            auto [i, j] = find_first();
            if (i < 0) {
                return true;
            }

            int idx = pick_idx(i, j);
            int other = (idx == i) ? j : i;

            Point mtv{0.0, 0.0};
            double ov_area = 0.0;
            if (!sep.overlap_mtv(p[static_cast<size_t>(idx)],
                                 p[static_cast<size_t>(other)],
                                 mtv,
                                 ov_area) ||
                !(std::hypot(mtv.x, mtv.y) > 1e-12)) {
                double dx = p[static_cast<size_t>(idx)].x - p[static_cast<size_t>(other)].x;
                double dy = p[static_cast<size_t>(idx)].y - p[static_cast<size_t>(other)].y;
                double norm = std::hypot(dx, dy);
                if (!(norm > 1e-12)) {
                    double ang = u01(rrng) * 2.0 * 3.14159265358979323846;
                    dx = std::cos(ang);
                    dy = std::sin(ang);
                    norm = 1.0;
                }
                mtv.x = dx / norm;
                mtv.y = dy / norm;
                const double step = std::max(1e-6, 1e-3 * radius);
                mtv.x *= step;
                mtv.y *= step;
            }

            mtv.x *= opt.ils_repair_mtv_damping;
            mtv.y *= opt.ils_repair_mtv_damping;

            double frac_idx = 1.0;
            double frac_other = 0.0;
            if (split) {
                frac_idx = opt.ils_repair_mtv_split;
                frac_other = 1.0 - opt.ils_repair_mtv_split;
            }

            double scale = 1.0;
            bool moved = false;
            for (int bt = 0; bt < 12 && !moved; ++bt) {
                TreePose cand_idx = p[static_cast<size_t>(idx)];
                TreePose cand_other = p[static_cast<size_t>(other)];
                cand_idx.x += mtv.x * frac_idx * scale;
                cand_idx.y += mtv.y * frac_idx * scale;
                if (split) {
                    cand_other.x -= mtv.x * frac_other * scale;
                    cand_other.y -= mtv.y * frac_other * scale;
                }

                cand_idx = quantize_pose_wrap_deg(cand_idx);
                if (split) {
                    cand_other = quantize_pose_wrap_deg(cand_other);
                }

                if (cand_idx.x < -100.0 || cand_idx.x > 100.0 || cand_idx.y < -100.0 ||
                    cand_idx.y > 100.0) {
                    scale *= 0.5;
                    continue;
                }
                if (split && (cand_other.x < -100.0 || cand_other.x > 100.0 ||
                              cand_other.y < -100.0 || cand_other.y > 100.0)) {
                    scale *= 0.5;
                    continue;
                }

                p[static_cast<size_t>(idx)] = cand_idx;
                centers[static_cast<size_t>(idx)] = Point{cand_idx.x, cand_idx.y};
                polys[static_cast<size_t>(idx)] = transform_polygon(base_poly, cand_idx);
                bbs[static_cast<size_t>(idx)] = bounding_box(polys[static_cast<size_t>(idx)]);

                if (split) {
                    p[static_cast<size_t>(other)] = cand_other;
                    centers[static_cast<size_t>(other)] = Point{cand_other.x, cand_other.y};
                    polys[static_cast<size_t>(other)] = transform_polygon(base_poly, cand_other);
                    bbs[static_cast<size_t>(other)] =
                        bounding_box(polys[static_cast<size_t>(other)]);
                }

                moved = true;
            }

            if (!moved) {
                break;
            }
        }

        return (find_first().first < 0);
    };

    SARefiner sa(base_poly, radius);

    for (int it = 0; it < opt.ils_iters; ++it) {
        std::vector<TreePose> cand = curr;
        if (!apply_shake(cand)) {
            continue;
        }

        out.attempts += 1;

        uint64_t cand_seed =
            seed ^
            (0xD1B54A32D192ED03ULL + static_cast<uint64_t>(it) * 0x9E3779B97F4A7C15ULL);

        if (!repair_mtv(cand, cand_seed)) {
            continue;
        }
        if (any_overlap(base_poly, cand, radius)) {
            continue;
        }

        if (opt.ils_sa_iters > 0 && opt.ils_sa_restarts > 0) {
            SARefiner::Params p;
            p.iters = opt.ils_sa_iters;
            p.w_micro = opt.sa_w_micro;
            p.w_swap_rot = opt.sa_w_swap_rot;
            p.w_relocate = opt.sa_w_relocate;
            p.w_block_translate = opt.sa_w_block_translate;
            p.w_block_rotate = opt.sa_w_block_rotate;
            p.w_lns = opt.sa_w_lns;
            p.w_push_contact = opt.sa_w_push_contact;
            p.w_squeeze = opt.sa_w_squeeze;
            p.block_size = opt.sa_block_size;
            p.lns_remove = opt.sa_lns_remove;
            p.lns_candidates = opt.sa_lns_candidates;
            p.lns_eval_attempts_per_tree = opt.sa_lns_eval_attempts_per_tree;
            p.hh_segment = opt.sa_hh_segment;
            p.hh_reaction = opt.sa_hh_reaction;
            p.overlap_metric = opt.sa_overlap_metric;
            p.overlap_weight = opt.sa_overlap_weight;
            p.overlap_weight_start = opt.sa_overlap_weight_start;
            p.overlap_weight_end = opt.sa_overlap_weight_end;
            p.overlap_weight_power = opt.sa_overlap_weight_power;
            p.overlap_weight_geometric = opt.sa_overlap_weight_geometric;
            p.overlap_eps_area = opt.sa_overlap_eps_area;
            p.overlap_cost_cap = opt.sa_overlap_cost_cap;
            p.plateau_eps = opt.sa_plateau_eps;
            p.w_resolve_overlap = opt.sa_w_resolve_overlap;
            p.resolve_attempts = opt.sa_resolve_attempts;
            p.resolve_step_frac_max = opt.sa_resolve_step_frac_max;
            p.resolve_step_frac_min = opt.sa_resolve_step_frac_min;
            p.resolve_noise_frac = opt.sa_resolve_noise_frac;
            p.push_max_step_frac = opt.sa_push_max_step_frac;
            p.push_bisect_iters = opt.sa_push_bisect_iters;
            p.push_overshoot_frac = opt.sa_push_overshoot_frac;
            p.squeeze_pushes = opt.sa_squeeze_pushes;
            if (opt.sa_aggressive) {
                SARefiner::apply_aggressive_preset(p);
            }

            double best_local_side = std::numeric_limits<double>::infinity();
            std::vector<TreePose> best_local = cand;
            for (int r = 0; r < opt.ils_sa_restarts; ++r) {
                uint64_t sr =
                    cand_seed ^
                    (0x94D049BB133111EBULL + static_cast<uint64_t>(r) * 0xBF58476D1CE4E5B9ULL);
                SARefiner::Result res = sa.refine_min_side(cand, sr, p);
                if (!std::isfinite(res.best_side)) {
                    continue;
                }
                if (any_overlap(base_poly, res.best_poses, radius)) {
                    continue;
                }
                if (res.best_side + 1e-15 < best_local_side) {
                    best_local_side = res.best_side;
                    best_local = std::move(res.best_poses);
                }
            }
            if (!std::isfinite(best_local_side)) {
                continue;
            }
            cand = std::move(best_local);
        }

        const double cand_side = side_from_poses(cand);

        const double T = ils_temp_at(it);
        bool accepted = false;
        if (cand_side + 1e-15 < curr_side) {
            accepted = true;
        } else if (opt.ils_accept_sa) {
            accepted = accept_worse(curr_side, cand_side, T);
        }

        if (accepted) {
            out.accepted += 1;
            curr = cand;
            curr_side = cand_side;
        }

        if (cand_side + 1e-15 < best_side) {
            best = std::move(cand);
            best_side = cand_side;
        }
    }

    if (best_side + 1e-15 < start_side) {
        poses = std::move(best);
        out.improved = true;
        out.best_side = best_side;
    }

    return out;
}

ILSResult mz_its_soft_compact_impl(const Polygon& base_poly,
                                   double radius,
                                   std::vector<TreePose>& poses,
                                   uint64_t seed,
                                   const Options& opt) {
    ILSResult out;
    if (opt.mz_its_iters <= 0 || poses.empty()) {
        return out;
    }

    const int n = static_cast<int>(poses.size());

    auto side_from_poses = [&](const std::vector<TreePose>& p) -> double {
        std::vector<BoundingBox> bbs = bounding_boxes_for_poses(base_poly, p);
        return side_from_extents(compute_extents(bbs));
    };

    const int phase_a_iters =
        (opt.mz_phase_a_iters > 0) ? opt.mz_phase_a_iters : std::max(200, 3 * n);
    const int phase_b_iters =
        (opt.mz_phase_b_iters > 0) ? opt.mz_phase_b_iters : std::max(400, 6 * n);

    const int perturb_depth =
        (opt.mz_perturb_depth >= 0) ? opt.mz_perturb_depth : (10 * n);
    int tabu_depth = opt.mz_tabu_depth;
    if (tabu_depth < 0) {
        tabu_depth = 10 * n;
    }

    poses = quantize_poses_wrap_deg(poses);

    const double start_side = side_from_poses(poses);
    out.start_side = start_side;
    out.best_side = start_side;

    SARefiner sa(base_poly, radius);

    auto build_common_params = [&]() -> SARefiner::Params {
        SARefiner::Params p;
        p.w_micro = opt.sa_w_micro;
        p.w_swap_rot = opt.sa_w_swap_rot;
        p.w_relocate = opt.sa_w_relocate;
        p.w_block_translate = opt.sa_w_block_translate;
        p.w_block_rotate = opt.sa_w_block_rotate;
        p.w_lns = opt.sa_w_lns;
        p.block_size = opt.sa_block_size;
        p.lns_remove = opt.sa_lns_remove;
        p.lns_candidates = opt.sa_lns_candidates;
        p.lns_eval_attempts_per_tree = opt.sa_lns_eval_attempts_per_tree;
        p.hh_segment = opt.sa_hh_segment;
        p.hh_reaction = opt.sa_hh_reaction;
        p.overlap_metric = opt.sa_overlap_metric;
        p.overlap_eps_area = opt.sa_overlap_eps_area;
        p.overlap_cost_cap = opt.sa_overlap_cost_cap;
        p.plateau_eps = opt.sa_plateau_eps;
        p.resolve_attempts = opt.sa_resolve_attempts;
        p.resolve_step_frac_max = opt.sa_resolve_step_frac_max;
        p.resolve_step_frac_min = opt.sa_resolve_step_frac_min;
        p.resolve_noise_frac = opt.sa_resolve_noise_frac;
        p.push_max_step_frac = opt.sa_push_max_step_frac;
        p.push_bisect_iters = opt.sa_push_bisect_iters;
        p.push_overshoot_frac = opt.sa_push_overshoot_frac;
        p.squeeze_pushes = opt.sa_squeeze_pushes;
        if (opt.sa_aggressive) {
            SARefiner::apply_aggressive_preset(p);
        }
        return p;
    };

    auto local_optimize = [&](const std::vector<TreePose>& start,
                              uint64_t s) -> SARefiner::Result {
        // Fase A: liquefação (overlap barato + push com overshoot alto).
        SARefiner::Params pa = build_common_params();
        pa.iters = phase_a_iters;
        pa.t0 = opt.mz_a_t0;
        pa.t1 = opt.mz_a_t1;
        pa.overlap_weight = opt.mz_overlap_a;
        pa.overlap_weight_start = -1.0;
        pa.overlap_weight_end = -1.0;
        pa.overlap_weight_power = 1.0;
        pa.overlap_weight_geometric = false;
        pa.w_resolve_overlap = 0.0;
        pa.w_push_contact = opt.mz_w_push_contact;
        pa.push_overshoot_frac = std::max(0.0, opt.mz_push_overshoot_a);

        SARefiner::Result ra = sa.refine_min_side(start, s ^ 0xA93F1C2D3E4B5A67ULL, pa);
        if (ra.final_poses.empty()) {
            return ra;
        }

        // Fase B: solidificação (penalidade crescente + resolve_overlap).
        SARefiner::Params pb = build_common_params();
        pb.iters = phase_b_iters;
        pb.t0 = opt.mz_b_t0;
        pb.t1 = opt.mz_b_t1;
        pb.overlap_weight = 0.0;
        pb.overlap_weight_start = opt.mz_overlap_b_start;
        pb.overlap_weight_end = opt.mz_overlap_b_end;
        pb.overlap_weight_power = 1.0;
        pb.overlap_weight_geometric = opt.mz_overlap_b_geometric;
        pb.w_resolve_overlap = opt.mz_w_resolve_overlap_b;
        pb.w_push_contact = opt.mz_w_push_contact;
        pb.push_overshoot_frac = std::max(0.0, opt.mz_push_overshoot_b);

        return sa.refine_min_side(ra.final_poses, s ^ 0x7C3A5D1E9B4F2601ULL, pb);
    };

    auto swap_tabu_search = [&](const std::vector<TreePose>& start_in,
                                uint64_t s) -> std::vector<TreePose> {
        if (tabu_depth <= 0 || n <= 1) {
            return start_in;
        }

        std::mt19937_64 trng(s);
        std::uniform_int_distribution<int> pick_move(0, n - 2);
        std::uniform_int_distribution<int> ten_rand(0, 10);

        const int samples = std::max(1, std::min(opt.mz_tabu_samples, n - 1));
        const int base_tenure = std::max(1, n / 5);

        std::vector<int> tabu_until(static_cast<size_t>(n - 1), -1);
        std::vector<TreePose> curr = start_in;
        double curr_side = side_from_poses(curr);
        std::vector<TreePose> best = curr;
        double best_side = curr_side;

        int last_best = 0;
        for (int it = 0; it < tabu_depth; ++it) {
            // Amostragem de vizinhos: swap de deg entre pares adjacentes (i,i+1).
            int best_mv = -1;
            std::vector<TreePose> best_cand;
            double best_cand_side = std::numeric_limits<double>::infinity();

            for (int sidx = 0; sidx < samples; ++sidx) {
                const int mv = pick_move(trng);

                std::vector<TreePose> cand = curr;
                std::swap(cand[static_cast<size_t>(mv)].deg,
                          cand[static_cast<size_t>(mv + 1)].deg);
                cand = quantize_poses_wrap_deg(cand);

                SARefiner::Result res =
                    local_optimize(cand,
                                   s ^ (0xD1B54A32D192ED03ULL +
                                        static_cast<uint64_t>(it) * 0x9E3779B97F4A7C15ULL +
                                        static_cast<uint64_t>(mv)));
                if (!std::isfinite(res.best_side)) {
                    continue;
                }
                if (any_overlap(base_poly, res.best_poses, radius)) {
                    continue;
                }
                const double cand_side = res.best_side;

                const bool is_tabu = (it < tabu_until[static_cast<size_t>(mv)]);
                const bool aspiration = (cand_side + 1e-15 < best_side);
                if (is_tabu && !aspiration) {
                    continue;
                }

                if (cand_side + 1e-15 < best_cand_side) {
                    best_mv = mv;
                    best_cand_side = cand_side;
                    best_cand = std::move(res.best_poses);
                }
            }

            if (best_mv < 0) {
                break;
            }

            curr = std::move(best_cand);
            curr_side = best_cand_side;
            tabu_until[static_cast<size_t>(best_mv)] =
                it + base_tenure + ten_rand(trng);

            if (curr_side + 1e-15 < best_side) {
                best = curr;
                best_side = curr_side;
                last_best = it;
            }
            if (it - last_best >= tabu_depth) {
                break;
            }
        }

        return best;
    };

    // Inicial: minimiza localmente (com soft overlap) e depois tabu em swap-rot.
    {
        SARefiner::Result res0 = local_optimize(poses, seed ^ 0x9E3779B97F4A7C15ULL);
        if (std::isfinite(res0.best_side) && !any_overlap(base_poly, res0.best_poses, radius)) {
            poses = std::move(res0.best_poses);
        }
        poses = swap_tabu_search(poses, seed ^ 0xBF58476D1CE4E5B9ULL);
    }

    std::vector<TreePose> best = poses;
    std::vector<TreePose> curr = poses;
    double best_side = side_from_poses(best);
    double curr_side = best_side;
    out.best_side = best_side;

    std::mt19937_64 rng(seed ^ 0x94D049BB133111EBULL);
    std::uniform_real_distribution<double> uni_pos(-100.0, 100.0);
    std::uniform_real_distribution<double> uni_deg(-180.0, 180.0);
    std::uniform_int_distribution<int> pick_idx(0, n - 1);
    const int strength_max = std::max(1, n / 8);
    std::uniform_int_distribution<int> pick_strength(1, strength_max);

    int last_best = 0;
    for (int it = 0; it < opt.mz_its_iters; ++it) {
        if (perturb_depth > 0 && (it - last_best) >= perturb_depth) {
            break;
        }

        out.attempts += 1;

        std::vector<TreePose> cand = curr;
        const int strength = pick_strength(rng);
        for (int k = 0; k < strength; ++k) {
            const int i = pick_idx(rng);
            cand[static_cast<size_t>(i)].x = uni_pos(rng);
            cand[static_cast<size_t>(i)].y = uni_pos(rng);
            cand[static_cast<size_t>(i)].deg = uni_deg(rng);
        }
        cand = quantize_poses_wrap_deg(cand);

        SARefiner::Result res =
            local_optimize(cand,
                           seed ^
                               (0xA24BAED4963EE407ULL +
                                static_cast<uint64_t>(it) * 0x9E3779B97F4A7C15ULL));
        if (!std::isfinite(res.best_side) ||
            any_overlap(base_poly, res.best_poses, radius)) {
            continue;
        }
        cand = std::move(res.best_poses);
        cand = swap_tabu_search(cand,
                                seed ^
                                    (0xDEADBEEFCAFEBABEULL +
                                     static_cast<uint64_t>(it) * 0xBF58476D1CE4E5B9ULL));

        const double cand_side = side_from_poses(cand);

        if (cand_side <= curr_side + 1e-15) {
            out.accepted += 1;
            curr = cand;
            curr_side = cand_side;
        }

        if (cand_side + 1e-15 < best_side) {
            best = std::move(cand);
            best_side = cand_side;
            last_best = it;
        }
    }

    if (best_side + 1e-15 < start_side) {
        poses = std::move(best);
        out.improved = true;
        out.best_side = best_side;
    }
    return out;
}

Extents extents_without_index(const std::vector<BoundingBox>& bbs, int skip) {
    Extents e;
    e.min_x = std::numeric_limits<double>::infinity();
    e.max_x = -std::numeric_limits<double>::infinity();
    e.min_y = std::numeric_limits<double>::infinity();
    e.max_y = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < bbs.size(); ++i) {
        if (static_cast<int>(i) == skip) {
            continue;
        }
        const auto& bb = bbs[i];
        e.min_x = std::min(e.min_x, bb.min_x);
        e.max_x = std::max(e.max_x, bb.max_x);
        e.min_y = std::min(e.min_y, bb.min_y);
        e.max_y = std::max(e.max_y, bb.max_y);
    }
    return e;
}

Extents extents_from_bb(const BoundingBox& bb) {
    return Extents{bb.min_x, bb.max_x, bb.min_y, bb.max_y};
}

Extents merge_extents_bb(const Extents& e, const BoundingBox& bb) {
    Extents out = e;
    out.min_x = std::min(out.min_x, bb.min_x);
    out.max_x = std::max(out.max_x, bb.max_x);
    out.min_y = std::min(out.min_y, bb.min_y);
    out.max_y = std::max(out.max_y, bb.max_y);
    return out;
}

std::vector<char> boundary_band_mask(const std::vector<BoundingBox>& bbs,
                                     const Extents& e,
                                     double band) {
    std::vector<char> active;
    active.assign(bbs.size(), 0);
    if (bbs.empty()) {
        return active;
    }
    if (!(band > 0.0)) {
        for (auto& v : active) {
            v = 1;
        }
        return active;
    }

    int cnt = 0;
    for (size_t i = 0; i < bbs.size(); ++i) {
        const auto& bb = bbs[i];
        bool on =
            (bb.min_x <= e.min_x + band) || (bb.max_x >= e.max_x - band) ||
            (bb.min_y <= e.min_y + band) || (bb.max_y >= e.max_y - band);
        if (on) {
            active[i] = 1;
            ++cnt;
        }
    }

    if (cnt <= 0) {
        for (auto& v : active) {
            v = 1;
        }
    }
    return active;
}

int pick_greedy_boundary_removal(const std::vector<BoundingBox>& bbs,
                                 const Extents& e,
                                 double tol) {
    if (bbs.empty()) {
        return -1;
    }
    std::vector<int> candidates = boundary_indices(bbs, e, tol);
    if (candidates.empty()) {
        candidates.resize(bbs.size());
        for (size_t i = 0; i < bbs.size(); ++i) {
            candidates[i] = static_cast<int>(i);
        }
    }

    int best_idx = candidates.front();
    double best_side = std::numeric_limits<double>::infinity();
    double best_key1 = -std::numeric_limits<double>::infinity();
    double best_key2 = -std::numeric_limits<double>::infinity();

    for (int idx : candidates) {
        Extents e2 = extents_without_index(bbs, idx);
        double s2 = side_from_extents(e2);
        const auto& bb = bbs[static_cast<size_t>(idx)];
        const double cx = 0.5 * (bb.min_x + bb.max_x);
        const double cy = 0.5 * (bb.min_y + bb.max_y);
        const double key1 = std::max(std::abs(cx), std::abs(cy));
        const double key2 = std::hypot(cx, cy);

        if (s2 < best_side - 1e-15 ||
            (std::abs(s2 - best_side) <= 1e-15 &&
             (key1 > best_key1 + 1e-15 ||
              (std::abs(key1 - best_key1) <= 1e-15 &&
               (key2 > best_key2 + 1e-15 ||
                (std::abs(key2 - best_key2) <= 1e-15 && idx < best_idx)))))) {
            best_side = s2;
            best_key1 = key1;
            best_key2 = key2;
            best_idx = idx;
        }
    }

    return best_idx;
}

struct RemovalCandidate {
    int idx = -1;
    double side = std::numeric_limits<double>::infinity();
    double key1 = -std::numeric_limits<double>::infinity();
    double key2 = -std::numeric_limits<double>::infinity();
};

std::vector<int> pick_removal_candidates(const std::vector<BoundingBox>& bbs,
                                         const Extents& e,
                                         double band,
                                         int max_keep) {
    std::vector<int> candidates;
    candidates.reserve(bbs.size());
    const double tol = std::max(1e-12, band);
    for (size_t i = 0; i < bbs.size(); ++i) {
        const auto& bb = bbs[i];
        if (bb.min_x <= e.min_x + tol || bb.max_x >= e.max_x - tol ||
            bb.min_y <= e.min_y + tol || bb.max_y >= e.max_y - tol) {
            candidates.push_back(static_cast<int>(i));
        }
    }
    if (candidates.empty()) {
        candidates.resize(bbs.size());
        for (size_t i = 0; i < bbs.size(); ++i) {
            candidates[i] = static_cast<int>(i);
        }
    }

    std::vector<RemovalCandidate> scored;
    scored.reserve(candidates.size());
    for (int idx : candidates) {
        Extents e2 = extents_without_index(bbs, idx);
        double s2 = side_from_extents(e2);
        const auto& bb = bbs[static_cast<size_t>(idx)];
        const double cx = 0.5 * (bb.min_x + bb.max_x);
        const double cy = 0.5 * (bb.min_y + bb.max_y);
        RemovalCandidate c;
        c.idx = idx;
        c.side = s2;
        c.key1 = std::max(std::abs(cx), std::abs(cy));
        c.key2 = std::hypot(cx, cy);
        scored.push_back(c);
    }

    std::sort(scored.begin(),
              scored.end(),
              [](const RemovalCandidate& a, const RemovalCandidate& b) {
                  if (a.side != b.side) {
                      return a.side < b.side;
                  }
                  if (a.key1 != b.key1) {
                      return a.key1 > b.key1;
                  }
                  if (a.key2 != b.key2) {
                      return a.key2 > b.key2;
                  }
                  return a.idx < b.idx;
              });

    if (max_keep > 0 && static_cast<int>(scored.size()) > max_keep) {
        scored.resize(static_cast<size_t>(max_keep));
    }

    std::vector<int> out;
    out.reserve(scored.size());
    for (const auto& c : scored) {
        out.push_back(c.idx);
    }
    return out;
}

}  // namespace

ILSResult ils_basin_hop_compact(const Polygon& base_poly,
                                double radius,
                                std::vector<TreePose>& poses,
                                uint64_t seed,
                                const Options& opt) {
    return ils_basin_hop_compact_impl(base_poly, radius, poses, seed, opt);
}

ILSResult mz_its_soft_compact(const Polygon& base_poly,
                              double radius,
                              std::vector<TreePose>& poses,
                              uint64_t seed,
                              const Options& opt) {
    return mz_its_soft_compact_impl(base_poly, radius, poses, seed, opt);
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

std::vector<double> prefix_sides_from_bbs(const std::vector<BoundingBox>& bbs) {
    std::vector<double> side;
    side.resize(bbs.size() + 1, 0.0);
    Extents e;
    e.min_x = std::numeric_limits<double>::infinity();
    e.max_x = -std::numeric_limits<double>::infinity();
    e.min_y = std::numeric_limits<double>::infinity();
    e.max_y = -std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < bbs.size(); ++i) {
        e.min_x = std::min(e.min_x, bbs[i].min_x);
        e.max_x = std::max(e.max_x, bbs[i].max_x);
        e.min_y = std::min(e.min_y, bbs[i].min_y);
        e.max_y = std::max(e.max_y, bbs[i].max_y);
        side[i + 1] = side_from_extents(e);
    }
    return side;
}

std::vector<double> greedy_pruned_sides(const std::vector<BoundingBox>& bbs_in,
                                        int n_max,
                                        double tol) {
    if (bbs_in.empty()) {
        return std::vector<double>(static_cast<size_t>(n_max + 1), 0.0);
    }
    if (static_cast<int>(bbs_in.size()) < n_max) {
        throw std::runtime_error("greedy_pruned_sides: bbs.size() < n_max.");
    }

    std::vector<BoundingBox> bbs = bbs_in;
    std::vector<double> side_by_n;
    side_by_n.resize(static_cast<size_t>(n_max + 1), 0.0);

    for (int m = static_cast<int>(bbs.size()); m >= 1; --m) {
        Extents e = compute_extents(bbs);
        double s = side_from_extents(e);
        if (m <= n_max) {
            side_by_n[static_cast<size_t>(m)] = s;
        }

        if (m == 1) {
            break;
        }

        std::vector<int> candidates = boundary_indices(bbs, e, tol);
        if (candidates.empty()) {
            candidates.resize(bbs.size());
            for (size_t i = 0; i < bbs.size(); ++i) {
                candidates[i] = static_cast<int>(i);
            }
        }

        int best_idx = candidates.front();
        double best_side = std::numeric_limits<double>::infinity();
        double best_key1 = -std::numeric_limits<double>::infinity();
        double best_key2 = -std::numeric_limits<double>::infinity();

        for (int idx : candidates) {
            Extents e2 = extents_without_index(bbs, idx);
            double s2 = side_from_extents(e2);
            const auto& bb = bbs[static_cast<size_t>(idx)];
            const double cx = 0.5 * (bb.min_x + bb.max_x);
            const double cy = 0.5 * (bb.min_y + bb.max_y);
            const double key1 = std::max(std::abs(cx), std::abs(cy));
            const double key2 = std::hypot(cx, cy);

            // Critério:
            //  1) minimizar o lado após remover `idx`;
            //  2) em empate, remover o mais "longe" do centro (key1/key2 maiores),
            //     para deixar um núcleo mais central para os `m` menores.
            if (s2 < best_side - 1e-15 ||
                (std::abs(s2 - best_side) <= 1e-15 &&
                 (key1 > best_key1 + 1e-15 ||
                  (std::abs(key1 - best_key1) <= 1e-15 &&
                   (key2 > best_key2 + 1e-15 ||
                    (std::abs(key2 - best_key2) <= 1e-15 && idx < best_idx)))))) {
                best_side = s2;
                best_key1 = key1;
                best_key2 = key2;
                best_idx = idx;
            }
        }

        bbs.erase(bbs.begin() + best_idx);
    }

    return side_by_n;
}

double total_score_from_sides(const std::vector<double>& side_by_n, int n_max) {
    double total = 0.0;
    for (int n = 1; n <= n_max; ++n) {
        double s = side_by_n[static_cast<size_t>(n)];
        total += (s * s) / static_cast<double>(n);
    }
    return total;
}

std::vector<TreePose> greedy_prefix_min_side(const Polygon& base_poly,
                                             const std::vector<TreePose>& pool,
                                             int n_max) {
    if (n_max <= 0) {
        return {};
    }
    if (static_cast<int>(pool.size()) < n_max) {
        throw std::runtime_error("greedy_prefix_min_side: pool.size() < n_max.");
    }

    std::vector<BoundingBox> bbs = bounding_boxes_for_poses(base_poly, pool);

    std::vector<int> remaining;
    remaining.reserve(pool.size());
    for (int i = 0; i < static_cast<int>(pool.size()); ++i) {
        remaining.push_back(i);
    }

    std::vector<TreePose> out;
    out.reserve(static_cast<size_t>(n_max));

    Extents e;
    e.min_x = std::numeric_limits<double>::infinity();
    e.max_x = -std::numeric_limits<double>::infinity();
    e.min_y = std::numeric_limits<double>::infinity();
    e.max_y = -std::numeric_limits<double>::infinity();

    for (int k = 0; k < n_max; ++k) {
        int best_pos = -1;
        Extents best_e{};
        double best_side = std::numeric_limits<double>::infinity();
        double best_key1 = std::numeric_limits<double>::infinity();
        double best_key2 = std::numeric_limits<double>::infinity();
        int best_idx = -1;

        for (int pos = 0; pos < static_cast<int>(remaining.size()); ++pos) {
            int idx = remaining[static_cast<size_t>(pos)];
            Extents e2 = (k == 0) ? extents_from_bb(bbs[static_cast<size_t>(idx)])
                                  : merge_extents_bb(e, bbs[static_cast<size_t>(idx)]);
            double side = side_from_extents(e2);
            double key1 = std::max(std::abs(pool[static_cast<size_t>(idx)].x),
                                   std::abs(pool[static_cast<size_t>(idx)].y));
            double key2 = std::hypot(pool[static_cast<size_t>(idx)].x,
                                     pool[static_cast<size_t>(idx)].y);

            if (side < best_side - 1e-15 ||
                (std::abs(side - best_side) <= 1e-15 &&
                 (key1 < best_key1 - 1e-15 ||
                  (std::abs(key1 - best_key1) <= 1e-15 &&
                   (key2 < best_key2 - 1e-15 ||
                    (std::abs(key2 - best_key2) <= 1e-15 && idx < best_idx)))))) {
                best_side = side;
                best_key1 = key1;
                best_key2 = key2;
                best_pos = pos;
                best_idx = idx;
                best_e = e2;
            }
        }

        if (best_pos < 0 || best_idx < 0) {
            throw std::runtime_error("greedy_prefix_min_side: falha ao selecionar candidato.");
        }

        out.push_back(pool[static_cast<size_t>(best_idx)]);
        e = best_e;

        remaining[static_cast<size_t>(best_pos)] = remaining.back();
        remaining.pop_back();
    }

    return out;
}

PruneResult build_greedy_pruned_solutions(const Polygon& base_poly,
                                         const std::vector<TreePose>& poses_pool,
                                         int n_max,
                                         double tol) {
    if (n_max <= 0) {
        return PruneResult{};
    }
    if (static_cast<int>(poses_pool.size()) < n_max) {
        throw std::runtime_error("build_greedy_pruned_solutions: pool.size() < n_max.");
    }

    std::vector<TreePose> poses = poses_pool;
    std::vector<BoundingBox> bbs = bounding_boxes_for_poses(base_poly, poses);

    PruneResult res;
    res.solutions_by_n.resize(static_cast<size_t>(n_max + 1));
    res.side_by_n.resize(static_cast<size_t>(n_max + 1), 0.0);

    for (int m = static_cast<int>(poses.size()); m >= 1; --m) {
        Extents e = compute_extents(bbs);
        double s = side_from_extents(e);
        if (m <= n_max) {
            res.total_score += (s * s) / static_cast<double>(m);
            res.side_by_n[static_cast<size_t>(m)] = s;
            res.solutions_by_n[static_cast<size_t>(m)] = poses;
        }

        if (m == 1) {
            break;
        }

        std::vector<int> candidates = boundary_indices(bbs, e, tol);
        if (candidates.empty()) {
            candidates.resize(bbs.size());
            for (size_t i = 0; i < bbs.size(); ++i) {
                candidates[i] = static_cast<int>(i);
            }
        }

        int best_idx = candidates.front();
        double best_side = std::numeric_limits<double>::infinity();
        double best_key1 = -std::numeric_limits<double>::infinity();
        double best_key2 = -std::numeric_limits<double>::infinity();

        for (int idx : candidates) {
            Extents e2 = extents_without_index(bbs, idx);
            double s2 = side_from_extents(e2);
            const auto& bb = bbs[static_cast<size_t>(idx)];
            const double cx = 0.5 * (bb.min_x + bb.max_x);
            const double cy = 0.5 * (bb.min_y + bb.max_y);
            const double key1 = std::max(std::abs(cx), std::abs(cy));
            const double key2 = std::hypot(cx, cy);

            // Mesma lógica do `greedy_pruned_sides`: em platôs (remoções que não
            // mudam o lado atual), preferimos descartar poses mais externas para
            // preservar um conjunto mais central para `n` pequenos/médios.
            if (s2 < best_side - 1e-15 ||
                (std::abs(s2 - best_side) <= 1e-15 &&
                 (key1 > best_key1 + 1e-15 ||
                  (std::abs(key1 - best_key1) <= 1e-15 &&
                   (key2 > best_key2 + 1e-15 ||
                    (std::abs(key2 - best_key2) <= 1e-15 && idx < best_idx)))))) {
                best_side = s2;
                best_key1 = key1;
                best_key2 = key2;
                best_idx = idx;
            }
        }

        poses.erase(poses.begin() + best_idx);
        bbs.erase(bbs.begin() + best_idx);
    }

    return res;
}

ChainResult build_sa_chain_solutions(const Polygon& base_poly,
                                     double radius,
                                     const std::vector<TreePose>& start_nmax,
                                     int n_max,
                                     double band_step,
                                     const Options& opt) {
    ChainResult out;
    out.solutions_by_n.resize(static_cast<size_t>(n_max + 1));
    out.side_by_n.resize(static_cast<size_t>(n_max + 1), 0.0);
    if (n_max <= 0) {
        return out;
    }
    if (static_cast<int>(start_nmax.size()) != n_max) {
        throw std::runtime_error("build_sa_chain_solutions: start_nmax.size() != n_max.");
    }

    SARefiner sa(base_poly, radius);

    std::vector<TreePose> curr = start_nmax;
    for (int n = n_max; n >= 1; --n) {
        if (static_cast<int>(curr.size()) != n) {
            throw std::runtime_error("build_sa_chain_solutions: tamanho inconsistente no chain.");
        }

        const int iters =
            (n >= opt.sa_chain_min_n)
                ? (opt.sa_chain_base_iters + opt.sa_chain_iters_per_n * n)
                : 0;

        if (iters > 0 && n >= 2) {
            std::vector<BoundingBox> bbs = bounding_boxes_for_poses(base_poly, curr);
            Extents e = compute_extents(bbs);
            const double band = opt.sa_chain_band_layers * std::max(1e-12, band_step);
            std::vector<char> active = boundary_band_mask(bbs, e, band);

            SARefiner::Params p;
            p.iters = iters;
            p.w_micro = opt.sa_w_micro;
            p.w_swap_rot = opt.sa_w_swap_rot;
            p.w_relocate = opt.sa_w_relocate;
            p.w_block_translate = opt.sa_w_block_translate;
            p.w_block_rotate = opt.sa_w_block_rotate;
            p.w_lns = opt.sa_w_lns;
            p.w_push_contact = opt.sa_w_push_contact;
            p.w_squeeze = opt.sa_w_squeeze;
            p.block_size = opt.sa_block_size;
            p.lns_remove = opt.sa_lns_remove;
            p.lns_candidates = opt.sa_lns_candidates;
            p.lns_eval_attempts_per_tree = opt.sa_lns_eval_attempts_per_tree;
            p.hh_segment = opt.sa_hh_segment;
            p.hh_reaction = opt.sa_hh_reaction;
            p.overlap_metric = opt.sa_overlap_metric;
            p.overlap_weight = opt.sa_overlap_weight;
            p.overlap_weight_start = opt.sa_overlap_weight_start;
            p.overlap_weight_end = opt.sa_overlap_weight_end;
            p.overlap_weight_power = opt.sa_overlap_weight_power;
            p.overlap_weight_geometric = opt.sa_overlap_weight_geometric;
            p.overlap_eps_area = opt.sa_overlap_eps_area;
            p.overlap_cost_cap = opt.sa_overlap_cost_cap;
            p.plateau_eps = opt.sa_plateau_eps;
            p.w_resolve_overlap = opt.sa_w_resolve_overlap;
            p.resolve_attempts = opt.sa_resolve_attempts;
            p.resolve_step_frac_max = opt.sa_resolve_step_frac_max;
            p.resolve_step_frac_min = opt.sa_resolve_step_frac_min;
            p.resolve_noise_frac = opt.sa_resolve_noise_frac;
            p.push_max_step_frac = opt.sa_push_max_step_frac;
            p.push_bisect_iters = opt.sa_push_bisect_iters;
            p.push_overshoot_frac = opt.sa_push_overshoot_frac;
            p.squeeze_pushes = opt.sa_squeeze_pushes;
            if (opt.sa_aggressive) {
                SARefiner::apply_aggressive_preset(p);
            }

            uint64_t seed =
                opt.seed ^
                (0x632be59bd9b4e019ULL +
                 static_cast<uint64_t>(n) * 0xbf58476d1ce4e5b9ULL);
            SARefiner::Result res = sa.refine_min_side(curr, seed, p, &active);
            auto cand_q = quantize_poses(res.best_poses);
            if (!any_overlap(base_poly, cand_q, radius)) {
                double old_side =
                    bounding_square_side(transformed_polygons(base_poly, curr));
                double cand_side =
                    bounding_square_side(transformed_polygons(base_poly, cand_q));
                if (cand_side <= old_side + 1e-12) {
                    curr = std::move(cand_q);
                }
            }
        }

        double side = bounding_square_side(transformed_polygons(base_poly, curr));
        out.side_by_n[static_cast<size_t>(n)] = side;
        out.solutions_by_n[static_cast<size_t>(n)] = curr;
        out.total_score += (side * side) / static_cast<double>(n);

        if (n == 1) {
            break;
        }

        std::vector<BoundingBox> bbs = bounding_boxes_for_poses(base_poly, curr);
        Extents e = compute_extents(bbs);
        int remove_idx = pick_greedy_boundary_removal(bbs, e, 1e-12);
        if (remove_idx < 0 || remove_idx >= n) {
            throw std::runtime_error("build_sa_chain_solutions: índice inválido na remoção.");
        }
        curr.erase(curr.begin() + remove_idx);
    }

    return out;
}

ChainResult build_sa_beam_chain_solutions(const Polygon& base_poly,
                                          double radius,
                                          const std::vector<TreePose>& start_nmax,
                                          int n_max,
                                          double band_step,
                                          const Options& opt) {
    ChainResult out;
    out.solutions_by_n.resize(static_cast<size_t>(n_max + 1));
    out.side_by_n.resize(static_cast<size_t>(n_max + 1), 0.0);
    if (n_max <= 0) {
        return out;
    }
    if (static_cast<int>(start_nmax.size()) != n_max) {
        throw std::runtime_error("build_sa_beam_chain_solutions: start_nmax.size() != n_max.");
    }
    if (opt.sa_beam_width <= 0) {
        throw std::runtime_error("build_sa_beam_chain_solutions: sa_beam_width <= 0.");
    }

    SARefiner sa(base_poly, radius);

    struct BeamState {
        std::vector<TreePose> poses;
        double side = std::numeric_limits<double>::infinity();
    };

    auto side_from_poses = [&](const std::vector<TreePose>& poses) -> double {
        std::vector<BoundingBox> bbs = bounding_boxes_for_poses(base_poly, poses);
        Extents e = compute_extents(bbs);
        return side_from_extents(e);
    };

    auto refine_in_place = [&](std::vector<TreePose>& poses,
                               int n,
                               uint64_t seed,
                               int iters) {
        if (iters <= 0 || n < 2) {
            return;
        }
        std::vector<BoundingBox> bbs = bounding_boxes_for_poses(base_poly, poses);
        Extents e = compute_extents(bbs);
        const double band = opt.sa_chain_band_layers * std::max(1e-12, band_step);
        std::vector<char> active = boundary_band_mask(bbs, e, band);

        SARefiner::Params p;
        p.iters = iters;
        p.w_micro = opt.sa_w_micro;
        p.w_swap_rot = opt.sa_w_swap_rot;
        p.w_relocate = opt.sa_w_relocate;
        p.w_block_translate = opt.sa_w_block_translate;
        p.w_block_rotate = opt.sa_w_block_rotate;
        p.w_lns = opt.sa_w_lns;
        p.w_push_contact = opt.sa_w_push_contact;
        p.w_squeeze = opt.sa_w_squeeze;
        p.block_size = opt.sa_block_size;
        p.lns_remove = opt.sa_lns_remove;
        p.lns_candidates = opt.sa_lns_candidates;
        p.lns_eval_attempts_per_tree = opt.sa_lns_eval_attempts_per_tree;
        p.hh_segment = opt.sa_hh_segment;
        p.hh_reaction = opt.sa_hh_reaction;
        p.overlap_metric = opt.sa_overlap_metric;
        p.overlap_weight = opt.sa_overlap_weight;
        p.overlap_weight_start = opt.sa_overlap_weight_start;
        p.overlap_weight_end = opt.sa_overlap_weight_end;
        p.overlap_weight_power = opt.sa_overlap_weight_power;
        p.overlap_weight_geometric = opt.sa_overlap_weight_geometric;
        p.overlap_eps_area = opt.sa_overlap_eps_area;
        p.overlap_cost_cap = opt.sa_overlap_cost_cap;
        p.plateau_eps = opt.sa_plateau_eps;
        p.w_resolve_overlap = opt.sa_w_resolve_overlap;
        p.resolve_attempts = opt.sa_resolve_attempts;
        p.resolve_step_frac_max = opt.sa_resolve_step_frac_max;
        p.resolve_step_frac_min = opt.sa_resolve_step_frac_min;
        p.resolve_noise_frac = opt.sa_resolve_noise_frac;
        p.push_max_step_frac = opt.sa_push_max_step_frac;
        p.push_bisect_iters = opt.sa_push_bisect_iters;
        p.push_overshoot_frac = opt.sa_push_overshoot_frac;
        p.squeeze_pushes = opt.sa_squeeze_pushes;
        if (opt.sa_aggressive) {
            SARefiner::apply_aggressive_preset(p);
        }

        SARefiner::Result res = sa.refine_min_side(poses, seed, p, &active);
        auto cand_q = quantize_poses(res.best_poses);
        if (!any_overlap(base_poly, cand_q, radius)) {
            double old_side = side_from_extents(e);
            double cand_side =
                bounding_square_side(transformed_polygons(base_poly, cand_q));
            if (cand_side <= old_side + 1e-12) {
                poses = std::move(cand_q);
            }
        }
    };

    std::vector<BeamState> beam;
    beam.reserve(static_cast<size_t>(opt.sa_beam_width));
    for (int i = 0; i < opt.sa_beam_width; ++i) {
        BeamState s;
        s.poses = start_nmax;
        if (opt.sa_beam_init_iters > 0) {
            uint64_t seed =
                opt.seed ^
                (0x9e3779b97f4a7c15ULL +
                 static_cast<uint64_t>(i) * 0xbf58476d1ce4e5b9ULL);
            refine_in_place(s.poses, n_max, seed, opt.sa_beam_init_iters);
        }
        s.side = side_from_poses(s.poses);
        beam.push_back(std::move(s));
    }

    for (int n = n_max; n >= 1; --n) {
        if (beam.empty()) {
            throw std::runtime_error("build_sa_beam_chain_solutions: beam vazio.");
        }
        double best_side = std::numeric_limits<double>::infinity();
        int best_idx = 0;
        for (int i = 0; i < static_cast<int>(beam.size()); ++i) {
            if (beam[static_cast<size_t>(i)].side < best_side) {
                best_side = beam[static_cast<size_t>(i)].side;
                best_idx = i;
            }
        }
        out.side_by_n[static_cast<size_t>(n)] = best_side;
        out.solutions_by_n[static_cast<size_t>(n)] =
            beam[static_cast<size_t>(best_idx)].poses;
        out.total_score += (best_side * best_side) / static_cast<double>(n);

        if (n == 1) {
            break;
        }

        std::vector<BeamState> next;
        next.reserve(static_cast<size_t>(opt.sa_beam_width * opt.sa_beam_remove));

        for (int b = 0; b < static_cast<int>(beam.size()); ++b) {
            const auto& state = beam[static_cast<size_t>(b)];
            if (static_cast<int>(state.poses.size()) != n) {
                throw std::runtime_error("build_sa_beam_chain_solutions: tamanho inconsistente no beam.");
            }
            std::vector<BoundingBox> bbs =
                bounding_boxes_for_poses(base_poly, state.poses);
            Extents e = compute_extents(bbs);
            const double band = opt.sa_chain_band_layers * std::max(1e-12, band_step);
            const int max_keep = std::max(1, opt.sa_beam_remove);
            std::vector<int> remove_idxs =
                pick_removal_candidates(bbs, e, band, max_keep);

            for (int idx : remove_idxs) {
                BeamState cand;
                cand.poses = state.poses;
                cand.poses.erase(cand.poses.begin() + idx);

                const int iters =
                    (n - 1 >= opt.sa_chain_min_n) ? opt.sa_beam_micro_iters : 0;
                if (iters > 0) {
                    uint64_t seed =
                        opt.seed ^
                        (0x632be59bd9b4e019ULL +
                         static_cast<uint64_t>(n) * 0xbf58476d1ce4e5b9ULL +
                         static_cast<uint64_t>(b) * 0x94d049bb133111ebULL +
                         static_cast<uint64_t>(idx) * 0x1d8e4e27c47d124fULL);
                    refine_in_place(cand.poses, n - 1, seed, iters);
                }
                cand.side = side_from_poses(cand.poses);
                next.push_back(std::move(cand));
            }
        }

        if (next.empty()) {
            throw std::runtime_error("build_sa_beam_chain_solutions: sem candidatos.");
        }

        const int keep = std::min(opt.sa_beam_width, static_cast<int>(next.size()));
        std::partial_sort(next.begin(),
                          next.begin() + keep,
                          next.end(),
                          [](const BeamState& a, const BeamState& b) {
                              if (a.side != b.side) {
                                  return a.side < b.side;
                              }
                              return a.poses.size() < b.poses.size();
                          });
        next.resize(static_cast<size_t>(keep));
        beam = std::move(next);
    }

    return out;
}
