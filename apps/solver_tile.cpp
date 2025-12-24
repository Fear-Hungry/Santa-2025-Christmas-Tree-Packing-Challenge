#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "geometry/collision.hpp"
#include "solvers/boundary_refine.hpp"
#include "solvers/compaction_contact.hpp"
#include "geometry/geom.hpp"
#include "utils/micro_adjust.hpp"
#include "solvers/post_opt.hpp"
#include "solvers/prefix_prune.hpp"
#include "solvers/sa.hpp"
#include "utils/solver_tile_cli.hpp"
#include "geometry/spatial_grid.hpp"
#include "utils/submission_io.hpp"
#include "solvers/tiling_pool.hpp"
#include "utils/wrap_utils.hpp"

namespace {

struct Candidate {
    TreePose pose;
    double key1;
    double key2;
};

struct Eval {
    double best_total = std::numeric_limits<double>::infinity();
    double best_angle = 0.0;
    std::vector<TreePose> best_poses;
};

enum class TargetTier {
    kA,
    kB,
    kC,
};

constexpr std::array<const char*, SARefiner::kNumOps> kHhOpNames = {
    "micro",
    "swap_rot",
    "relocate",
    "block_translate",
    "block_rotate",
    "lns",
    "push_contact",
    "slide_contact",
    "squeeze",
    "global_rotate",
    "eject_chain",
    "resolve_overlap",
};

SARefiner::HHState aggregate_hh_states(const std::vector<SARefiner::HHState>& states,
                                       int n_lo,
                                       int n_hi) {
    SARefiner::HHState out;
    int count = 0;
    for (int n = n_lo; n <= n_hi; ++n) {
        if (n < 0 || n >= static_cast<int>(states.size())) {
            continue;
        }
        const SARefiner::HHState& state = states[static_cast<size_t>(n)];
        if (!state.initialized) {
            continue;
        }
        for (size_t k = 0; k < kHhOpNames.size(); ++k) {
            out.weights[k] += state.weights[k];
            out.op_score[k] += state.op_score[k];
            out.op_uses[k] += state.op_uses[k];
        }
        count += 1;
    }
    if (count > 0) {
        for (size_t k = 0; k < kHhOpNames.size(); ++k) {
            out.weights[k] /= static_cast<double>(count);
            out.op_score[k] /= static_cast<double>(count);
        }
        out.initialized = true;
    }
    return out;
}

void dump_hh_bucket(const char* label, const SARefiner::HHState& state) {
    if (!state.initialized) {
        std::cout << "SA HH bucket " << label << ": empty\n";
        return;
    }
    const std::ios::fmtflags flags = std::cout.flags();
    const std::streamsize precision = std::cout.precision();
    std::cout.setf(std::ios::fixed);
    std::cout << std::setprecision(4);

    std::cout << "SA HH bucket " << label << " weights:";
    for (size_t k = 0; k < kHhOpNames.size(); ++k) {
        std::cout << " " << kHhOpNames[k] << "=" << state.weights[k];
    }
    std::cout << "\n";

    std::cout << "SA HH bucket " << label << " uses:";
    for (size_t k = 0; k < kHhOpNames.size(); ++k) {
        std::cout << " " << kHhOpNames[k] << "=" << state.op_uses[k];
    }
    std::cout << "\n";

    std::cout.flags(flags);
    std::cout.precision(precision);
}

struct SideArea {
    double side = 0.0;
    double area = 0.0;
};

struct TargetEntry {
    int n = 0;
    double term = 0.0;
    TargetTier tier = TargetTier::kC;
};

int clamp_int(int value, int lo, int hi) {
    return std::min(hi, std::max(lo, value));
}

double normalized_budget_scale(double scale) {
    return (scale > 0.0) ? scale : 1.0;
}

SideArea eval_side_area(const Polygon& base_poly,
                        const std::vector<TreePose>& poses) {
    if (poses.empty()) {
        return {};
    }
    double min_x = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    for (const auto& pose : poses) {
        BoundingBox bb = bounding_box(transform_polygon(base_poly, pose));
        min_x = std::min(min_x, bb.min_x);
        max_x = std::max(max_x, bb.max_x);
        min_y = std::min(min_y, bb.min_y);
        max_y = std::max(max_y, bb.max_y);
    }
    const double width = max_x - min_x;
    const double height = max_y - min_y;
    SideArea out;
    out.side = std::max(width, height);
    out.area = width * height;
    return out;
}

bool improves_side_area(const SideArea& cand,
                        const SideArea& best,
                        double plateau_eps) {
    if (!std::isfinite(best.side)) {
        return std::isfinite(cand.side);
    }
    if (cand.side < best.side - 1e-12) {
        return true;
    }
    if (cand.side <= best.side + plateau_eps && cand.area < best.area - 1e-12) {
        return true;
    }
    return false;
}

Extents compute_extents_from_bbs(const std::vector<BoundingBox>& bbs) {
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

bool aabb_overlap(const BoundingBox& a, const BoundingBox& b) {
    if (a.max_x < b.min_x || b.max_x < a.min_x) {
        return false;
    }
    if (a.max_y < b.min_y || b.max_y < a.min_y) {
        return false;
    }
    return true;
}

bool repair_overlaps_mtv(const Polygon& base_poly,
                         double radius,
                         std::vector<TreePose>& poses,
                         int passes,
                         double damping,
                         double split,
                         uint64_t seed) {
    const int n = static_cast<int>(poses.size());
    if (n < 2) {
        return true;
    }
    if (passes <= 0) {
        return !any_overlap(base_poly, poses, radius);
    }

    const double thr = 2.0 * radius + 1e-9;
    const double limit_sq = thr * thr;

    std::vector<Point> centers;
    centers.reserve(poses.size());
    std::vector<Polygon> polys;
    polys.reserve(poses.size());
    std::vector<BoundingBox> bbs;
    bbs.reserve(poses.size());
    UniformGridIndex grid(n, thr);
    std::vector<int> neigh;
    neigh.reserve(64);
    for (int i = 0; i < n; ++i) {
        const auto& pose = poses[static_cast<size_t>(i)];
        centers.push_back(Point{pose.x, pose.y});
        Polygon poly = transform_polygon(base_poly, pose);
        polys.push_back(std::move(poly));
        bbs.push_back(bounding_box(polys.back()));
        grid.insert(i, pose.x, pose.y);
    }

    auto find_first = [&]() -> std::pair<int, int> {
        for (int i = 0; i < n; ++i) {
            grid.gather(centers[static_cast<size_t>(i)].x,
                        centers[static_cast<size_t>(i)].y,
                        neigh);
            std::sort(neigh.begin(), neigh.end());
            neigh.erase(std::unique(neigh.begin(), neigh.end()), neigh.end());

            for (int j : neigh) {
                if (j <= i) {
                    continue;
                }
                const double dx = centers[static_cast<size_t>(i)].x -
                                  centers[static_cast<size_t>(j)].x;
                const double dy = centers[static_cast<size_t>(i)].y -
                                  centers[static_cast<size_t>(j)].y;
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
        double ka = centrality_key(poses[static_cast<size_t>(a)]);
        double kb = centrality_key(poses[static_cast<size_t>(b)]);
        return (ka < kb) ? a : b;
    };

    SARefiner sep(base_poly, radius);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    const bool split_move = (split > 0.0 && split < 1.0);
    for (int pass = 0; pass < passes; ++pass) {
        bool any = false;
        for (int i = 0; i < n; ++i) {
            grid.gather(centers[static_cast<size_t>(i)].x,
                        centers[static_cast<size_t>(i)].y,
                        neigh);
            std::sort(neigh.begin(), neigh.end());
            neigh.erase(std::unique(neigh.begin(), neigh.end()), neigh.end());

            for (int j : neigh) {
                if (j <= i) {
                    continue;
                }
                const double dx = centers[static_cast<size_t>(i)].x -
                                  centers[static_cast<size_t>(j)].x;
                const double dy = centers[static_cast<size_t>(i)].y -
                                  centers[static_cast<size_t>(j)].y;
                if (dx * dx + dy * dy > limit_sq) {
                    continue;
                }
                if (!aabb_overlap(bbs[static_cast<size_t>(i)],
                                  bbs[static_cast<size_t>(j)])) {
                    continue;
                }
                if (!polygons_intersect(polys[static_cast<size_t>(i)],
                                        polys[static_cast<size_t>(j)])) {
                    continue;
                }
                any = true;

                int idx = pick_idx(i, j);
                int other = (idx == i) ? j : i;

                Point mtv{0.0, 0.0};
                double ov_area = 0.0;
                if (!sep.overlap_mtv(poses[static_cast<size_t>(idx)],
                                     poses[static_cast<size_t>(other)],
                                     mtv,
                                     ov_area) ||
                    !(std::hypot(mtv.x, mtv.y) > 1e-12)) {
                    double ddx = poses[static_cast<size_t>(idx)].x -
                                 poses[static_cast<size_t>(other)].x;
                    double ddy = poses[static_cast<size_t>(idx)].y -
                                 poses[static_cast<size_t>(other)].y;
                    double norm = std::hypot(ddx, ddy);
                    if (!(norm > 1e-12)) {
                        double ang = u01(rng) * 2.0 * 3.14159265358979323846;
                        ddx = std::cos(ang);
                        ddy = std::sin(ang);
                        norm = 1.0;
                    }
                    mtv.x = ddx / norm;
                    mtv.y = ddy / norm;
                    const double step = std::max(1e-6, 1e-3 * radius);
                    mtv.x *= step;
                    mtv.y *= step;
                }

                mtv.x *= damping;
                mtv.y *= damping;

                double frac_idx = 1.0;
                double frac_other = 0.0;
                if (split_move) {
                    frac_idx = split;
                    frac_other = 1.0 - split;
                }

                double scale = 1.0;
                bool moved = false;
                for (int bt = 0; bt < 12 && !moved; ++bt) {
                    TreePose cand_idx = poses[static_cast<size_t>(idx)];
                    TreePose cand_other = poses[static_cast<size_t>(other)];
                    cand_idx.x += mtv.x * frac_idx * scale;
                    cand_idx.y += mtv.y * frac_idx * scale;
                    if (split_move) {
                        cand_other.x -= mtv.x * frac_other * scale;
                        cand_other.y -= mtv.y * frac_other * scale;
                    }

                    cand_idx = quantize_pose_wrap_deg(cand_idx);
                    if (split_move) {
                        cand_other = quantize_pose_wrap_deg(cand_other);
                    }

                    if (cand_idx.x < -100.0 || cand_idx.x > 100.0 ||
                        cand_idx.y < -100.0 || cand_idx.y > 100.0) {
                        scale *= 0.5;
                        continue;
                    }
                    if (split_move &&
                        (cand_other.x < -100.0 || cand_other.x > 100.0 ||
                         cand_other.y < -100.0 || cand_other.y > 100.0)) {
                        scale *= 0.5;
                        continue;
                    }

                    poses[static_cast<size_t>(idx)] = cand_idx;
                    centers[static_cast<size_t>(idx)] = Point{cand_idx.x, cand_idx.y};
                    polys[static_cast<size_t>(idx)] =
                        transform_polygon(base_poly, cand_idx);
                    bbs[static_cast<size_t>(idx)] =
                        bounding_box(polys[static_cast<size_t>(idx)]);
                    grid.update_position(idx, cand_idx.x, cand_idx.y);

                    if (split_move) {
                        poses[static_cast<size_t>(other)] = cand_other;
                        centers[static_cast<size_t>(other)] =
                            Point{cand_other.x, cand_other.y};
                        polys[static_cast<size_t>(other)] =
                            transform_polygon(base_poly, cand_other);
                        bbs[static_cast<size_t>(other)] =
                            bounding_box(polys[static_cast<size_t>(other)]);
                        grid.update_position(other, cand_other.x, cand_other.y);
                    }

                    moved = true;
                }
            }
        }

        if (!any) {
            return true;
        }
    }

    return (find_first().first < 0);
}

struct GlobalContractionResult {
    bool improved = false;
    double start_side = std::numeric_limits<double>::infinity();
    double best_side = std::numeric_limits<double>::infinity();
    int steps_ok = 0;
};

GlobalContractionResult apply_global_contraction_nmax(const Polygon& base_poly,
                                                      double radius,
                                                      std::vector<TreePose>& poses,
                                                      uint64_t seed,
                                                      const Options& opt) {
    GlobalContractionResult out;
    if (opt.global_contract_steps <= 0 || poses.size() < 2) {
        return out;
    }

    const double thr = 2.0 * radius + 1e-9;
    const double thr_sq = thr * thr;

    out.start_side = bounding_square_side(transformed_polygons(base_poly, poses));
    out.best_side = out.start_side;

    std::vector<TreePose> best = poses;
    std::vector<TreePose> curr = poses;

    SARefiner sep(base_poly, radius);
    std::mt19937_64 rng(seed ^ 0xBF58476D1CE4E5B9ULL);
    std::normal_distribution<double> normal(0.0, 1.0);

    for (int step = 0; step < opt.global_contract_steps; ++step) {
        const std::vector<TreePose> base = curr;
        double step_scale = opt.global_contract_scale;
        uint64_t step_seed_base =
            seed ^
            (0x94d049bb133111ebULL + static_cast<uint64_t>(step) * 0x9e3779b97f4a7c15ULL);

        bool step_ok = false;
        for (int backoff = 0; backoff < 12 && !step_ok; ++backoff) {
            curr = base;

            std::vector<BoundingBox> curr_bbs = bounding_boxes_for_poses(base_poly, curr);
            Extents e = compute_extents_from_bbs(curr_bbs);
            const double cx = 0.5 * (e.min_x + e.max_x);
            const double cy = 0.5 * (e.min_y + e.max_y);

            for (auto& p : curr) {
                p.x = cx + step_scale * (p.x - cx);
                p.y = cy + step_scale * (p.y - cy);
            }
            curr = quantize_poses_wrap_deg(curr);

            const bool do_relax = (backoff == 0);
            if (do_relax &&
                opt.global_contract_relax_iters > 0 &&
                (opt.global_contract_overlap_force > 0.0 ||
                 opt.global_contract_center_force > 0.0)) {
                for (int it = 0; it < opt.global_contract_relax_iters; ++it) {
                    std::vector<BoundingBox> bbs = bounding_boxes_for_poses(base_poly, curr);
                    Extents e2 = compute_extents_from_bbs(bbs);
                    const double ccx = 0.5 * (e2.min_x + e2.max_x);
                    const double ccy = 0.5 * (e2.min_y + e2.max_y);
                    const double side = side_from_extents(e2);
                    const double max_step =
                        opt.global_contract_step_frac * std::max(1e-9, side);

                    std::vector<Point> forces(curr.size(), Point{0.0, 0.0});
                    if (opt.global_contract_center_force > 0.0) {
                        for (size_t i = 0; i < curr.size(); ++i) {
                            forces[i].x +=
                                (ccx - curr[i].x) * opt.global_contract_center_force;
                            forces[i].y +=
                                (ccy - curr[i].y) * opt.global_contract_center_force;
                        }
                    }

                    if (opt.global_contract_overlap_force > 0.0) {
                        for (size_t i = 0; i < curr.size(); ++i) {
                            for (size_t j = i + 1; j < curr.size(); ++j) {
                                const double dx = curr[i].x - curr[j].x;
                                const double dy = curr[i].y - curr[j].y;
                                if (dx * dx + dy * dy > thr_sq) {
                                    continue;
                                }
                                if (!aabb_overlap(bbs[i], bbs[j])) {
                                    continue;
                                }
                                Point mtv{0.0, 0.0};
                                double area = 0.0;
                                if (!sep.overlap_mtv(curr[i], curr[j], mtv, area)) {
                                    continue;
                                }
                                forces[i].x += mtv.x * opt.global_contract_overlap_force;
                                forces[i].y += mtv.y * opt.global_contract_overlap_force;
                                forces[j].x -= mtv.x * opt.global_contract_overlap_force;
                                forces[j].y -= mtv.y * opt.global_contract_overlap_force;
                            }
                        }
                    }

                    for (size_t i = 0; i < curr.size(); ++i) {
                        double dx = forces[i].x;
                        double dy = forces[i].y;
                        const double norm = std::hypot(dx, dy);
                        if (norm > max_step && norm > 1e-12) {
                            const double scale = max_step / norm;
                            dx *= scale;
                            dy *= scale;
                        }
                        curr[i].x += dx + normal(rng) * (1e-6 * max_step);
                        curr[i].y += dy + normal(rng) * (1e-6 * max_step);
                        curr[i] = quantize_pose_wrap_deg(curr[i]);
                    }
                }
            }

            uint64_t step_seed =
                step_seed_base ^ (static_cast<uint64_t>(backoff) * 0xBF58476D1CE4E5B9ULL);
            if (!repair_overlaps_mtv(base_poly,
                                     radius,
                                     curr,
                                     opt.global_contract_repair_passes,
                                     opt.ils_repair_mtv_damping,
                                     opt.ils_repair_mtv_split,
                                     step_seed)) {
                step_scale = 0.5 * (step_scale + 1.0);
                continue;
            }

            if (opt.global_contract_sa_restarts > 0 && opt.global_contract_sa_iters > 0) {
                SARefiner sa(base_poly, radius);
                SARefiner::Params p;
                p.iters = opt.global_contract_sa_iters;
                p.w_micro = opt.sa_w_micro;
                p.w_swap_rot = opt.sa_w_swap_rot;
                p.w_relocate = opt.sa_w_relocate;
                p.w_block_translate = opt.sa_w_block_translate;
                p.w_block_rotate = opt.sa_w_block_rotate;
                p.w_lns = opt.sa_w_lns;
                p.w_push_contact = opt.sa_w_push_contact;
                p.w_slide_contact = opt.sa_w_slide_contact;
                p.w_squeeze = opt.sa_w_squeeze;
                p.block_size = opt.sa_block_size;
                p.lns_remove = opt.sa_lns_remove;
                p.lns_candidates = opt.sa_lns_candidates;
                p.lns_eval_attempts_per_tree = opt.sa_lns_eval_attempts_per_tree;
                p.hh_segment = opt.sa_hh_segment;
                p.hh_reaction = opt.sa_hh_reaction;
                p.hh_auto = opt.sa_hh_auto;
                p.overlap_metric = opt.sa_overlap_metric;
                p.overlap_weight = 0.0;
                p.plateau_eps = opt.sa_plateau_eps;
                p.w_resolve_overlap = opt.sa_w_resolve_overlap;
                p.resolve_attempts = opt.sa_resolve_attempts;
                p.resolve_step_frac_max = opt.sa_resolve_step_frac_max;
                p.resolve_step_frac_min = opt.sa_resolve_step_frac_min;
                p.resolve_noise_frac = opt.sa_resolve_noise_frac;
                p.push_max_step_frac = opt.sa_push_max_step_frac;
                p.push_bisect_iters = opt.sa_push_bisect_iters;
                p.push_overshoot_frac = opt.sa_push_overshoot_frac;
                p.slide_dirs = opt.sa_slide_dirs;
                p.slide_dir_bias = opt.sa_slide_dir_bias;
                p.slide_max_step_frac = opt.sa_slide_max_step_frac;
                p.slide_bisect_iters = opt.sa_slide_bisect_iters;
                p.slide_min_gain = opt.sa_slide_min_gain;
                p.slide_schedule_max_frac = opt.sa_slide_schedule_max_frac;
                p.squeeze_pushes = opt.sa_squeeze_pushes;
                if (opt.sa_aggressive) {
                    SARefiner::apply_aggressive_preset(p);
                }
                if (opt.sa_hh_auto) {
                    SARefiner::apply_hh_auto_preset(p);
                }

                double best_sa_side = std::numeric_limits<double>::infinity();
                std::vector<TreePose> best_sa = curr;
                for (int r = 0; r < opt.global_contract_sa_restarts; ++r) {
                    uint64_t sr =
                        step_seed ^
                        (0x9e3779b97f4a7c15ULL +
                         static_cast<uint64_t>(r) * 0x2545F4914F6CDD1DULL);
                    SARefiner::Result res = sa.refine_min_side(curr, sr, p, nullptr, nullptr);
                    auto cand_q = quantize_poses(res.best_poses);
                    if (any_overlap(base_poly, cand_q, radius)) {
                        continue;
                    }
                    double side = bounding_square_side(transformed_polygons(base_poly, cand_q));
                    if (side + 1e-15 < best_sa_side) {
                        best_sa_side = side;
                        best_sa = std::move(cand_q);
                    }
                }
                curr = std::move(best_sa);
            }

            if (any_overlap(base_poly, curr, radius)) {
                step_scale = 0.5 * (step_scale + 1.0);
                continue;
            }
            step_ok = true;
        }

        if (!step_ok) {
            curr = base;
            break;
        }

        const double side = bounding_square_side(transformed_polygons(base_poly, curr));
        out.steps_ok += 1;
        if (side + 1e-12 < out.best_side) {
            out.best_side = side;
            best = curr;
            out.improved = true;
        }
    }

    if (out.improved) {
        poses = std::move(best);
    }
    return out;
}

TargetTier tier_for_rank(int idx, int tier_a, int tier_b) {
    if (idx < tier_a) {
        return TargetTier::kA;
    }
    if (idx < tier_a + tier_b) {
        return TargetTier::kB;
    }
    return TargetTier::kC;
}

int scale_count(int base, double scale, int min_floor, int max_cap) {
    const double s = normalized_budget_scale(scale);
    int scaled = static_cast<int>(std::round(static_cast<double>(base) * s));
    scaled = std::max(min_floor, scaled);
    if (max_cap > 0) {
        scaled = std::min(max_cap, scaled);
    }
    return scaled;
}

std::vector<TargetEntry> select_target_entries(
    const Polygon& base_poly,
    const std::vector<std::vector<TreePose>>& solutions_by_n,
    const Options& opt,
    double* total_score_out) {
    struct TermRow {
        int n = 0;
        double term = 0.0;
    };
    std::vector<TermRow> rows;
    rows.reserve(static_cast<size_t>(opt.n_max));
    double total_score = 0.0;
    for (int n = 1; n <= opt.n_max; ++n) {
        const auto& sol = solutions_by_n[static_cast<size_t>(n)];
        if (static_cast<int>(sol.size()) != n) {
            continue;
        }
        auto sol_q = quantize_poses(sol);
        const double side =
            bounding_square_side(transformed_polygons(base_poly, sol_q));
        const double term = (side * side) / static_cast<double>(n);
        total_score += term;
        rows.push_back(TermRow{n, term});
    }
    if (total_score_out) {
        *total_score_out = total_score;
    }
    if (rows.empty()) {
        return {};
    }
    std::sort(rows.begin(), rows.end(), [](const TermRow& a, const TermRow& b) {
        return a.term > b.term;
    });

    int target_count = 0;
    if (opt.target_m > 0) {
        target_count = clamp_int(opt.target_m, opt.target_m_min, opt.target_m_max);
    } else {
        const double cover = std::max(0.0, std::min(1.0, opt.target_cover));
        const double target_score = cover * total_score;
        double acc = 0.0;
        for (const auto& row : rows) {
            acc += row.term;
            ++target_count;
            if (acc >= target_score) {
                break;
            }
        }
        target_count = clamp_int(target_count, opt.target_m_min, opt.target_m_max);
    }
    const double scale = normalized_budget_scale(opt.target_budget_scale);
    if (std::abs(scale - 1.0) > 1e-12) {
        int scaled = static_cast<int>(std::round(target_count * std::sqrt(scale)));
        target_count = clamp_int(scaled, opt.target_m_min, opt.target_m_max);
    }
    target_count = std::min(target_count, static_cast<int>(rows.size()));

    const int tier_a = std::min(opt.target_tier_a, target_count);
    const int tier_b =
        std::min(opt.target_tier_b, std::max(0, target_count - tier_a));

    std::vector<TargetEntry> entries;
    entries.reserve(static_cast<size_t>(target_count));
    for (int i = 0; i < target_count; ++i) {
        entries.push_back(TargetEntry{
            rows[static_cast<size_t>(i)].n,
            rows[static_cast<size_t>(i)].term,
            tier_for_rank(i, tier_a, tier_b)});
    }
    return entries;
}

struct TierEarlyStop {
    int min_passes = 0;
    int patience_passes = 0;
    int min_iters = 0;
    int patience_iters = 0;
};

TierEarlyStop early_stop_defaults(TargetTier tier) {
    switch (tier) {
    case TargetTier::kA:
        return TierEarlyStop{6, 3, 3000, 1200};
    case TargetTier::kB:
        return TierEarlyStop{4, 2, 1500, 800};
    case TargetTier::kC:
        return TierEarlyStop{3, 2, 800, 600};
    }
    return TierEarlyStop{};
}

void apply_compact_early_stop(compaction_contact::Params& params,
                              TargetTier tier,
                              const Options& opt) {
    if (!opt.target_early_stop) {
        params.early_stop.enabled = false;
        return;
    }
    TierEarlyStop base = early_stop_defaults(tier);
    params.early_stop.enabled = true;
    params.early_stop.min_passes =
        scale_count(base.min_passes, opt.target_budget_scale, 2, params.passes);
    params.early_stop.patience_passes =
        scale_count(base.patience_passes, opt.target_budget_scale, 1, params.passes);
}

void apply_sa_early_stop(SARefiner::Params& params,
                         TargetTier tier,
                         const Options& opt) {
    if (!opt.target_early_stop) {
        params.early_stop = false;
        return;
    }
    TierEarlyStop base = early_stop_defaults(tier);
    params.early_stop = true;
    params.early_stop_check_interval = opt.target_sa_check_interval;
    params.early_stop_min_iters =
        scale_count(base.min_iters, opt.target_budget_scale, 400, params.iters);
    params.early_stop_patience_iters =
        scale_count(base.patience_iters, opt.target_budget_scale, 200, params.iters);
}

compaction_contact::Params make_compact_params(TargetTier tier, const Options& opt) {
    compaction_contact::Params params;
    const double scale = normalized_budget_scale(opt.target_budget_scale);
    const int base_passes = (tier == TargetTier::kA) ? 18 : (tier == TargetTier::kB ? 14 : 12);
    const int base_attempts =
        (tier == TargetTier::kA) ? 48 : (tier == TargetTier::kB ? 40 : 32);
    const int base_patience = (tier == TargetTier::kA) ? 5 : (tier == TargetTier::kB ? 4 : 3);
    const int base_boundary = (tier == TargetTier::kA) ? 20 : (tier == TargetTier::kB ? 16 : 14);

    params.passes = clamp_int(static_cast<int>(std::round(base_passes * scale)), 4, 200);
    params.attempts_per_pass =
        clamp_int(static_cast<int>(std::round(base_attempts * scale)), 8, 256);
    params.patience =
        clamp_int(static_cast<int>(std::round(base_patience * std::sqrt(scale))), 2, params.passes);
    params.boundary_topk =
        clamp_int(static_cast<int>(std::round(base_boundary * std::sqrt(scale))), 8, 40);
    params.push_bisect_iters = 12;
    params.push_max_step_frac = 0.9;
    params.plateau_eps = 1e-4;
    params.alt_axis = true;
    params.final_rigid = false;
    params.quantize_decimals = 9;
    apply_compact_early_stop(params, tier, opt);
    return params;
}

double tier_scale(TargetTier tier) {
    switch (tier) {
    case TargetTier::kA:
        return 1.0;
    case TargetTier::kB:
        return 0.70;
    case TargetTier::kC:
        return 0.50;
    }
    return 1.0;
}

SARefiner::Params make_targeted_sa_params(TargetTier tier,
                                          const Options& opt,
                                          int n) {
    SARefiner::Params p;
    const int base_iters = 2500 + 10 * n;
    const double scale = normalized_budget_scale(opt.target_budget_scale);
    p.iters = static_cast<int>(std::round(base_iters * tier_scale(tier) * scale));
    p.t0 = 0.15;
    p.t1 = 0.01;
    p.w_micro = 0.8;
    p.w_swap_rot = 0.25;
    p.w_relocate = 0.15;
    p.w_block_translate = (tier == TargetTier::kA) ? 0.08 : (tier == TargetTier::kB ? 0.06 : 0.04);
    p.w_block_rotate = (tier == TargetTier::kA) ? 0.03 : (tier == TargetTier::kB ? 0.02 : 0.015);
    p.w_lns = (tier == TargetTier::kA) ? 0.02 : (tier == TargetTier::kB ? 0.015 : 0.01);
    p.w_push_contact = (tier == TargetTier::kA) ? 0.20 : (tier == TargetTier::kB ? 0.15 : 0.10);
    p.w_squeeze = (tier == TargetTier::kA) ? 0.08 : (tier == TargetTier::kB ? 0.06 : 0.04);
    p.block_size = 6;
    p.lns_remove = std::max(6, n / 25);
    p.lns_attempts_per_tree = (tier == TargetTier::kA) ? 40 : (tier == TargetTier::kB ? 30 : 20);
    p.lns_candidates = (tier == TargetTier::kA) ? 3 : (tier == TargetTier::kB ? 2 : 1);
    p.lns_p_contact = 0.55;
    p.lns_p_uniform = 0.15;
    p.push_bisect_iters = 12;
    p.push_max_step_frac = 0.9;
    p.plateau_eps = 1e-4;
    p.quantize_decimals = 9;
    return p;
}

bool use_soft_overlap(TargetTier tier, const Options& opt) {
    if (!opt.target_soft_overlap) {
        return false;
    }
    if (!opt.target_soft_overlap_tier_a_only) {
        return true;
    }
    return tier == TargetTier::kA;
}

std::vector<TreePose> run_targeted_sa(SARefiner& sa,
                                      const std::vector<TreePose>& start,
                                      const Options& opt,
                                      TargetTier tier,
                                      int n,
                                      uint64_t seed_base) {
    SARefiner::Params base = make_targeted_sa_params(tier, opt, n);
    if (base.iters <= 0) {
        return start;
    }

    if (use_soft_overlap(tier, opt)) {
        const int total_iters = base.iters;
        const int soft_iters = std::max(
            1, std::min(total_iters,
                        static_cast<int>(std::round(total_iters * opt.target_soft_overlap_cut))));
        const int hard_iters = total_iters - soft_iters;

        std::vector<TreePose> seed_poses = start;
        if (soft_iters > 0) {
            SARefiner::Params soft = base;
            soft.iters = soft_iters;
            soft.overlap_metric = SARefiner::OverlapMetric::kArea;
            soft.overlap_weight = 0.2;
            soft.overlap_weight_start = 0.2;
            soft.overlap_weight_end = 2e4;
            soft.overlap_weight_geometric = true;
            soft.w_resolve_overlap = 0.05;
            soft.push_overshoot_frac = 0.10;
            apply_sa_early_stop(soft, tier, opt);
            SARefiner::Result res =
                sa.refine_min_side(seed_poses, seed_base ^ 0xB5C8D57A3E4F29B1ULL, soft);
            seed_poses = res.best_poses;
        }

        if (hard_iters > 0) {
            SARefiner::Params hard = base;
            hard.iters = hard_iters;
            hard.overlap_weight = 0.0;
            hard.overlap_weight_start = -1.0;
            hard.overlap_weight_end = -1.0;
            hard.overlap_weight_geometric = false;
            hard.push_overshoot_frac = 0.0;
            hard.w_resolve_overlap = 0.0;
            apply_sa_early_stop(hard, tier, opt);
            SARefiner::Result res =
                sa.refine_min_side(seed_poses, seed_base ^ 0x8D1F5A9C7E3B2A41ULL, hard);
            return res.best_poses;
        }
        return seed_poses;
    }

    apply_sa_early_stop(base, tier, opt);
    SARefiner::Result res = sa.refine_min_side(start, seed_base, base);
    return res.best_poses;
}

double recompute_total_score(const Polygon& base_poly,
                             const std::vector<std::vector<TreePose>>& solutions_by_n,
                             int n_max) {
    double total = 0.0;
    for (int n = 1; n <= n_max; ++n) {
        const auto& sol = solutions_by_n[static_cast<size_t>(n)];
        if (static_cast<int>(sol.size()) != n) {
            continue;
        }
        const double side =
            bounding_square_side(transformed_polygons(base_poly, sol));
        total += (side * side) / static_cast<double>(n);
    }
    return total;
}

void run_targeted_refine(const Polygon& base_poly,
                         double radius,
                         std::vector<std::vector<TreePose>>& solutions_by_n,
                         const Options& opt,
                         double& total_score) {
    if (!opt.target_refine) {
        return;
    }

    const int rounds = std::max(1, opt.target_rounds);
    const double plateau_eps = 1e-4;
    SARefiner sa(base_poly, radius);
    double best_total = total_score;

    for (int round = 1; round <= rounds; ++round) {
        double total_score_est = 0.0;
        std::vector<TargetEntry> targets =
            select_target_entries(base_poly, solutions_by_n, opt, &total_score_est);
        if (targets.empty()) {
            return;
        }

        int improved_count = 0;
        int tier_a = 0;
        int tier_b = 0;
        for (const auto& entry : targets) {
            auto& sol = solutions_by_n[static_cast<size_t>(entry.n)];
            if (static_cast<int>(sol.size()) != entry.n) {
                continue;
            }

            if (entry.tier == TargetTier::kA) {
                ++tier_a;
            } else if (entry.tier == TargetTier::kB) {
                ++tier_b;
            }

            auto base_q = quantize_poses(sol);
            if (any_overlap(base_poly, base_q, radius)) {
                continue;
            }

            SideArea best_eval = eval_side_area(base_poly, base_q);
            SideArea orig_eval = best_eval;
            std::vector<TreePose> best_sol = base_q;

            uint64_t seed = opt.seed ^
                            (0xC6BC279692B5CC83ULL +
                             static_cast<uint64_t>(entry.n) * 0x9E3779B97F4A7C15ULL);
            std::mt19937_64 rng(seed);
            compaction_contact::Params cparams = make_compact_params(entry.tier, opt);
            std::vector<TreePose> cand = best_sol;
            compaction_contact::Stats cstats =
                compaction_contact::compact_contact(base_poly, cand, cparams, rng);
            if (cstats.ok) {
                auto cand_q = quantize_poses(cand);
                if (!any_overlap(base_poly, cand_q, radius)) {
                    SideArea cand_eval = eval_side_area(base_poly, cand_q);
                    if (improves_side_area(cand_eval, best_eval, plateau_eps)) {
                        best_eval = cand_eval;
                        best_sol = std::move(cand_q);
                    }
                }
            }

            uint64_t sa_seed = opt.seed ^
                               (0x9E3779B97F4A7C15ULL +
                                static_cast<uint64_t>(entry.n) * 0xBF58476D1CE4E5B9ULL);
            std::vector<TreePose> sa_out =
                run_targeted_sa(sa, best_sol, opt, entry.tier, entry.n, sa_seed);
            if (!sa_out.empty()) {
                auto sa_q = quantize_poses(sa_out);
                if (!any_overlap(base_poly, sa_q, radius)) {
                    SideArea sa_eval = eval_side_area(base_poly, sa_q);
                    if (improves_side_area(sa_eval, best_eval, plateau_eps)) {
                        best_eval = sa_eval;
                        best_sol = std::move(sa_q);
                    }
                }
            }

            if (improves_side_area(best_eval, orig_eval, plateau_eps)) {
                ++improved_count;
            }
            sol = std::move(best_sol);
        }

        const int tier_c = static_cast<int>(targets.size()) - tier_a - tier_b;
        total_score = recompute_total_score(base_poly, solutions_by_n, opt.n_max);
        std::cout << "Targeted refine round " << round << "/" << rounds << ": "
                  << targets.size() << " targets"
                  << " (tierA=" << tier_a
                  << ", tierB=" << tier_b
                  << ", tierC=" << tier_c
                  << ", cover=" << std::fixed << std::setprecision(3)
                  << std::min(1.0, std::max(0.0, opt.target_cover))
                  << ", improved=" << improved_count
                  << ", total=" << std::setprecision(6) << total_score
                  << ")\n";

        if (total_score >= best_total - 1e-9) {
            break;
        }
        best_total = total_score;
    }
}

double total_score_min_sides(const std::vector<double>& prefix_side_by_n,
                             const std::vector<double>* prune_side_by_n,
                             int n_max,
                             double best_cap) {
    double total = 0.0;
    for (int n = 1; n <= n_max; ++n) {
        double s = prefix_side_by_n[static_cast<size_t>(n)];
        if (prune_side_by_n) {
            s = std::min(s, (*prune_side_by_n)[static_cast<size_t>(n)]);
        }
        total += (s * s) / static_cast<double>(n);
        if (total >= best_cap) {
            break;
        }
    }
    return total;
}

Eval choose_best_angle_for_prefix_score(const Polygon& base_poly,
                                        const Pattern& pattern,
                                        double radius,
                                        double spacing,
                                        const Options& opt) {
    Eval best;
    for (double ang : opt.angle_candidates) {
        auto pool = opt.pool_window_scan
                        ? generate_windowed_tiling(opt.pool_size,
                                                   spacing,
                                                   ang,
                                                   pattern,
                                                   base_poly,
                                                   opt.pool_window_radius,
                                                   opt.n_max)
                        : generate_ordered_tiling(opt.pool_size, spacing, ang, pattern);
        if (any_overlap(base_poly, pool, radius)) {
            continue;
        }

        // A seleção do melhor ângulo precisa refletir exatamente o que vai pro
        // submission (valores quantizados em string). Caso contrário, o ranking
        // entre ângulos pode inverter após o arredondamento.
        auto pool_q = quantize_poses(pool);
        if (any_overlap(base_poly, pool_q, radius)) {
            continue;
        }

        std::vector<TreePose> prefix;
        if (opt.prefix_order == "greedy") {
            prefix = greedy_prefix_min_side(base_poly, pool_q, opt.n_max);
        } else if (opt.prefix_order == "central") {
            prefix = std::vector<TreePose>(pool_q.begin(), pool_q.begin() + opt.n_max);
        } else {
            throw std::runtime_error("--prefix-order inválido: " + opt.prefix_order);
        }

        std::vector<double> prefix_side_by_n =
            prefix_sides_from_bbs(bounding_boxes_for_poses(base_poly, prefix));

        double total = 0.0;
        if (opt.prune) {
            std::vector<double> prune_side_by_n = greedy_pruned_sides(
                bounding_boxes_for_poses(base_poly, pool_q), opt.n_max, 1e-12);
            for (int n = 1; n <= opt.n_max; ++n) {
                double s =
                    std::min(prefix_side_by_n[static_cast<size_t>(n)],
                             prune_side_by_n[static_cast<size_t>(n)]);
                total += (s * s) / static_cast<double>(n);
            }
        } else {
            total = total_score_from_sides(prefix_side_by_n, opt.n_max);
        }

        if (total < best.best_total) {
            best.best_total = total;
            best.best_angle = ang;
            best.best_poses = std::move(pool);
        }
    }
    if (!std::isfinite(best.best_total)) {
        throw std::runtime_error("Nenhum ângulo candidato gerou configuração válida.");
    }
    return best;
}

struct ShiftFullEval {
    double total = std::numeric_limits<double>::infinity();
    double best_angle = 0.0;
    bool ok = false;
};

double eval_shift_score_fast(const Polygon& base_poly,
                             const Pattern& pattern,
                             const Options& opt,
                             double spacing,
                             int pool_size,
                             int n_eval) {
    if (pool_size < n_eval) {
        pool_size = n_eval;
    }
    std::vector<TreePose> pool = opt.pool_window_scan
                                     ? generate_windowed_tiling(pool_size,
                                                                spacing,
                                                                0.0,
                                                                pattern,
                                                                base_poly,
                                                                opt.pool_window_radius,
                                                                n_eval)
                                     : generate_ordered_tiling(pool_size, spacing, 0.0, pattern);
    std::vector<TreePose> prefix(pool.begin(), pool.begin() + n_eval);
    std::vector<double> side_by_n =
        prefix_sides_from_bbs(bounding_boxes_for_poses(base_poly, prefix));
    return total_score_from_sides(side_by_n, n_eval);
}

ShiftFullEval eval_shift_score_full(const Polygon& base_poly,
                                    const Pattern& pattern,
                                    double radius,
                                    double spacing,
                                    const Options& opt,
                                    int pool_size,
                                    int n_eval) {
    ShiftFullEval best;
    pool_size = std::max(pool_size, n_eval);

    for (double ang : opt.angle_candidates) {
        std::vector<TreePose> pool = opt.pool_window_scan
                                         ? generate_windowed_tiling(pool_size,
                                                                    spacing,
                                                                    ang,
                                                                    pattern,
                                                                    base_poly,
                                                                    opt.pool_window_radius,
                                                                    n_eval)
                                         : generate_ordered_tiling(pool_size, spacing, ang, pattern);
        if (any_overlap(base_poly, pool, radius)) {
            continue;
        }
        auto pool_q = quantize_poses(pool);
        if (any_overlap(base_poly, pool_q, radius)) {
            continue;
        }

        std::vector<TreePose> prefix_nmax;
        if (opt.prefix_order == "greedy") {
            prefix_nmax = greedy_prefix_min_side(base_poly, pool_q, n_eval);
        } else if (opt.prefix_order == "central") {
            prefix_nmax = std::vector<TreePose>(pool_q.begin(), pool_q.begin() + n_eval);
        } else {
            throw std::runtime_error("--prefix-order inválido: " + opt.prefix_order);
        }

        std::vector<double> prefix_side_by_n =
            prefix_sides_from_bbs(bounding_boxes_for_poses(base_poly, prefix_nmax));

        double total = 0.0;
        if (opt.prune) {
            std::vector<double> prune_side_by_n =
                greedy_pruned_sides(bounding_boxes_for_poses(base_poly, pool_q), n_eval, 1e-12);
            total = total_score_min_sides(prefix_side_by_n, &prune_side_by_n, n_eval, best.total);
        } else {
            total = total_score_min_sides(prefix_side_by_n, nullptr, n_eval, best.total);
        }

        if (total < best.total) {
            best.total = total;
            best.best_angle = ang;
            best.ok = true;
        }
    }

    return best;
}

struct ShiftSearchResult {
    double shift_a = 0.0;
    double shift_b = 0.0;
    double best_angle = 0.0;
    double best_total = std::numeric_limits<double>::infinity();
    bool ok = false;
};

ShiftSearchResult shift_search_multires(const Polygon& base_poly,
                                        const Pattern& base_pattern,
                                        double radius,
                                        double spacing,
                                        const Options& opt) {
    ShiftSearchResult res;

    const int n_eval = std::min(opt.tile_score_nmax, opt.n_max);
    int pool_fast = opt.tile_score_pool_size > 0 ? opt.tile_score_pool_size : opt.pool_size;
    pool_fast = std::max(pool_fast, n_eval);

    int pool_full = opt.shift_pool_size > 0 ? opt.shift_pool_size : opt.pool_size;
    pool_full = std::max(pool_full, opt.n_max);

    const int grid = opt.shift_grid;
    const int keep = opt.shift_keep;
    const int levels = opt.shift_levels;

    struct Cand {
        double a = 0.0;
        double b = 0.0;
        double fast = std::numeric_limits<double>::infinity();
    };

    auto eval_fast = [&](double a, double b) -> double {
        Pattern p = base_pattern;
        p.shift_a = wrap01(a);
        p.shift_b = wrap01(b);
        return eval_shift_score_fast(base_poly, p, opt, spacing, pool_fast, n_eval);
    };

    auto keep_top = [&](std::vector<Cand>& cands) {
        std::sort(cands.begin(), cands.end(), [](const Cand& x, const Cand& y) {
            if (x.fast != y.fast) {
                return x.fast < y.fast;
            }
            if (x.a != y.a) {
                return x.a < y.a;
            }
            return x.b < y.b;
        });
        if (static_cast<int>(cands.size()) > keep) {
            cands.resize(static_cast<size_t>(keep));
        }
    };

    std::vector<Cand> frontier;
    frontier.reserve(static_cast<size_t>(grid * grid));
    for (int ia = 0; ia < grid; ++ia) {
        for (int ib = 0; ib < grid; ++ib) {
            double a = (static_cast<double>(ia) + 0.5) / static_cast<double>(grid);
            double b = (static_cast<double>(ib) + 0.5) / static_cast<double>(grid);
            frontier.push_back(Cand{wrap01(a), wrap01(b), eval_fast(a, b)});
        }
    }
    keep_top(frontier);

    double step = 1.0 / static_cast<double>(grid);
    for (int lvl = 1; lvl < levels; ++lvl) {
        step *= 0.5;
        std::vector<Cand> next;
        next.reserve(frontier.size() * 9);

        for (const auto& c : frontier) {
            for (int da = -1; da <= 1; ++da) {
                for (int db = -1; db <= 1; ++db) {
                    double a = wrap01(c.a + static_cast<double>(da) * step);
                    double b = wrap01(c.b + static_cast<double>(db) * step);
                    next.push_back(Cand{a, b, eval_fast(a, b)});
                }
            }
        }

        std::sort(next.begin(), next.end(), [](const Cand& x, const Cand& y) {
            if (x.a != y.a) {
                return x.a < y.a;
            }
            return x.b < y.b;
        });
        next.erase(std::unique(next.begin(),
                               next.end(),
                               [](const Cand& x, const Cand& y) {
                                   return std::abs(x.a - y.a) <= 1e-15 &&
                                          std::abs(x.b - y.b) <= 1e-15;
                               }),
                   next.end());

        frontier = std::move(next);
        keep_top(frontier);
    }

    for (const auto& c : frontier) {
        Pattern p = base_pattern;
        p.shift_a = c.a;
        p.shift_b = c.b;
        ShiftFullEval full = eval_shift_score_full(
            base_poly, p, radius, spacing, opt, pool_full, opt.n_max);
        if (!full.ok) {
            continue;
        }
        if (full.total < res.best_total) {
            res.best_total = full.total;
            res.shift_a = c.a;
            res.shift_b = c.b;
            res.best_angle = full.best_angle;
            res.ok = true;
        }
    }

    return res;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);

        Polygon base_poly = get_tree_polygon();
        const double radius = enclosing_circle_radius(base_poly);
        const double eps = 1e-9;

        Pattern pattern = make_initial_pattern(opt.k);
        pattern.lattice_v_ratio = opt.lattice_v_ratio;
        pattern.lattice_theta_deg = opt.lattice_theta_deg;
        pattern = optimize_tile_by_spacing(base_poly, pattern, radius, opt);
        pattern.shift_a = wrap01(opt.shift_a);
        pattern.shift_b = wrap01(opt.shift_b);

        double min_spacing =
            find_min_safe_spacing(base_poly, pattern, radius, eps, 2.0 * radius);
        if (!std::isfinite(min_spacing)) {
            throw std::runtime_error("Não foi possível encontrar spacing seguro.");
        }
        const double spacing = min_spacing * opt.spacing_safety;

        if (opt.shift_search == ShiftSearchMode::kMultires) {
            ShiftSearchResult ss =
                shift_search_multires(base_poly, pattern, radius, spacing, opt);
            if (!ss.ok) {
                throw std::runtime_error("Shift search falhou: nenhum candidato válido.");
            }
            pattern.shift_a = ss.shift_a;
            pattern.shift_b = ss.shift_b;
            std::cout << "Shift search: a=" << std::fixed << std::setprecision(9)
                      << pattern.shift_a << " b=" << pattern.shift_b
                      << " (score=" << std::setprecision(9) << ss.best_total
                      << ", angle=" << std::setprecision(3) << ss.best_angle << ")\n";
        }

        Eval chosen = choose_best_angle_for_prefix_score(
            base_poly, pattern, radius, spacing, opt);

        std::vector<TreePose> poses_pool = std::move(chosen.best_poses);

        BoundaryRefineParams params;
        params.radius = radius;
        params.iters = opt.refine_iters;
        params.seed = opt.seed + 999;
        params.step_hint = spacing;
        refine_boundary(base_poly, poses_pool, params);

        // Reordena por "centralidade" após o refino (melhora recortes n pequenos).
        {
            std::vector<Candidate> tmp;
            tmp.reserve(poses_pool.size());
            for (const auto& pose : poses_pool) {
                tmp.push_back({pose,
                               std::max(std::abs(pose.x), std::abs(pose.y)),
                               std::hypot(pose.x, pose.y)});
            }
            std::sort(tmp.begin(),
                      tmp.end(),
                      [](const Candidate& a, const Candidate& b) {
                          if (a.key1 != b.key1) {
                              return a.key1 < b.key1;
                          }
                          if (a.key2 != b.key2) {
                              return a.key2 < b.key2;
                          }
                          if (a.pose.x != b.pose.x) {
                              return a.pose.x < b.pose.x;
                          }
                          if (a.pose.y != b.pose.y) {
                              return a.pose.y < b.pose.y;
                          }
                          return a.pose.deg < b.pose.deg;
                      });
            for (size_t i = 0; i < poses_pool.size(); ++i) {
                poses_pool[i] = tmp[i].pose;
            }
        }

        if (any_overlap(base_poly, poses_pool, radius)) {
            throw std::runtime_error("Overlap detectado após refino.");
        }

        auto poses_pool_q = quantize_poses(poses_pool);
        if (any_overlap(base_poly, poses_pool_q, radius)) {
            throw std::runtime_error(
                "Overlap detectado após arredondamento para o submission.");
        }

        std::vector<TreePose> prefix_nmax;
        if (opt.prefix_order == "greedy") {
            prefix_nmax = greedy_prefix_min_side(base_poly, poses_pool_q, opt.n_max);
        } else {
            prefix_nmax = std::vector<TreePose>(poses_pool_q.begin(),
                                                poses_pool_q.begin() + opt.n_max);
        }

        std::vector<std::vector<TreePose>> prefix_by_n;
        prefix_by_n.resize(static_cast<size_t>(opt.n_max + 1));
        for (int n = 1; n <= opt.n_max; ++n) {
            prefix_by_n[static_cast<size_t>(n)] =
                std::vector<TreePose>(prefix_nmax.begin(), prefix_nmax.begin() + n);
        }

        std::vector<double> prefix_side_by_n =
            prefix_sides_from_bbs(bounding_boxes_for_poses(base_poly, prefix_nmax));

        std::vector<std::vector<TreePose>> solutions_by_n;
        solutions_by_n.resize(static_cast<size_t>(opt.n_max + 1));

        double total_score = 0.0;
        if (opt.prune) {
            PruneResult pr =
                build_greedy_pruned_solutions(base_poly, poses_pool_q, opt.n_max);
            for (int n = 1; n <= opt.n_max; ++n) {
                double s_prefix = prefix_side_by_n[static_cast<size_t>(n)];
                double s_prune = pr.side_by_n[static_cast<size_t>(n)];
                if (s_prune + 1e-15 < s_prefix) {
                    solutions_by_n[static_cast<size_t>(n)] = pr.solutions_by_n[static_cast<size_t>(n)];
                    total_score += (s_prune * s_prune) / static_cast<double>(n);
                } else {
                    solutions_by_n[static_cast<size_t>(n)] = prefix_by_n[static_cast<size_t>(n)];
                    total_score += (s_prefix * s_prefix) / static_cast<double>(n);
                }
            }
        } else {
            for (int n = 1; n <= opt.n_max; ++n) {
                double s = prefix_side_by_n[static_cast<size_t>(n)];
                solutions_by_n[static_cast<size_t>(n)] = std::move(prefix_by_n[static_cast<size_t>(n)]);
                total_score += (s * s) / static_cast<double>(n);
            }
        }

        if (opt.mz_its_iters > 0) {
            std::vector<TreePose>& sol_nmax =
                solutions_by_n[static_cast<size_t>(opt.n_max)];
            double side_before = bounding_square_side(
                transformed_polygons(base_poly, sol_nmax));
            ILSResult mz = mz_its_soft_compact(
                base_poly,
                radius,
                sol_nmax,
                opt.seed ^
                    (0xC6BC279692B5CC83ULL +
                     static_cast<uint64_t>(opt.n_max) * 0x9E3779B97F4A7C15ULL),
                opt);
            double side_after = bounding_square_side(
                transformed_polygons(base_poly, sol_nmax));
            if (mz.improved) {
                total_score +=
                    ((side_after * side_after) - (side_before * side_before)) /
                    static_cast<double>(opt.n_max);
            }
            if (any_overlap(base_poly, sol_nmax, radius)) {
                throw std::runtime_error("Overlap detectado após MZ-ITS.");
            }
            std::cout << "MZ-ITS (n=" << opt.n_max << "): "
                      << (mz.improved ? "improved" : "no-improve")
                      << " (attempts=" << mz.attempts
                      << ", accepted=" << mz.accepted
                      << ", side=" << std::fixed << std::setprecision(9)
                      << mz.start_side << " -> " << mz.best_side
                      << ")\n";
        }

        if (opt.ils_iters > 0) {
            std::vector<TreePose>& sol_nmax =
                solutions_by_n[static_cast<size_t>(opt.n_max)];
            double side_before = bounding_square_side(
                transformed_polygons(base_poly, sol_nmax));
            ILSResult ils = ils_basin_hop_compact(
                base_poly,
                radius,
                sol_nmax,
                opt.seed ^
                    (0xA24BAED4963EE407ULL +
                     static_cast<uint64_t>(opt.n_max) * 0x9E3779B97F4A7C15ULL),
                opt);
            double side_after = bounding_square_side(
                transformed_polygons(base_poly, sol_nmax));
            if (ils.improved) {
                total_score +=
                    ((side_after * side_after) - (side_before * side_before)) /
                    static_cast<double>(opt.n_max);
            }
            std::cout << "ILS (n=" << opt.n_max << "): "
                      << (ils.improved ? "improved" : "no-improve")
                      << " (attempts=" << ils.attempts
                      << ", accepted=" << ils.accepted
                      << ", side=" << std::fixed << std::setprecision(9)
                      << ils.start_side << " -> " << ils.best_side
                      << ")\n";
        }

        if (opt.global_contract_steps > 0) {
            std::vector<TreePose>& sol_nmax =
                solutions_by_n[static_cast<size_t>(opt.n_max)];
            GlobalContractionResult gc = apply_global_contraction_nmax(
                base_poly,
                radius,
                sol_nmax,
                opt.seed ^
                    (0x5a1fc2af0e3f3ab1ULL +
                     static_cast<uint64_t>(opt.n_max) * 0x9E3779B97F4A7C15ULL),
                opt);

            if (gc.improved) {
                total_score +=
                    ((gc.best_side * gc.best_side) - (gc.start_side * gc.start_side)) /
                    static_cast<double>(opt.n_max);
            }
            if (any_overlap(base_poly, sol_nmax, radius)) {
                throw std::runtime_error("Overlap detectado após global contraction.");
            }
            std::cout << "Global contraction (n=" << opt.n_max << "): "
                      << (gc.improved ? "improved" : "no-improve")
                      << " (steps_ok=" << gc.steps_ok
                      << "/" << opt.global_contract_steps
                      << ", side=" << std::fixed << std::setprecision(9)
                      << gc.start_side << " -> " << gc.best_side
                      << ")\n";
        }

        const bool use_beam = opt.sa_beam && opt.sa_beam_width > 0;
        const bool use_chain = opt.sa_chain &&
                               (opt.sa_chain_base_iters > 0 ||
                                opt.sa_chain_iters_per_n > 0);
        if (use_beam || use_chain) {
            std::vector<double> base_side_by_n;
            base_side_by_n.resize(static_cast<size_t>(opt.n_max + 1), 0.0);
            for (int n = 1; n <= opt.n_max; ++n) {
                base_side_by_n[static_cast<size_t>(n)] =
                    bounding_square_side(transformed_polygons(
                        base_poly, solutions_by_n[static_cast<size_t>(n)]));
            }

            const double band_step = spacing * std::min(1.0, pattern.lattice_v_ratio);
            ChainResult cr = use_beam
                                 ? build_sa_beam_chain_solutions(base_poly,
                                                                 radius,
                                                                 solutions_by_n[static_cast<size_t>(opt.n_max)],
                                                                 opt.n_max,
                                                                 band_step,
                                                                 opt)
                                 : build_sa_chain_solutions(base_poly,
                                                            radius,
                                                            solutions_by_n[static_cast<size_t>(opt.n_max)],
                                                            opt.n_max,
                                                            band_step,
                                                            opt);

            double total_score_chain = 0.0;
            for (int n = 1; n <= opt.n_max; ++n) {
                double best_side = base_side_by_n[static_cast<size_t>(n)];
                std::vector<TreePose> best_sol = solutions_by_n[static_cast<size_t>(n)];

                if (!cr.solutions_by_n[static_cast<size_t>(n)].empty() &&
                    cr.side_by_n[static_cast<size_t>(n)] + 1e-15 < best_side) {
                    best_side = cr.side_by_n[static_cast<size_t>(n)];
                    best_sol = std::move(cr.solutions_by_n[static_cast<size_t>(n)]);
                }

                solutions_by_n[static_cast<size_t>(n)] = std::move(best_sol);
                total_score_chain += (best_side * best_side) / static_cast<double>(n);
            }
            total_score = total_score_chain;
        }

        if (opt.sa_restarts > 0 && opt.sa_base_iters > 0) {
            double total_score_sa = 0.0;
            SARefiner sa(base_poly, radius);
            std::vector<SARefiner::HHState> hh_states(
                static_cast<size_t>(std::max(0, opt.n_max) + 1));

            for (int n = 1; n <= opt.n_max; ++n) {
                SARefiner::Params p;
                p.iters = opt.sa_base_iters + opt.sa_iters_per_n * n;
                p.w_micro = opt.sa_w_micro;
                p.w_swap_rot = opt.sa_w_swap_rot;
                p.w_relocate = opt.sa_w_relocate;
                p.w_block_translate = opt.sa_w_block_translate;
                p.w_block_rotate = opt.sa_w_block_rotate;
                p.w_lns = opt.sa_w_lns;
                p.w_push_contact = opt.sa_w_push_contact;
                p.w_slide_contact = opt.sa_w_slide_contact;
                p.w_squeeze = opt.sa_w_squeeze;
                p.block_size = opt.sa_block_size;
                p.lns_remove = opt.sa_lns_remove;
                p.lns_candidates = opt.sa_lns_candidates;
                p.lns_eval_attempts_per_tree = opt.sa_lns_eval_attempts_per_tree;
                p.hh_segment = opt.sa_hh_segment;
                p.hh_reaction = opt.sa_hh_reaction;
                p.hh_auto = opt.sa_hh_auto;
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
                p.slide_dirs = opt.sa_slide_dirs;
                p.slide_dir_bias = opt.sa_slide_dir_bias;
                p.slide_max_step_frac = opt.sa_slide_max_step_frac;
                p.slide_bisect_iters = opt.sa_slide_bisect_iters;
                p.slide_min_gain = opt.sa_slide_min_gain;
                p.slide_schedule_max_frac = opt.sa_slide_schedule_max_frac;
                p.squeeze_pushes = opt.sa_squeeze_pushes;
                if (opt.sa_aggressive) {
                    SARefiner::apply_aggressive_preset(p);
                }
                if (opt.sa_hh_auto) {
                    SARefiner::apply_hh_auto_preset(p);
                }

                std::vector<TreePose> best_sol =
                    solutions_by_n[static_cast<size_t>(n)];
                double best_side = bounding_square_side(
                    transformed_polygons(base_poly, best_sol));

                for (int r = 0; r < opt.sa_restarts; ++r) {
                    uint64_t seed =
                        opt.seed ^
                        (0x9e3779b97f4a7c15ULL +
                         static_cast<uint64_t>(n) * 0xbf58476d1ce4e5b9ULL +
                         static_cast<uint64_t>(r) * 0x94d049bb133111ebULL);
                    SARefiner::HHState* hh_state =
                        opt.sa_hh_auto ? &hh_states[static_cast<size_t>(n)] : nullptr;
                    SARefiner::Result res =
                        sa.refine_min_side(best_sol, seed, p, nullptr, hh_state);

                    auto cand_q = quantize_poses(res.best_poses);
                    if (any_overlap(base_poly, cand_q, radius)) {
                        continue;
                    }
                    double cand_side = bounding_square_side(
                        transformed_polygons(base_poly, cand_q));
                    if (cand_side + 1e-15 < best_side) {
                        best_side = cand_side;
                        best_sol = std::move(cand_q);
                    }
                }

                solutions_by_n[static_cast<size_t>(n)] = best_sol;
                total_score_sa += (best_side * best_side) /
                                  static_cast<double>(n);
            }

            if (opt.sa_hh_auto) {
                dump_hh_bucket("1-25", aggregate_hh_states(hh_states, 1, 25));
                dump_hh_bucket("26-80", aggregate_hh_states(hh_states, 26, 80));
                dump_hh_bucket("81-200", aggregate_hh_states(hh_states, 81, opt.n_max));
            }

            total_score = total_score_sa;
        }

        run_targeted_refine(base_poly, radius, solutions_by_n, opt, total_score);

        // Pós-processamento "final rigid": otimiza um ângulo global por n.
        if (opt.final_rigid) {
            double total_score_rigid = 0.0;
            for (int n = 1; n <= opt.n_max; ++n) {
                std::vector<TreePose> best_sol =
                    solutions_by_n[static_cast<size_t>(n)];
                double best_side = bounding_square_side(
                    transformed_polygons(base_poly, best_sol));

                std::vector<TreePose> rigid_sol = best_sol;
                optimize_rigid_rotation(base_poly, rigid_sol);
                auto rigid_q = quantize_poses(rigid_sol);
                if (!any_overlap(base_poly, rigid_q, radius)) {
                    double rigid_side = bounding_square_side(
                        transformed_polygons(base_poly, rigid_q));
                    if (rigid_side + 1e-15 < best_side) {
                        best_side = rigid_side;
                        best_sol = std::move(rigid_q);
                        solutions_by_n[static_cast<size_t>(n)] = best_sol;
                    }
                }

                total_score_rigid += (best_side * best_side) /
                                     static_cast<double>(n);
            }
            total_score = total_score_rigid;
        }

        const bool use_micro =
            (opt.micro_rot_steps > 0 && opt.micro_rot_eps > 0.0) ||
            (opt.micro_shift_steps > 0 && opt.micro_shift_eps > 0.0);
        int micro_improved = 0;
        if (use_micro) {
            double total_score_micro = 0.0;
            const MicroAdjustOptions micro_opt{
                opt.micro_rot_eps,
                opt.micro_rot_steps,
                opt.micro_shift_eps,
                opt.micro_shift_steps,
                9
            };

            for (int n = 1; n <= opt.n_max; ++n) {
                std::vector<TreePose> best_sol =
                    solutions_by_n[static_cast<size_t>(n)];
                double best_side = bounding_square_side(
                    transformed_polygons(base_poly, best_sol));

                MicroAdjustOutcome micro_out = apply_micro_adjustments(
                    base_poly, best_sol, radius, micro_opt);
                if (micro_out.result.improved &&
                    micro_out.result.best_side + 1e-15 < best_side) {
                    best_side = micro_out.result.best_side;
                    best_sol = std::move(micro_out.poses);
                    ++micro_improved;
                }

                solutions_by_n[static_cast<size_t>(n)] = std::move(best_sol);
                total_score_micro += (best_side * best_side) /
                                     static_cast<double>(n);
            }
            total_score = total_score_micro;
        }

        if (opt.post_opt) {
            PostOptOptions post_opt;
            post_opt.enabled = true;
            post_opt.iters = opt.post_iters;
            post_opt.restarts = opt.post_restarts;
            post_opt.t0 = opt.post_t0;
            post_opt.tm = opt.post_tm;
            post_opt.enable_squeeze = opt.post_enable_squeeze;
            post_opt.enable_compaction = opt.post_enable_compaction;
            post_opt.enable_edge_slide = opt.post_enable_edge_slide;
            post_opt.enable_local_search = opt.post_enable_local_search;
            post_opt.remove_ratio = opt.post_remove_ratio;
            post_opt.free_area_min_n = opt.post_free_area_min_n;
            post_opt.enable_free_area = opt.post_enable_free_area;

            post_opt.enable_term_scheduler = opt.post_term_scheduler;
            post_opt.term_epochs = opt.post_term_epochs;
            post_opt.term_tier_a = opt.post_term_tier_a;
            post_opt.term_tier_b = opt.post_term_tier_b;
            post_opt.term_min_n = opt.post_term_min_n;
            post_opt.tier_a_iters_mult = opt.post_tier_a_iters_mult;
            post_opt.tier_a_restarts_mult = opt.post_tier_a_restarts_mult;
            post_opt.tier_b_iters_mult = opt.post_tier_b_iters_mult;
            post_opt.tier_b_restarts_mult = opt.post_tier_b_restarts_mult;
            post_opt.tier_c_iters_mult = opt.post_tier_c_iters_mult;
            post_opt.tier_c_restarts_mult = opt.post_tier_c_restarts_mult;
            post_opt.tier_a_tighten_mult = opt.post_tier_a_tighten_mult;
            post_opt.tier_b_tighten_mult = opt.post_tier_b_tighten_mult;
            post_opt.tier_c_tighten_mult = opt.post_tier_c_tighten_mult;
            post_opt.accept_term_eps = opt.post_accept_term_eps;

            post_opt.enable_guided_reinsert = opt.post_guided_reinsert;
            post_opt.reinsert_attempts_tier_a = opt.post_reinsert_attempts_tier_a;
            post_opt.reinsert_attempts_tier_b = opt.post_reinsert_attempts_tier_b;
            post_opt.reinsert_attempts_tier_c = opt.post_reinsert_attempts_tier_c;
            post_opt.reinsert_shell_anchors = opt.post_reinsert_shell_anchors;
            post_opt.reinsert_core_anchors = opt.post_reinsert_core_anchors;
            post_opt.reinsert_jitter_attempts = opt.post_reinsert_jitter_attempts;
            post_opt.reinsert_angle_jitter_deg = opt.post_reinsert_angle_jitter_deg;
            post_opt.reinsert_early_stop_rel = opt.post_reinsert_early_stop_rel;

            post_opt.enable_backprop = opt.post_enable_backprop;
            post_opt.backprop_passes = opt.post_backprop_passes;
            post_opt.backprop_span = opt.post_backprop_span;
            post_opt.backprop_max_combos = opt.post_backprop_max_combos;
            post_opt.enable_backprop_explore = opt.post_backprop_explore;
            post_opt.backprop_span_tier_a = opt.post_backprop_span_a;
            post_opt.backprop_span_tier_b = opt.post_backprop_span_b;
            post_opt.backprop_span_tier_c = opt.post_backprop_span_c;
            post_opt.backprop_max_combos_tier_a = opt.post_backprop_max_combos_a;
            post_opt.backprop_max_combos_tier_b = opt.post_backprop_max_combos_b;
            post_opt.backprop_max_combos_tier_c = opt.post_backprop_max_combos_c;
            post_opt.threads = opt.post_threads;
            post_opt.seed = opt.seed;

            PostOptStats post_stats;
            post_optimize_submission(base_poly, solutions_by_n, post_opt, &post_stats);
            total_score = post_stats.final_score;

            const std::ios::fmtflags flags = std::cout.flags();
            const std::streamsize precision = std::cout.precision();
            std::cout.setf(std::ios::fixed);
            std::cout << "Post-opt: on"
                      << " (iters=" << post_opt.iters
                      << ", restarts=" << post_opt.restarts
                      << ", free-area=" << (post_opt.enable_free_area ? "on" : "off")
                      << ", backprop=" << (post_opt.enable_backprop ? "on" : "off")
                      << ", threads=" << post_opt.threads << ")\n";
            std::cout << "Post-opt score: " << std::setprecision(9)
                      << post_stats.initial_score << " -> "
                      << post_stats.final_score
                      << " (phase1=" << post_stats.phase1_improved
                      << ", backprop=" << post_stats.backprop_improved
                      << ", time=" << std::setprecision(1)
                      << post_stats.elapsed_sec << "s)\n";

            if (post_opt.enable_term_scheduler && !post_stats.term_epochs.empty()) {
                const PostOptTermSummary& ts = post_stats.term_summary;
                const double drop_a = ts.tier_a_term_before - ts.tier_a_term_after;
                const double drop_b = ts.tier_b_term_before - ts.tier_b_term_after;
                const int k = ts.tier_a_count + ts.tier_b_count;

                std::cout << "Post-opt term tiers: A=" << ts.tier_a_count
                          << ", B=" << ts.tier_b_count
                          << " (K=" << k << ")"
                          << ", epochs=" << post_opt.term_epochs
                          << ", min_n=" << post_opt.term_min_n << "\n";
                std::cout << "Tier A term: " << std::setprecision(9)
                          << ts.tier_a_term_before << " -> " << ts.tier_a_term_after
                          << " (drop=" << drop_a << ")\n";
                std::cout << "Tier B term: " << std::setprecision(9)
                          << ts.tier_b_term_before << " -> " << ts.tier_b_term_after
                          << " (drop=" << drop_b << ")\n";
                std::cout << "Tier A n: {";
                for (size_t i = 0; i < ts.tier_a_ns.size(); ++i) {
                    if (i) {
                        std::cout << ",";
                    }
                    std::cout << ts.tier_a_ns[i];
                }
                std::cout << "}\n";

                for (const PostOptTermEpochStats& e : post_stats.term_epochs) {
                    const double edrop_a = e.tier_a_term_before - e.tier_a_term_after;
                    const double edrop_b = e.tier_b_term_before - e.tier_b_term_after;
                    std::cout << "Epoch " << e.epoch
                              << ": A drop=" << std::setprecision(9) << edrop_a
                              << ", B drop=" << std::setprecision(9) << edrop_b << "\n";
                }
            }
            std::cout.flags(flags);
            std::cout.precision(precision);
        }

        std::ofstream out(opt.output_path);
        if (!out) {
            throw std::runtime_error("Erro ao abrir arquivo de saída: " + opt.output_path);
        }
        out << "id,x,y,deg\n";
        for (int n = 1; n <= opt.n_max; ++n) {
            const auto& sol = solutions_by_n[static_cast<size_t>(n)];
            for (int i = 0; i < n; ++i) {
                const auto& pose = sol[static_cast<size_t>(i)];
                out << std::setw(3) << std::setfill('0') << n << "_" << i << ","
                    << fmt_submission_value(pose.x) << ","
                    << fmt_submission_value(pose.y) << ","
                    << fmt_submission_value(pose.deg) << "\n";
            }
        }

        std::cout << "Submission gerada em " << opt.output_path << "\n";
        std::cout << "k (tile): " << opt.k << "\n";
        std::cout << "Pool size: " << opt.pool_size << "\n";
        std::cout << "Pool window: " << (opt.pool_window_scan ? "on" : "off");
        if (opt.pool_window_scan) {
            std::cout << " (radius=" << opt.pool_window_radius << ")";
        }
        std::cout << "\n";
        std::cout << "Prefix order: " << opt.prefix_order << "\n";
        std::cout << "Tile iters: " << opt.tile_iters << "\n";
        std::cout << "Refine iters: " << opt.refine_iters << "\n";
        std::cout << "Lattice v_ratio: " << std::fixed << std::setprecision(6)
                  << pattern.lattice_v_ratio << "\n";
        std::cout << "Lattice theta: " << std::fixed << std::setprecision(3)
                  << pattern.lattice_theta_deg << "\n";
        std::cout << "Min spacing: " << std::fixed << std::setprecision(9) << min_spacing << "\n";
        std::cout << "Spacing (safety): " << std::fixed << std::setprecision(9) << spacing << "\n";
        std::cout << "Best angle: " << std::fixed << std::setprecision(3) << chosen.best_angle << "\n";
        std::cout << "Score (local): " << std::fixed << std::setprecision(9) << total_score << "\n";
        std::cout << "Prune: " << (opt.prune ? "on" : "off") << "\n";
        std::cout << "Final rigid: " << (opt.final_rigid ? "on" : "off") << "\n";
        std::cout << "Target refine: " << (opt.target_refine ? "on" : "off") << "\n";
        std::cout << "Micro adjust: " << (use_micro ? "on" : "off");
        if (use_micro) {
            std::cout << " (rot_eps=" << opt.micro_rot_eps
                      << ", rot_steps=" << opt.micro_rot_steps
                      << ", shift_eps=" << opt.micro_shift_eps
                      << ", shift_steps=" << opt.micro_shift_steps
                      << ", improved=" << micro_improved << ")";
        }
        std::cout << "\n";
        if (opt.sa_beam) {
            std::cout << "SA beam: on (width=" << opt.sa_beam_width
                      << ", remove=" << opt.sa_beam_remove
                      << ", micro=" << opt.sa_beam_micro_iters
                      << ", init=" << opt.sa_beam_init_iters
                      << ", band_layers=" << std::fixed << std::setprecision(3)
                      << opt.sa_chain_band_layers << ")\n";
        } else {
            std::cout << "SA beam: off\n";
        }
        if (opt.sa_chain && !opt.sa_beam) {
            std::cout << "SA chain: on (base=" << opt.sa_chain_base_iters
                      << ", per_n=" << opt.sa_chain_iters_per_n
                      << ", min_n=" << opt.sa_chain_min_n
                      << ", band_layers=" << std::fixed << std::setprecision(3)
                      << opt.sa_chain_band_layers << ")\n";
        } else if (!opt.sa_beam) {
            std::cout << "SA chain: off\n";
        }
        std::cout << "SA HH mode: " << (opt.sa_hh_auto ? "auto" : "off") << "\n";
        std::cout << "SA aggressive: " << (opt.sa_aggressive ? "on" : "off") << "\n";

    } catch (const std::exception& ex) {
        std::cerr << "Erro: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
