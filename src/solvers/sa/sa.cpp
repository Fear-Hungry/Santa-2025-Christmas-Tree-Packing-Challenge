#include "solvers/sa.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <cstring>
#include <random>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "geometry/collision.hpp"
#include "sa_geometry.hpp"
#include "utils/submission_io.hpp"
#include "utils/wrap_utils.hpp"

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
    p.w_slide_contact = std::max(p.w_slide_contact, 0.12);
    p.w_squeeze = std::max(p.w_squeeze, 0.06);
    p.w_global_rotate = std::max(p.w_global_rotate, 0.02);
    p.w_eject_chain = std::max(p.w_eject_chain, 0.01);

    p.block_size = std::max(p.block_size, 8);
    p.lns_remove = std::max(p.lns_remove, 10);
    p.lns_candidates = std::max(p.lns_candidates, 8);

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
    p.global_rot_deg = std::max(p.global_rot_deg, 0.05);
    p.eject_relax_iters = std::max(p.eject_relax_iters, 4);
    p.eject_reinsert_attempts = std::max(p.eject_reinsert_attempts, 30);

    if (p.iters > 0 && p.reheat_iters <= 0) {
        p.reheat_iters = std::max(50, p.iters / 25);
    }
    p.reheat_mult = std::max(p.reheat_mult, 1.5);
    p.reheat_step_mult = std::max(p.reheat_step_mult, 1.3);
    p.reheat_max = std::max(p.reheat_max, 3);
}

void SARefiner::apply_hh_auto_preset(Params& p) {
    p.hh_auto = true;
    if (p.hh_segment <= 0) {
        p.hh_segment = 40;
    }
    if (!(p.hh_reaction > 0.0)) {
        p.hh_reaction = 0.25;
    }
    if (!(p.hh_min_weight > 0.0)) {
        p.hh_min_weight = 0.02;
    } else {
        p.hh_min_weight = std::min(p.hh_min_weight, 0.02);
    }
    p.hh_reward_best = std::max(p.hh_reward_best, 6.0);
    p.hh_reward_improve = std::max(p.hh_reward_improve, 2.0);
    p.hh_reward_accept = 0.0;

    p.w_swap_rot = std::max(p.w_swap_rot, 0.30);
    p.w_relocate = std::max(p.w_relocate, 0.20);
    p.w_block_translate = std::max(p.w_block_translate, 0.05);
    p.w_block_rotate = std::max(p.w_block_rotate, 0.02);
    p.w_lns = std::max(p.w_lns, 0.005);
    if (p.w_push_contact > 0.0) {
        p.w_push_contact = std::max(p.w_push_contact, 0.03);
    }
    if (p.w_slide_contact > 0.0) {
        p.w_slide_contact = std::max(p.w_slide_contact, 0.02);
    }
    if (p.w_squeeze > 0.0) {
        p.w_squeeze = std::max(p.w_squeeze, 0.01);
    }
    p.w_global_rotate = std::max(p.w_global_rotate, 0.005);
    if (p.w_eject_chain > 0.0) {
        p.w_eject_chain = std::max(p.w_eject_chain, 0.003);
    }
    p.w_resolve_overlap = std::max(p.w_resolve_overlap, 0.01);
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
