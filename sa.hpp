#pragma once

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "collision.hpp"
#include "geom.hpp"

class SARefiner {
public:
    struct Params {
        int iters = 0;
        double t0 = 0.15;
        double t1 = 0.01;
        double p_rot = 0.30;
        double ddeg_max = 20.0;
        double ddeg_min = 2.0;
        double step_frac_max = 0.10;
        double step_frac_min = 0.004;
        double p_pick_extreme = 0.95;
        int extreme_topk = 14;
        int rebuild_extreme_every = 25;
        double p_random_dir = 0.15;
        double kick_prob = 0.02;
        double kick_mult = 3.0;

        // Portfólio de movimentos ("hiperheurística"): pesos iniciais.
        // Setar um peso como 0 desliga o operador.
        double w_micro = 1.0;
        double w_swap_rot = 0.25;
        double w_relocate = 0.15;
        double w_block_translate = 0.05;
        double w_block_rotate = 0.02;
        double w_lns = 0.001;

        // Controlador adaptativo (ALNS-style).
        int hh_segment = 50;
        double hh_reaction = 0.20;
        double hh_min_weight = 0.05;
        double hh_max_block_weight = 0.15;
        double hh_max_lns_weight = 0.01;
        // Escala do reward: recompensa proporcional a Δs / custo do operador.
        double hh_reward_scale = 1000.0;
        double hh_reward_best = 5.0;
        double hh_reward_improve = 2.0;
        double hh_reward_accept = 0.0;

        // Relocate (macro-movimento).
        int relocate_attempts = 10;
        double relocate_pull_min = 0.50;
        double relocate_pull_max = 1.00;
        double relocate_noise_frac = 0.08;  // relativo a curr_side
        double relocate_p_rot = 0.70;

        // Block moves (macro): move um subconjunto coerente (vizinhos próximos).
        int block_size = 6;
        double block_step_frac_max = 0.25;
        double block_step_frac_min = 0.03;
        double block_p_random_dir = 0.10;
        double block_rot_deg_max = 25.0;
        double block_rot_deg_min = 3.0;

        // LNS (macro): remove um subconjunto da borda e reinsere por amostragem.
        int lns_remove = 6;
        int lns_attempts_per_tree = 30;
        double lns_p_uniform = 0.30;
        double lns_p_contact = 0.35;
        double lns_pull_min = 0.30;
        double lns_pull_max = 0.90;
        double lns_noise_frac = 0.15;  // relativo a curr_side
        double lns_p_rot = 0.60;
        double lns_box_mult = 1.05;
    };

    struct Result {
        std::vector<TreePose> best_poses;
        double best_side = std::numeric_limits<double>::infinity();
    };

    SARefiner(const Polygon& base_poly, double radius)
        : base_poly_(base_poly), radius_(radius) {}

    Result refine_min_side(const std::vector<TreePose>& start,
                           uint64_t seed,
                           const Params& p) const {
        const int n = static_cast<int>(start.size());
        if (n <= 0) {
            return Result{start, 0.0};
        }
        if (p.iters <= 0) {
            auto polys = transformed_polygons(base_poly_, start);
            return Result{start, bounding_square_side(polys)};
        }

        std::vector<TreePose> poses = start;
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

        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::normal_distribution<double> normal(0.0, 1.0);
        std::uniform_real_distribution<double> uni_deg(-1.0, 1.0);

        const double thr = 2.0 * radius_ + 1e-9;
        const double thr_sq = thr * thr;
        const double tol = 1e-12;

        enum Op : int {
            kMicro = 0,
            kSwapRot = 1,
            kRelocate = 2,
            kBlockTranslate = 3,
            kBlockRotate = 4,
            kLNS = 5,
            kNumOps = 6,
        };

        std::array<double, kNumOps> weights = {std::max(0.0, p.w_micro),
                                               std::max(0.0, p.w_swap_rot),
                                               std::max(0.0, p.w_relocate),
                                               std::max(0.0, p.w_block_translate),
                                               std::max(0.0, p.w_block_rotate),
                                               std::max(0.0, p.w_lns)};
        std::array<double, kNumOps> op_cost = {
            1.0,
            2.0,
            static_cast<double>(std::max(1, p.relocate_attempts)),
            static_cast<double>(std::max(1, p.block_size)),
            static_cast<double>(std::max(1, p.block_size)),
            static_cast<double>(std::max(1, p.lns_remove)) *
                static_cast<double>(std::max(1, p.lns_attempts_per_tree)),
        };
        std::array<char, kNumOps> enabled = {
            static_cast<char>(weights[kMicro] > 0.0),
            static_cast<char>(weights[kSwapRot] > 0.0),
            static_cast<char>(weights[kRelocate] > 0.0),
            static_cast<char>(weights[kBlockTranslate] > 0.0),
            static_cast<char>(weights[kBlockRotate] > 0.0),
            static_cast<char>(weights[kLNS] > 0.0),
        };
        if (!(enabled[kMicro] || enabled[kSwapRot] || enabled[kRelocate] ||
              enabled[kBlockTranslate] || enabled[kBlockRotate] || enabled[kLNS])) {
            enabled[kMicro] = 1;
            weights[kMicro] = 1.0;
        }

        std::array<double, kNumOps> op_score = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        std::array<int, kNumOps> op_uses = {0, 0, 0, 0, 0, 0};

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

        auto build_boundary_pool = [&]() -> std::vector<int> {
            const double boundary_tol = 1e-9;
            std::vector<int> boundary;
            boundary.reserve(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) {
                const auto& bb = bbs[static_cast<size_t>(i)];
                if (bb.min_x <= gmnx + boundary_tol || bb.max_x >= gmxx - boundary_tol ||
                    bb.min_y <= gmny + boundary_tol || bb.max_y >= gmxy - boundary_tol) {
                    boundary.push_back(i);
                }
            }
            if (boundary.empty()) {
                boundary = build_extreme_pool(bbs, p.extreme_topk);
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
            std::uniform_int_distribution<int> pick(0, n - 1);
            return PickedIndex{pick(rng), false};
        };

        auto pick_other_index = [&](int i) -> int {
            if (n <= 1) {
                return 0;
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
                              double old_side,
                              double new_side) {
            op_uses[static_cast<size_t>(op)] += 1;
            if (!accepted) {
                return;
            }
            const double denom = std::max(1e-12, old_side);
            const double rel_improve = (old_side - new_side) / denom;
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

        auto accept_move = [&](double old_side, double new_side, double T) -> bool {
            double delta = new_side - old_side;
            if (delta <= 0.0) {
                return true;
            }
            if (!(T > 0.0)) {
                return false;
            }
            double rel = delta / std::max(1e-12, old_side);
            double prob = std::exp(-rel / std::max(1e-12, T));
            return (uni(rng) < prob);
        };

        auto build_block = [&](int anchor, int want) -> std::vector<int> {
            want = std::max(1, std::min(want, n));
            struct DistIdx {
                double d2;
                int idx;
            };
            std::vector<DistIdx> d;
            d.reserve(static_cast<size_t>(n));
            const double ax = poses[static_cast<size_t>(anchor)].x;
            const double ay = poses[static_cast<size_t>(anchor)].y;
            for (int j = 0; j < n; ++j) {
                double dx = poses[static_cast<size_t>(j)].x - ax;
                double dy = poses[static_cast<size_t>(j)].y - ay;
                d.push_back(DistIdx{dx * dx + dy * dy, j});
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

            if (p.rebuild_extreme_every > 0 && (t % p.rebuild_extreme_every) == 0) {
                boundary_pool = build_boundary_pool();
            }

            const double step =
                (p.step_frac_max * (1.0 - frac) + p.step_frac_min * frac) *
                std::max(1e-9, curr_side);
            const double ddeg_rng = p.ddeg_max * (1.0 - frac) + p.ddeg_min * frac;
            const double cx = 0.5 * (gmnx + gmxx);
            const double cy = 0.5 * (gmny + gmxy);

            int op = pick_op();
            bool accepted = false;
            bool improved_best = false;
            bool improved_curr = false;
            double reward_old_side = curr_side;
            double reward_new_side = curr_side;

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

                TreePose cand = poses[static_cast<size_t>(i)];
                cand.x += dx;
                cand.y += dy;
                if (uni(rng) < p.p_rot) {
                    cand.deg = wrap_deg(cand.deg + uni_deg(rng) * ddeg_rng);
                }

                if (cand.x < -100.0 || cand.x > 100.0 || cand.y < -100.0 ||
                    cand.y > 100.0) {
                    add_reward(op, false, false, false, curr_side, curr_side);
                    maybe_update_controller(t);
                    continue;
                }

                Polygon cand_poly = transform_polygon(base_poly_, cand);
                BoundingBox cand_bb = bounding_box(cand_poly);

                bool ok = true;
                for (int j = 0; j < n; ++j) {
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
                if (!ok) {
                    add_reward(op, false, false, false, curr_side, curr_side);
                    maybe_update_controller(t);
                    continue;
                }

                const BoundingBox old_bb = bbs[static_cast<size_t>(i)];
                const double old_gmnx = gmnx, old_gmxx = gmxx, old_gmny = gmny,
                             old_gmxy = gmxy;
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
                reward_old_side = old_side;
                reward_new_side = new_side;
                accepted = accept_move(old_side, new_side, T);
                if (accepted) {
                    improved_curr = (new_side + 1e-15 < old_side);
                    improved_best = (new_side + 1e-15 < best.best_side);
                    poses[static_cast<size_t>(i)] = cand;
                    polys[static_cast<size_t>(i)] = std::move(cand_poly);
                    curr_side = new_side;
                    if (improved_best) {
                        best.best_side = curr_side;
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
            } else if (op == kSwapRot) {
                int i = pick_index(p.p_pick_extreme).idx;
                int j = pick_other_index(i);

                TreePose cand_i = poses[static_cast<size_t>(i)];
                TreePose cand_j = poses[static_cast<size_t>(j)];
                std::swap(cand_i.deg, cand_j.deg);

                Polygon poly_i = transform_polygon(base_poly_, cand_i);
                Polygon poly_j = transform_polygon(base_poly_, cand_j);
                BoundingBox bb_i = bounding_box(poly_i);
                BoundingBox bb_j = bounding_box(poly_j);

                bool ok = true;
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
                    for (int k = 0; k < n; ++k) {
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
                        ddx = cand_j.x - poses[static_cast<size_t>(k)].x;
                        ddy = cand_j.y - poses[static_cast<size_t>(k)].y;
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

                if (!ok) {
                    add_reward(op, false, false, false, curr_side, curr_side);
                    maybe_update_controller(t);
                    continue;
                }

                const BoundingBox old_bb_i = bbs[static_cast<size_t>(i)];
                const BoundingBox old_bb_j = bbs[static_cast<size_t>(j)];
                const double old_gmnx = gmnx, old_gmxx = gmxx, old_gmny = gmny,
                             old_gmxy = gmxy;
                const double old_side = curr_side;

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

                double new_side = std::max(gmxx - gmnx, gmxy - gmny);
                reward_old_side = old_side;
                reward_new_side = new_side;
                accepted = accept_move(old_side, new_side, T);
                if (accepted) {
                    improved_curr = (new_side + 1e-15 < old_side);
                    improved_best = (new_side + 1e-15 < best.best_side);
                    poses[static_cast<size_t>(i)].deg = cand_i.deg;
                    poses[static_cast<size_t>(j)].deg = cand_j.deg;
                    polys[static_cast<size_t>(i)] = std::move(poly_i);
                    polys[static_cast<size_t>(j)] = std::move(poly_j);
                    curr_side = new_side;
                    if (improved_best) {
                        best.best_side = curr_side;
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
                }
            } else if (op == kRelocate) {
                int i = pick_index(p.p_pick_extreme).idx;

                TreePose cand;
                Polygon cand_poly;
                BoundingBox cand_bb;
                bool found = false;

                const double noise = p.relocate_noise_frac * std::max(1e-9, curr_side);
                for (int attempt = 0; attempt < std::max(1, p.relocate_attempts);
                     ++attempt) {
                    double pull = p.relocate_pull_min +
                                  (p.relocate_pull_max - p.relocate_pull_min) *
                                      uni(rng);

                    cand = poses[static_cast<size_t>(i)];
                    cand.x += pull * (cx - cand.x) + normal(rng) * noise;
                    cand.y += pull * (cy - cand.y) + normal(rng) * noise;
                    if (uni(rng) < p.relocate_p_rot) {
                        cand.deg = wrap_deg(cand.deg + uni_deg(rng) * ddeg_rng);
                    }

                    if (cand.x < -100.0 || cand.x > 100.0 || cand.y < -100.0 ||
                        cand.y > 100.0) {
                        continue;
                    }

                    cand_poly = transform_polygon(base_poly_, cand);
                    cand_bb = bounding_box(cand_poly);

                    bool ok = true;
                    for (int k = 0; k < n; ++k) {
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
                        if (polygons_intersect(cand_poly,
                                               polys[static_cast<size_t>(k)])) {
                            ok = false;
                            break;
                        }
                    }
                    if (!ok) {
                        continue;
                    }

                    found = true;
                    break;
                }

                if (!found) {
                    add_reward(op, false, false, false, curr_side, curr_side);
                    maybe_update_controller(t);
                    continue;
                }

                const BoundingBox old_bb = bbs[static_cast<size_t>(i)];
                const double old_gmnx = gmnx, old_gmxx = gmxx, old_gmny = gmny,
                             old_gmxy = gmxy;
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
                reward_old_side = old_side;
                reward_new_side = new_side;
                accepted = accept_move(old_side, new_side, T);
                if (accepted) {
                    improved_curr = (new_side + 1e-15 < old_side);
                    improved_best = (new_side + 1e-15 < best.best_side);
                    poses[static_cast<size_t>(i)] = cand;
                    polys[static_cast<size_t>(i)] = std::move(cand_poly);
                    curr_side = new_side;
                    if (improved_best) {
                        best.best_side = curr_side;
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
            } else if (op == kBlockTranslate) {
                if (n <= 1) {
                    add_reward(op, false, false, false, curr_side, curr_side);
                    maybe_update_controller(t);
                    continue;
                }
                int anchor = pick_index(1.0).idx;
                std::vector<int> block = build_block(anchor, p.block_size);
                std::vector<char> in_block(static_cast<size_t>(n), 0);
                for (int idx : block) {
                    in_block[static_cast<size_t>(idx)] = 1;
                }

                const double block_step =
                    (p.block_step_frac_max * (1.0 - frac) + p.block_step_frac_min * frac) *
                    std::max(1e-9, curr_side);
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

                bool ok = true;
                for (int idx : block) {
                    TreePose cand = poses[static_cast<size_t>(idx)];
                    cand.x += dx;
                    cand.y += dy;
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
                    add_reward(op, false, false, false, curr_side, curr_side);
                    maybe_update_controller(t);
                    continue;
                }

                for (size_t bi = 0; bi < block.size() && ok; ++bi) {
                    const auto& cand = moved_pose[bi];
                    const auto& cand_bb = moved_bb[bi];
                    const auto& cand_poly = moved_poly[bi];
                    for (int j = 0; j < n; ++j) {
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
                }
                if (!ok) {
                    add_reward(op, false, false, false, curr_side, curr_side);
                    maybe_update_controller(t);
                    continue;
                }

                Extents e2 = compute_extents_mixed(bbs, block, moved_bb);
                double new_side = side_from_extents(e2);
                reward_old_side = curr_side;
                reward_new_side = new_side;
                accepted = accept_move(curr_side, new_side, T);
                if (accepted) {
                    improved_curr = (new_side + 1e-15 < curr_side);
                    improved_best = (new_side + 1e-15 < best.best_side);
                    for (size_t bi = 0; bi < block.size(); ++bi) {
                        int idx = block[bi];
                        poses[static_cast<size_t>(idx)] = moved_pose[bi];
                        polys[static_cast<size_t>(idx)] = std::move(moved_poly[bi]);
                        bbs[static_cast<size_t>(idx)] = moved_bb[bi];
                    }
                    gmnx = e2.min_x;
                    gmxx = e2.max_x;
                    gmny = e2.min_y;
                    gmxy = e2.max_y;
                    curr_side = new_side;
                    if (improved_best) {
                        best.best_side = curr_side;
                        best.best_poses = poses;
                    }
                }
            } else if (op == kBlockRotate) {
                if (n <= 1) {
                    add_reward(op, false, false, false, curr_side, curr_side);
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

                const double rot_rng =
                    p.block_rot_deg_max * (1.0 - frac) + p.block_rot_deg_min * frac;
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
                    add_reward(op, false, false, false, curr_side, curr_side);
                    maybe_update_controller(t);
                    continue;
                }

                for (size_t bi = 0; bi < block.size() && ok; ++bi) {
                    const auto& cand = moved_pose[bi];
                    const auto& cand_bb = moved_bb[bi];
                    const auto& cand_poly = moved_poly[bi];
                    for (int j = 0; j < n; ++j) {
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
                }
                if (!ok) {
                    add_reward(op, false, false, false, curr_side, curr_side);
                    maybe_update_controller(t);
                    continue;
                }

                Extents e2 = compute_extents_mixed(bbs, block, moved_bb);
                double new_side = side_from_extents(e2);
                reward_old_side = curr_side;
                reward_new_side = new_side;
                accepted = accept_move(curr_side, new_side, T);
                if (accepted) {
                    improved_curr = (new_side + 1e-15 < curr_side);
                    improved_best = (new_side + 1e-15 < best.best_side);
                    for (size_t bi = 0; bi < block.size(); ++bi) {
                        int idx = block[bi];
                        poses[static_cast<size_t>(idx)] = moved_pose[bi];
                        polys[static_cast<size_t>(idx)] = std::move(moved_poly[bi]);
                        bbs[static_cast<size_t>(idx)] = moved_bb[bi];
                    }
                    gmnx = e2.min_x;
                    gmxx = e2.max_x;
                    gmny = e2.min_y;
                    gmxy = e2.max_y;
                    curr_side = new_side;
                    if (improved_best) {
                        best.best_side = curr_side;
                        best.best_poses = poses;
                    }
                }
            } else if (op == kLNS) {
                if (n <= 2 || p.lns_remove <= 0) {
                    add_reward(op, false, false, false, curr_side, curr_side);
                    maybe_update_controller(t);
                    continue;
                }

                const double boundary_tol = 1e-9;
                std::vector<int> boundary;
                boundary.reserve(static_cast<size_t>(n));
                for (int i = 0; i < n; ++i) {
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
                    add_reward(op, false, false, false, curr_side, curr_side);
                    maybe_update_controller(t);
                    continue;
                }

                int m = std::min(std::max(1, p.lns_remove), n - 1);
                if (static_cast<int>(boundary.size()) < m) {
                    m = static_cast<int>(boundary.size());
                }
                std::shuffle(boundary.begin(), boundary.end(), rng);
                boundary.resize(static_cast<size_t>(m));

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
                    add_reward(op, false, false, false, curr_side, curr_side);
                    maybe_update_controller(t);
                    continue;
                }

                double box_side =
                    std::max(max_x - min_x, max_y - min_y) * std::max(1e-6, p.lns_box_mult);
                const double ccx = 0.5 * (min_x + max_x);
                const double ccy = 0.5 * (min_y + max_y);
                const double half = 0.5 * box_side;
                const double noise = p.lns_noise_frac * std::max(1e-9, curr_side);

                std::vector<TreePose> cand_poses = poses;
                std::vector<Polygon> cand_polys = polys;
                std::vector<BoundingBox> cand_bbs = bbs;

                std::shuffle(boundary.begin(), boundary.end(), rng);
                bool ok = true;
                for (int idx : boundary) {
                    TreePose best_pose;
                    Polygon best_poly;
                    BoundingBox best_bb;
                    double best_side = std::numeric_limits<double>::infinity();
                    bool found = false;

                    for (int attempt = 0; attempt < std::max(1, p.lns_attempts_per_tree); ++attempt) {
                        TreePose cand = poses[static_cast<size_t>(idx)];
                        double mode = uni(rng);
                        if (mode < p.lns_p_uniform) {
                            cand.x = ccx + (2.0 * uni(rng) - 1.0) * half + normal(rng) * (0.35 * noise);
                            cand.y = ccy + (2.0 * uni(rng) - 1.0) * half + normal(rng) * (0.35 * noise);
                        } else if (mode < p.lns_p_uniform + p.lns_p_contact) {
                            int other = -1;
                            std::uniform_int_distribution<int> pick_any(0, n - 1);
                            for (int tries = 0; tries < 8; ++tries) {
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
                                double pull = p.lns_pull_min +
                                              (p.lns_pull_max - p.lns_pull_min) * uni(rng);
                                cand.x += pull * (ccx - cand.x) + normal(rng) * noise;
                                cand.y += pull * (ccy - cand.y) + normal(rng) * noise;
                            } else {
                                const double ang = 2.0 * 3.14159265358979323846 * uni(rng);
                                const double dist = thr * (0.88 + 0.14 * uni(rng));
                                cand.x = cand_poses[static_cast<size_t>(other)].x +
                                         dist * std::cos(ang) + normal(rng) * (0.20 * noise);
                                cand.y = cand_poses[static_cast<size_t>(other)].y +
                                         dist * std::sin(ang) + normal(rng) * (0.20 * noise);
                            }
                        } else {
                            double pull = p.lns_pull_min +
                                          (p.lns_pull_max - p.lns_pull_min) * uni(rng);
                            cand.x += pull * (ccx - cand.x) + normal(rng) * noise;
                            cand.y += pull * (ccy - cand.y) + normal(rng) * noise;
                        }
                        if (uni(rng) < p.lns_p_rot) {
                            cand.deg = wrap_deg(cand.deg + uni_deg(rng) * ddeg_rng);
                        }

                        if (cand.x < -100.0 || cand.x > 100.0 || cand.y < -100.0 || cand.y > 100.0) {
                            continue;
                        }

                        Polygon poly = transform_polygon(base_poly_, cand);
                        BoundingBox bb = bounding_box(poly);

                        bool collide = false;
                        for (int j = 0; j < n; ++j) {
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
                                collide = true;
                                break;
                            }
                        }
                        if (collide) {
                            continue;
                        }

                        double nmin_x = std::min(min_x, bb.min_x);
                        double nmax_x = std::max(max_x, bb.max_x);
                        double nmin_y = std::min(min_y, bb.min_y);
                        double nmax_y = std::max(max_y, bb.max_y);
                        double side = std::max(nmax_x - nmin_x, nmax_y - nmin_y);
                        if (side + 1e-15 < best_side) {
                            best_side = side;
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
                    min_x = std::min(min_x, best_bb.min_x);
                    max_x = std::max(max_x, best_bb.max_x);
                    min_y = std::min(min_y, best_bb.min_y);
                    max_y = std::max(max_y, best_bb.max_y);
                }
                if (!ok) {
                    add_reward(op, false, false, false, curr_side, curr_side);
                    maybe_update_controller(t);
                    continue;
                }

                double new_side = std::max(max_x - min_x, max_y - min_y);
                reward_old_side = curr_side;
                reward_new_side = new_side;
                accepted = accept_move(curr_side, new_side, T);
                if (accepted) {
                    improved_curr = (new_side + 1e-15 < curr_side);
                    improved_best = (new_side + 1e-15 < best.best_side);
                    poses = std::move(cand_poses);
                    polys = std::move(cand_polys);
                    bbs = std::move(cand_bbs);
                    gmnx = min_x;
                    gmxx = max_x;
                    gmny = min_y;
                    gmxy = max_y;
                    curr_side = new_side;
                    if (improved_best) {
                        best.best_side = curr_side;
                        best.best_poses = poses;
                    }
                }
            } else {
                add_reward(op, false, false, false, curr_side, curr_side);
                maybe_update_controller(t);
                continue;
            }

            add_reward(op, accepted, improved_best, improved_curr, reward_old_side, reward_new_side);
            maybe_update_controller(t);
        }

        return best;
    }

private:
    const Polygon& base_poly_;
    double radius_;

    static double wrap_deg(double deg) {
        deg = std::fmod(deg, 360.0);
        if (deg <= -180.0) {
            deg += 360.0;
        } else if (deg > 180.0) {
            deg -= 360.0;
        }
        return deg;
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
};
