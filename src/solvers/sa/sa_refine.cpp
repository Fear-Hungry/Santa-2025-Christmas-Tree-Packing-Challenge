#include "sa.hpp"

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

#include "collision.hpp"
#include "sa_geometry.hpp"
#include "sa_refine_cost.hpp"
#include "sa_refine_grid.hpp"
#include "sa_refine_pick.hpp"
#include "submission_io.hpp"
#include "wrap_utils.hpp"

namespace {

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


    class RefineContext {
    public:
        using Result = SARefiner::Result;
        using Params = SARefiner::Params;
        using OverlapMetric = SARefiner::OverlapMetric;
        using HHState = SARefiner::HHState;

        RefineContext(const SARefiner& refiner,
                      const Polygon& base_poly,
                      const std::vector<std::array<Point, 3>>& base_tris,
                      double radius,
                      const std::vector<TreePose>& start,
                      uint64_t seed,
                      const Params& params,
                      const std::vector<char>* active_mask,
                      HHState* hh_state)
            : refiner(refiner),
              base_poly_(base_poly),
              base_tris_(base_tris),
              radius_(radius),
              start(start),
              seed(seed),
              p(params),
              active_mask(active_mask),
              hh_state(hh_state) {}

        Result run() {
            const int n = static_cast<int>(start.size());
            if (n <= 0) {
                Result out;
                out.best_poses = start;
                out.final_poses = start;
                out.best_side = 0.0;
                out.final_side = 0.0;
                out.final_overlap = 0.0;
                out.final_cost = 0.0;
                return out;
            }
            if (active_mask && static_cast<int>(active_mask->size()) != n) {
                throw std::runtime_error("SARefiner: active_mask size inválido.");
            }
            if (p.iters <= 0) {
                auto polys = transformed_polygons(base_poly_, start);
                Result out;
                out.best_poses = start;
                out.final_poses = start;
                out.best_side = bounding_square_side(polys);
                out.final_side = out.best_side;
                out.final_overlap = 0.0;
                out.final_cost = out.best_side;
                return out;
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
                    pose = quantize_pose_wrap_deg(pose, qdec);
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

            const bool use_early_stop =
                p.early_stop && p.early_stop_check_interval > 0 &&
                p.early_stop_patience_iters > 0;
            auto early_improves = [&](double side,
                                      double area,
                                      double best_side,
                                      double best_area) -> bool {
                if (!std::isfinite(best_side)) {
                    return std::isfinite(side);
                }
                if (side < best_side - 1e-12) {
                    return true;
                }
                if (side <= best_side + p.plateau_eps && area < best_area - 1e-12) {
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

        sa_refine::CostModel cost_model(p);
        const bool soft_overlap = cost_model.soft_overlap;
        const bool use_mtv_metric = cost_model.use_mtv_metric;

        auto clamp_overlap = [&](double metric) -> double {
            return cost_model.clamp_overlap(metric);
        };
        auto cost_from = [&](double width, double height, double overlap_value) -> double {
            return cost_model.cost_from(width, height, overlap_value);
        };
        auto overlap_metric = [&](const TreePose& a, const TreePose& b) -> double {
            return cost_model.overlap_metric(
                a,
                b,
                [&](const TreePose& aa, const TreePose& bb, Point& mtv, double& area) {
                    return overlap_mtv(aa, bb, mtv, area);
                });
        };

        // Broad-phase incremental: hash grid uniforme com célula ~ 2r.
        // Como o raio é um círculo envolvente do polígono, qualquer interseção
        // implica dist(centros) <= 2r, então basta olhar 3x3 células.
        sa_refine::UniformGrid grid(n, thr);
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
                    const auto& pi = poses[static_cast<size_t>(i)];
                    grid.gather(pi.x, pi.y, neigh);
                    std::sort(neigh.begin(), neigh.end());
                    for (int j : neigh) {
                        if (j <= i) {
                            continue;
                        }
                        const auto& pj = poses[static_cast<size_t>(j)];
                        double dx = pi.x - pj.x;
                        double dy = pi.y - pj.y;
                        if (dx * dx + dy * dy > thr_sq) {
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
                        curr_overlap += overlap_metric(pi, pj);
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

            bool early_has_best =
                !use_early_stop || !soft_overlap || curr_overlap <= p.overlap_eps_area;
            double early_best_side = early_has_best ? curr_side
                                                    : std::numeric_limits<double>::infinity();
            double early_best_area = early_has_best ? (curr_width * curr_height)
                                                    : std::numeric_limits<double>::infinity();
            int early_last_improve_iter = 0;

            	        enum Op : int {
            	            kMicro = 0,
            	            kSwapRot = 1,
            	            kRelocate = 2,
            	            kBlockTranslate = 3,
            	            kBlockRotate = 4,
            	            kLNS = 5,
            	            kPushContact = 6,
            	            kSlideContact = 7,
            	            kSqueeze = 8,
            	            kGlobalRotate = 9,
            	            kEjectChain = 10,
            	            kResolveOverlap = 11,
            	            kNumOps = 12,
            	        };

            	        std::array<double, kNumOps> weights = {std::max(0.0, p.w_micro),
            	                                               std::max(0.0, p.w_swap_rot),
            	                                               std::max(0.0, p.w_relocate),
            	                                               std::max(0.0, p.w_block_translate),
            	                                               std::max(0.0, p.w_block_rotate),
            	                                               std::max(0.0, p.w_lns),
            	                                               std::max(0.0, p.w_push_contact),
            	                                               std::max(0.0, p.w_slide_contact),
            	                                               std::max(0.0, p.w_squeeze),
            	                                               std::max(0.0, p.w_global_rotate),
            	                                               std::max(0.0, p.w_eject_chain),
            	                                               std::max(0.0, p.w_resolve_overlap)};
            	        if (!soft_overlap) {
            	            weights[kResolveOverlap] = 0.0;
            	        } else {
            	            if (!(p.push_overshoot_frac > 0.0)) {
            	                weights[kPushContact] = 0.0;
            	            }
            	            weights[kSlideContact] = 0.0;
            	            weights[kSqueeze] = 0.0;
            	        }
            const bool hh_auto = p.hh_auto;
            auto lns_effective_remove = [&](int want_remove) -> int {
                if (n <= 2 || want_remove <= 0) {
                    return 0;
                }
                int m = std::min(std::max(1, want_remove), n - 1);
                if (hh_auto && n <= 8) {
                    m = std::min(m, std::max(1, n / 3));
                }
                return m;
            };
            const int lns_remove_effective = lns_effective_remove(p.lns_remove);

            	        std::array<double, kNumOps> op_cost = {
            	            1.0,
            	            2.0,
            	            static_cast<double>(std::max(1, p.relocate_attempts)),
                static_cast<double>(std::max(1, p.block_size)),
                static_cast<double>(std::max(1, p.block_size)),
                static_cast<double>(std::max(1, lns_remove_effective)) *
                    static_cast<double>(std::max(1, p.lns_attempts_per_tree)) *
                    static_cast<double>(std::max(1, p.lns_candidates)),
                static_cast<double>(std::max(1, p.push_bisect_iters)),
                static_cast<double>(std::max(1, p.slide_bisect_iters)),
            	            static_cast<double>(std::max(1, p.squeeze_pushes)) *
            	                static_cast<double>(std::max(1, p.push_bisect_iters)),
                static_cast<double>(std::max(1, n)),
                static_cast<double>(std::max(1, p.eject_relax_iters)) *
                        static_cast<double>(std::max(1, n)) +
                    static_cast<double>(std::max(1, p.eject_reinsert_attempts)),
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
            	            static_cast<char>(weights[kSlideContact] > 0.0),
            	            static_cast<char>(weights[kSqueeze] > 0.0 && p.squeeze_pushes > 0),
            	            static_cast<char>(weights[kGlobalRotate] > 0.0 && p.global_rot_deg > 0.0),
            	            static_cast<char>(weights[kEjectChain] > 0.0 && p.eject_reinsert_attempts > 0),
            	            static_cast<char>(weights[kResolveOverlap] > 0.0),
            	        };
            	        if (!(enabled[kMicro] || enabled[kSwapRot] || enabled[kRelocate] ||
            	              enabled[kBlockTranslate] || enabled[kBlockRotate] || enabled[kLNS] ||
            	              enabled[kPushContact] || enabled[kSlideContact] || enabled[kSqueeze] ||
            	              enabled[kGlobalRotate] || enabled[kEjectChain] || enabled[kResolveOverlap])) {
            	            enabled[kMicro] = 1;
            	            weights[kMicro] = 1.0;
            	        }

            HHState* state = hh_auto ? hh_state : nullptr;
            if (state && state->initialized) {
                for (int k = 0; k < kNumOps; ++k) {
                    if (enabled[static_cast<size_t>(k)]) {
                        weights[static_cast<size_t>(k)] =
                            std::max(0.0, state->weights[static_cast<size_t>(k)]);
                    } else {
                        weights[static_cast<size_t>(k)] = 0.0;
                    }
                }
            }

            	        std::array<double, kNumOps> op_score = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            	                                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            	        std::array<int, kNumOps> op_uses = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            if (state && state->initialized) {
                op_score = state->op_score;
                op_uses = state->op_uses;
                for (int k = 0; k < kNumOps; ++k) {
                    if (!enabled[static_cast<size_t>(k)]) {
                        op_score[static_cast<size_t>(k)] = 0.0;
                        op_uses[static_cast<size_t>(k)] = 0;
                    }
                }
            }

            double slide_schedule_frac = 0.0;
            auto effective_weight = [&](int op) -> double {
                double w = weights[static_cast<size_t>(op)];
                if (op == kSlideContact &&
                    enabled[static_cast<size_t>(kSlideContact)] &&
                    p.slide_schedule_max_frac > 0.0) {
                    const double f = std::min(0.999,
                                              std::max(0.0,
                                                       slide_schedule_frac * p.slide_schedule_max_frac));
                    if (f > 0.0) {
                        double sum_other = 0.0;
                        for (int k = 0; k < kNumOps; ++k) {
                            if (k == kSlideContact) {
                                continue;
                            }
                            if (enabled[static_cast<size_t>(k)]) {
                                sum_other += weights[static_cast<size_t>(k)];
                            }
                        }
                        if (sum_other > 0.0) {
                            const double w_sched =
                                (f / std::max(1e-12, 1.0 - f)) * sum_other;
                            w = std::max(w, w_sched);
                        }
                    }
                }
                return w;
            };

            auto sum_weights = [&]() -> double {
                double s = 0.0;
                for (int k = 0; k < kNumOps; ++k) {
                    if (enabled[static_cast<size_t>(k)]) {
                        s += effective_weight(k);
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
                    double w = effective_weight(k);
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
            if (p.lns_candidates < 1) {
                throw std::runtime_error("Parâmetros inválidos: lns_candidates precisa ser >= 1.");
            }
            if (p.lns_eval_attempts_per_tree < 0) {
                throw std::runtime_error("Parâmetros inválidos: lns_eval_attempts_per_tree precisa ser >= 0.");
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
            if (p.slide_dirs != 4 && p.slide_dirs != 8) {
                throw std::runtime_error("Parâmetros inválidos: slide_dirs precisa ser 4 ou 8.");
            }
            if (p.slide_dir_bias < 0.0) {
                throw std::runtime_error("Parâmetros inválidos: slide_dir_bias precisa ser >= 0.");
            }
            if (!(p.slide_max_step_frac > 0.0)) {
                throw std::runtime_error("Parâmetros inválidos: slide_max_step_frac precisa ser > 0.");
            }
            if (p.slide_bisect_iters <= 0) {
                throw std::runtime_error("Parâmetros inválidos: slide_bisect_iters precisa ser > 0.");
            }
            if (p.slide_min_gain < 0.0) {
                throw std::runtime_error("Parâmetros inválidos: slide_min_gain precisa ser >= 0.");
            }
            if (p.slide_schedule_max_frac < 0.0 || p.slide_schedule_max_frac >= 1.0) {
                throw std::runtime_error("Parâmetros inválidos: slide_schedule_max_frac precisa estar em [0,1).");
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
            if (p.time_budget_sec < 0.0) {
                throw std::runtime_error("Parâmetros inválidos: time_budget_sec precisa ser >= 0.");
            }
            if (p.global_rot_deg < 0.0) {
                throw std::runtime_error("Parâmetros inválidos: global_rot_deg precisa ser >= 0.");
            }
            if (p.eject_center_topk < 1) {
                throw std::runtime_error("Parâmetros inválidos: eject_center_topk precisa ser >= 1.");
            }
            if (p.eject_relax_iters < 0) {
                throw std::runtime_error("Parâmetros inválidos: eject_relax_iters precisa ser >= 0.");
            }
            if (!(p.eject_step_frac > 0.0)) {
                throw std::runtime_error("Parâmetros inválidos: eject_step_frac precisa ser > 0.");
            }
            if (p.eject_reinsert_attempts < 0) {
                throw std::runtime_error("Parâmetros inválidos: eject_reinsert_attempts precisa ser >= 0.");
            }
            if (p.eject_reinsert_noise_frac < 0.0) {
                throw std::runtime_error("Parâmetros inválidos: eject_reinsert_noise_frac precisa ser >= 0.");
            }
            if (p.eject_reinsert_rot_deg < 0.0) {
                throw std::runtime_error("Parâmetros inválidos: eject_reinsert_rot_deg precisa ser >= 0.");
            }
            if (p.eject_reinsert_p_rot < 0.0 || p.eject_reinsert_p_rot > 1.0) {
                throw std::runtime_error("Parâmetros inválidos: eject_reinsert_p_rot precisa estar em [0,1].");
            }

            sa_refine::IndexPicker picker(n,
                                          bbs,
                                          active_indices,
                                          active_pos,
                                          active_mask,
                                          use_mask,
                                          p.extreme_topk,
                                          rng,
                                          uni);
            picker.rebuild(gmnx, gmxx, gmny, gmxy);

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

            std::array<double, kNumOps> min_w{};
            if (hh_auto) {
                min_w.fill(p.hh_min_weight);
            } else {
                min_w[static_cast<size_t>(kMicro)] = p.hh_min_weight;
                min_w[static_cast<size_t>(kSwapRot)] = p.hh_min_weight;
                min_w[static_cast<size_t>(kRelocate)] = p.hh_min_weight;
            }
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
            	                                             std::numeric_limits<double>::infinity(),
            	                                             std::numeric_limits<double>::infinity(),
            	                                             max_lns_w,
            	                                             max_lns_w};

            auto normalize_weights = [&]() {
                std::array<double, kNumOps> base = weights;
                std::array<char, kNumOps> fixed{};
                for (int k = 0; k < kNumOps; ++k) {
                    if (!enabled[static_cast<size_t>(k)]) {
                        weights[static_cast<size_t>(k)] = 0.0;
                        base[static_cast<size_t>(k)] = 0.0;
                        fixed[static_cast<size_t>(k)] = 1;
                    }
                }
                for (int k = 0; k < kNumOps; ++k) {
                    if (fixed[static_cast<size_t>(k)]) {
                        continue;
                    }
                    base[static_cast<size_t>(k)] =
                        std::max(base[static_cast<size_t>(k)],
                                 min_w[static_cast<size_t>(k)]);
                }

                double remaining = 1.0;
                for (int iter = 0; iter < kNumOps; ++iter) {
                    double sum_base = 0.0;
                    int free_count = 0;
                    for (int k = 0; k < kNumOps; ++k) {
                        if (fixed[static_cast<size_t>(k)]) {
                            continue;
                        }
                        sum_base += base[static_cast<size_t>(k)];
                        free_count += 1;
                    }
                    if (free_count == 0) {
                        break;
                    }
                    if (!(sum_base > 0.0)) {
                        const double share = remaining / static_cast<double>(free_count);
                        for (int k = 0; k < kNumOps; ++k) {
                            if (fixed[static_cast<size_t>(k)]) {
                                continue;
                            }
                            weights[static_cast<size_t>(k)] = share;
                        }
                        break;
                    }

                    const double scale = remaining / sum_base;
                    bool any_fixed = false;
                    for (int k = 0; k < kNumOps; ++k) {
                        if (fixed[static_cast<size_t>(k)]) {
                            continue;
                        }
                        double w = base[static_cast<size_t>(k)] * scale;
                        const double lo = min_w[static_cast<size_t>(k)];
                        const double hi = max_w[static_cast<size_t>(k)];
                        if (w < lo) {
                            w = lo;
                            fixed[static_cast<size_t>(k)] = 1;
                            remaining -= w;
                            any_fixed = true;
                        } else if (w > hi) {
                            w = hi;
                            fixed[static_cast<size_t>(k)] = 1;
                            remaining -= w;
                            any_fixed = true;
                        }
                        weights[static_cast<size_t>(k)] = w;
                    }

                    if (remaining < 0.0) {
                        double sum = 0.0;
                        for (int k = 0; k < kNumOps; ++k) {
                            if (enabled[static_cast<size_t>(k)]) {
                                sum += weights[static_cast<size_t>(k)];
                            }
                        }
                        if (sum > 0.0) {
                            const double inv = 1.0 / sum;
                            for (int k = 0; k < kNumOps; ++k) {
                                if (enabled[static_cast<size_t>(k)]) {
                                    weights[static_cast<size_t>(k)] *= inv;
                                }
                            }
                        }
                        return;
                    }
                    if (!any_fixed) {
                        break;
                    }
                }
            };

            if (hh_auto) {
                normalize_weights();
            }

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
                    if (!hh_auto) {
                        w = std::max(min_w[static_cast<size_t>(k)],
                                     std::min(max_w[static_cast<size_t>(k)], w));
                    }
                    weights[static_cast<size_t>(k)] = w;
                    op_score[static_cast<size_t>(k)] = 0.0;
                    op_uses[static_cast<size_t>(k)] = 0;
                }
                if (hh_auto) {
                    normalize_weights();
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
                    picker.rebuild(gmnx, gmxx, gmny, gmxy);
                    const auto& boundary_pool = picker.boundary_pool();
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

            auto active_centroid = [&]() -> Point {
                double sx = 0.0;
                double sy = 0.0;
                int cnt = 0;
                if (use_mask) {
                    for (int idx : active_indices) {
                        sx += poses[static_cast<size_t>(idx)].x;
                        sy += poses[static_cast<size_t>(idx)].y;
                        ++cnt;
                    }
                } else {
                    for (const auto& pose : poses) {
                        sx += pose.x;
                        sy += pose.y;
                        ++cnt;
                    }
                }
                if (cnt <= 0) {
                    return Point{0.0, 0.0};
                }
                return Point{sx / static_cast<double>(cnt), sy / static_cast<double>(cnt)};
            };

            constexpr double kInvSqrt2 = 0.7071067811865475244;
            constexpr double kPi = 3.14159265358979323846;
            const std::array<Point, 8> slide_dirs = {
                Point{1.0, 0.0},
                Point{-1.0, 0.0},
                Point{0.0, 1.0},
                Point{0.0, -1.0},
                Point{kInvSqrt2, kInvSqrt2},
                Point{-kInvSqrt2, kInvSqrt2},
                Point{kInvSqrt2, -kInvSqrt2},
                Point{-kInvSqrt2, -kInvSqrt2},
            };

            struct SlideGapCacheEntry {
                double gap = std::numeric_limits<double>::infinity();
                uint64_t key_i = 0;
                uint64_t key_j = 0;
                bool valid = false;
            };

            auto dbl_bits = [](double v) -> uint64_t {
                uint64_t out = 0;
                std::memcpy(&out, &v, sizeof(out));
                return out;
            };
            auto mix_u64 = [](uint64_t v) -> uint64_t {
                v += 0x9e3779b97f4a7c15ULL;
                v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ULL;
                v = (v ^ (v >> 27)) * 0x94d049bb133111ebULL;
                return v ^ (v >> 31);
            };
            auto pose_key = [&](const TreePose& pose) -> uint64_t {
                uint64_t h = mix_u64(dbl_bits(pose.x));
                h ^= mix_u64(dbl_bits(pose.y) + 0x85ebca77c2b2ae63ULL);
                h ^= mix_u64(dbl_bits(pose.deg) + 0xc2b2ae3d27d4eb4fULL);
                return h;
            };

            const size_t slide_cache_stride = static_cast<size_t>(n) * 8;
            std::vector<SlideGapCacheEntry> slide_gap_cache(slide_cache_stride * static_cast<size_t>(n));

            auto cache_index = [&](int a, int b, int dir_idx) -> size_t {
                return (static_cast<size_t>(a) * static_cast<size_t>(n) +
                        static_cast<size_t>(b)) *
                           8 +
                       static_cast<size_t>(dir_idx);
            };

            auto pick_slide_direction =
                [&](int i, double& out_dx, double& out_dy, int& out_idx) -> bool {
                const int dir_count = (p.slide_dirs <= 4) ? 4 : 8;
                if (dir_count <= 0) {
                    return false;
                }

                Point target = active_centroid();
                double vx = target.x - poses[static_cast<size_t>(i)].x;
                double vy = target.y - poses[static_cast<size_t>(i)].y;
                const double vnorm = std::hypot(vx, vy);
                if (!(vnorm > 1e-12) || !(p.slide_dir_bias > 0.0)) {
                    std::uniform_int_distribution<int> pick(0, dir_count - 1);
                    const int idx = pick(rng);
                    const Point dir = slide_dirs[static_cast<size_t>(idx)];
                    out_dx = dir.x;
                    out_dy = dir.y;
                    out_idx = idx;
                    return true;
                }
                vx /= vnorm;
                vy /= vnorm;

                double max_dot = -2.0;
                std::array<double, 8> dots{};
                for (int k = 0; k < dir_count; ++k) {
                    double dot = slide_dirs[static_cast<size_t>(k)].x * vx +
                                 slide_dirs[static_cast<size_t>(k)].y * vy;
                    dot = std::max(-1.0, std::min(1.0, dot));
                    dots[static_cast<size_t>(k)] = dot;
                    if (dot > max_dot) {
                        max_dot = dot;
                    }
                }

                double sum = 0.0;
                std::array<double, 8> weights{};
                for (int k = 0; k < dir_count; ++k) {
                    double w = std::exp(p.slide_dir_bias * (dots[static_cast<size_t>(k)] - max_dot));
                    weights[static_cast<size_t>(k)] = w;
                    sum += w;
                }
                if (!(sum > 0.0)) {
                    std::uniform_int_distribution<int> pick(0, dir_count - 1);
                    const Point dir = slide_dirs[static_cast<size_t>(pick(rng))];
                    out_dx = dir.x;
                    out_dy = dir.y;
                    return true;
                }

                double r = uni(rng) * sum;
                for (int k = 0; k < dir_count; ++k) {
                    r -= weights[static_cast<size_t>(k)];
                    if (r <= 0.0) {
                        const Point dir = slide_dirs[static_cast<size_t>(k)];
                        out_dx = dir.x;
                        out_dy = dir.y;
                        out_idx = k;
                        return true;
                    }
                }
                const Point dir = slide_dirs[0];
                out_dx = dir.x;
                out_dy = dir.y;
                out_idx = 0;
                return true;
            };

            auto build_tris_for_pose = [&](const TreePose& pose,
                                           std::vector<std::array<Point, 3>>& out) {
                const double rad = pose.deg * kPi / 180.0;
                const double cA = std::cos(rad);
                const double sA = std::sin(rad);
                out.resize(base_tris_.size());
                for (size_t t = 0; t < base_tris_.size(); ++t) {
                    const auto& tri = base_tris_[t];
                    for (int k = 0; k < 3; ++k) {
                        const Point& p = tri[static_cast<size_t>(k)];
                        out[t][static_cast<size_t>(k)] = Point{
                            pose.x + cA * p.x - sA * p.y,
                            pose.y + sA * p.x + cA * p.y,
                        };
                    }
                }
            };

            auto tri_toi = [&](const std::array<Point, 3>& a,
                               const std::array<Point, 3>& b,
                               double vx,
                               double vy,
                               double max_t,
                               double& out_t) -> bool {
                double t_enter = 0.0;
                double t_exit = max_t;

                auto axis_update = [&](double ax, double ay) -> bool {
                    const double axis_len2 = ax * ax + ay * ay;
                    if (!(axis_len2 > 1e-18)) {
                        return true;
                    }
                    double min_a = ax * a[0].x + ay * a[0].y;
                    double max_a = min_a;
                    double min_b = ax * b[0].x + ay * b[0].y;
                    double max_b = min_b;
                    for (int k = 1; k < 3; ++k) {
                        double pa = ax * a[static_cast<size_t>(k)].x +
                                    ay * a[static_cast<size_t>(k)].y;
                        min_a = std::min(min_a, pa);
                        max_a = std::max(max_a, pa);
                        double pb = ax * b[static_cast<size_t>(k)].x +
                                    ay * b[static_cast<size_t>(k)].y;
                        min_b = std::min(min_b, pb);
                        max_b = std::max(max_b, pb);
                    }

                    const double v_proj = ax * vx + ay * vy;
                    if (std::abs(v_proj) < 1e-12) {
                        if (max_a < min_b || max_b < min_a) {
                            return false;
                        }
                        return true;
                    }

                    double t0 = (min_b - max_a) / v_proj;
                    double t1 = (max_b - min_a) / v_proj;
                    if (t0 > t1) {
                        std::swap(t0, t1);
                    }
                    t_enter = std::max(t_enter, t0);
                    t_exit = std::min(t_exit, t1);
                    return (t_enter <= t_exit);
                };

                for (int k = 0; k < 3; ++k) {
                    const Point& p0 = a[static_cast<size_t>(k)];
                    const Point& p1 = a[static_cast<size_t>((k + 1) % 3)];
                    const double ex = p1.x - p0.x;
                    const double ey = p1.y - p0.y;
                    if (!axis_update(-ey, ex)) {
                        return false;
                    }
                }
                for (int k = 0; k < 3; ++k) {
                    const Point& p0 = b[static_cast<size_t>(k)];
                    const Point& p1 = b[static_cast<size_t>((k + 1) % 3)];
                    const double ex = p1.x - p0.x;
                    const double ey = p1.y - p0.y;
                    if (!axis_update(-ey, ex)) {
                        return false;
                    }
                }

                if (t_exit < 0.0 || t_enter > max_t) {
                    return false;
                }
                out_t = std::max(0.0, t_enter);
                return true;
            };

            auto poly_toi = [&](const std::vector<std::array<Point, 3>>& a,
                                const std::vector<std::array<Point, 3>>& b,
                                double vx,
                                double vy,
                                double max_t,
                                double& out_t) -> bool {
                double best = max_t;
                for (const auto& ta : a) {
                    for (const auto& tb : b) {
                        double t_hit = 0.0;
                        if (!tri_toi(ta, tb, vx, vy, best, t_hit)) {
                            continue;
                        }
                        if (t_hit < best) {
                            best = t_hit;
                            if (best <= 0.0) {
                                out_t = 0.0;
                                return true;
                            }
                        }
                    }
                }
                if (best < max_t) {
                    out_t = best;
                    return true;
                }
                return false;
            };

            auto slide_contact_search = [&](int i,
                                            int dir_idx,
                                            double dir_x,
                                            double dir_y,
                                            TreePose& best_pose,
                                            Polygon& best_poly,
                                            BoundingBox& best_bb,
                                            double& best_delta) -> bool {
                const double dir_norm = std::hypot(dir_x, dir_y);
                if (!(dir_norm > 1e-12)) {
                    return false;
                }
                dir_x /= dir_norm;
                dir_y /= dir_norm;

                double max_step = p.slide_max_step_frac * std::max(1e-9, curr_side);
                if (std::abs(dir_x) > 1e-12) {
                    const double limit =
                        (dir_x > 0.0) ? (100.0 - poses[static_cast<size_t>(i)].x)
                                     : (poses[static_cast<size_t>(i)].x + 100.0);
                    max_step = std::min(max_step, limit / std::abs(dir_x));
                }
                if (std::abs(dir_y) > 1e-12) {
                    const double limit =
                        (dir_y > 0.0) ? (100.0 - poses[static_cast<size_t>(i)].y)
                                     : (poses[static_cast<size_t>(i)].y + 100.0);
                    max_step = std::min(max_step, limit / std::abs(dir_y));
                }

                double toi_max = max_step;
                const double toi_r = 2.0 * radius_;
                const double toi_r2 = toi_r * toi_r;
                const double toi_gap_limit = 400.0;
                const uint64_t key_i = pose_key(poses[static_cast<size_t>(i)]);
                std::vector<std::array<Point, 3>> tris_i;
                std::vector<std::array<Point, 3>> tris_j;
                build_tris_for_pose(poses[static_cast<size_t>(i)], tris_i);

                for (int j = 0; j < n; ++j) {
                    if (j == i || !is_active(j)) {
                        continue;
                    }
                    const uint64_t key_j = pose_key(poses[static_cast<size_t>(j)]);
                    SlideGapCacheEntry& entry =
                        slide_gap_cache[cache_index(i, j, dir_idx)];
                    if (entry.valid && entry.key_i == key_i && entry.key_j == key_j) {
                        if (entry.gap < toi_max) {
                            toi_max = entry.gap;
                        }
                        if (toi_max <= p.slide_min_gain) {
                            break;
                        }
                        continue;
                    }

                    double gap = std::numeric_limits<double>::infinity();
                    double rx = poses[static_cast<size_t>(i)].x - poses[static_cast<size_t>(j)].x;
                    double ry = poses[static_cast<size_t>(i)].y - poses[static_cast<size_t>(j)].y;
                    double r2 = rx * rx + ry * ry;
                    double rdot = rx * dir_x + ry * dir_y;
                    if (rdot >= 0.0) {
                        entry = SlideGapCacheEntry{gap, key_i, key_j, true};
                        continue;
                    }
                    double disc = rdot * rdot - (r2 - toi_r2);
                    if (disc <= 0.0) {
                        entry = SlideGapCacheEntry{gap, key_i, key_j, true};
                        continue;
                    }
                    double t_circle = -rdot - std::sqrt(disc);
                    if (t_circle < 0.0) {
                        t_circle = 0.0;
                    }
                    if (t_circle <= toi_gap_limit) {
                        build_tris_for_pose(poses[static_cast<size_t>(j)], tris_j);
                        double t_poly = 0.0;
                        if (poly_toi(tris_i, tris_j, dir_x, dir_y, toi_gap_limit, t_poly)) {
                            gap = t_poly;
                        }
                    }
                    entry = SlideGapCacheEntry{gap, key_i, key_j, true};
                    if (gap < toi_max) {
                        toi_max = gap;
                        if (toi_max <= p.slide_min_gain) {
                            break;
                        }
                    }
                }

                max_step = std::min(max_step, toi_max);
                if (!(max_step > p.slide_min_gain)) {
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

                best_delta = 0.0;
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

                    for (int it = 0; it < p.slide_bisect_iters; ++it) {
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

                return (best_delta > p.slide_min_gain);
            };

                const auto time_start = std::chrono::steady_clock::now();
            	        for (int t = 0; t < p.iters; ++t) {
                    if (p.time_budget_sec > 0.0) {
                        const auto now = std::chrono::steady_clock::now();
                        const double elapsed =
                            std::chrono::duration_cast<std::chrono::duration<double>>(now -
                                                                                     time_start)
                                .count();
                        if (elapsed >= p.time_budget_sec) {
                            break;
                        }
                    }
            	            double frac = static_cast<double>(t) / std::max(1, p.iters - 1);
            	            double T = p.t0 * (1.0 - frac) + p.t1 * frac;
                    slide_schedule_frac = (p.slide_schedule_max_frac > 0.0) ? frac : 0.0;

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
                cost_model.update_weight(t);
                const double width = gmxx - gmnx;
                const double height = gmxy - gmny;
                curr_cost = cost_from(width, height, curr_overlap);
            }

                if (p.rebuild_extreme_every > 0 && (t % p.rebuild_extreme_every) == 0) {
                    picker.rebuild(gmnx, gmxx, gmny, gmxy);
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

            	            switch (op) {
            	            case kMicro: {
            	                #include "sa_refine_ops/micro.inl"
            	                break;
            	            }
            	            case kPushContact: {
            	                #include "sa_refine_ops/push_contact.inl"
            	                break;
            	            }
            	            case kSlideContact: {
            	                #include "sa_refine_ops/slide_contact.inl"
            	                break;
            	            }
            	            case kSqueeze: {
            	                #include "sa_refine_ops/squeeze.inl"
            	                break;
            	            }
            	            case kSwapRot: {
            	                #include "sa_refine_ops/swap_rot.inl"
            	                break;
            	            }
            	            case kRelocate: {
            	                #include "sa_refine_ops/relocate.inl"
            	                break;
            	            }
            	            case kBlockTranslate: {
            	                #include "sa_refine_ops/block_translate.inl"
            	                break;
            	            }
            	            case kBlockRotate: {
            	                #include "sa_refine_ops/block_rotate.inl"
            	                break;
            	            }
            	            case kLNS: {
            	                #include "sa_refine_ops/lns.inl"
            	                break;
            	            }
            	            case kGlobalRotate: {
            	                #include "sa_refine_ops/global_rotate.inl"
            	                break;
            	            }
            	            case kEjectChain: {
            	                #include "sa_refine_ops/eject_chain.inl"
            	                break;
            	            }
            	            case kResolveOverlap: {
            	                #include "sa_refine_ops/resolve_overlap.inl"
            	                break;
            	            }
            	            default: {
            	                #include "sa_refine_ops/default.inl"
            	                break;
            	            }
            	            }


                add_reward(op, accepted, improved_best, improved_curr, reward_old_cost, reward_new_cost);
                if (improved_best) {
                    last_best_iter = t;
                }
                maybe_update_controller(t);

                if (use_early_stop &&
                    (p.early_stop_check_interval > 0) &&
                    ((t + 1) % p.early_stop_check_interval == 0)) {
                    if (!soft_overlap || curr_overlap <= p.overlap_eps_area) {
                        const double width = gmxx - gmnx;
                        const double height = gmxy - gmny;
                        const double side = std::max(width, height);
                        const double area = width * height;
                        if (!early_has_best ||
                            early_improves(side, area, early_best_side, early_best_area)) {
                            early_best_side = side;
                            early_best_area = area;
                            early_last_improve_iter = t + 1;
                            early_has_best = true;
                        }
                    }
                    if (early_has_best &&
                        (t + 1) >= p.early_stop_min_iters &&
                        (t + 1 - early_last_improve_iter) >= p.early_stop_patience_iters) {
                        break;
                    }
                }
            }

            if (state) {
                state->weights = weights;
                state->op_score = op_score;
                state->op_uses = op_uses;
                state->initialized = true;
            }

            best.final_poses = poses;
            best.final_side = curr_side;
            best.final_overlap = curr_overlap;
            best.final_cost = curr_cost;
            return best;
        }

    private:
        bool overlap_mtv(const TreePose& a,
                         const TreePose& b,
                         Point& out_mtv,
                         double& out_overlap_area) const {
            return refiner.overlap_mtv(a, b, out_mtv, out_overlap_area);
        }

        const SARefiner& refiner;
        const Polygon& base_poly_;
        const std::vector<std::array<Point, 3>>& base_tris_;
        const double radius_;
        const std::vector<TreePose>& start;
        const uint64_t seed;
        const Params& p;
        const std::vector<char>* active_mask;
        HHState* hh_state;
    };

}  // namespace

SARefiner::Result SARefiner::refine_min_side(const std::vector<TreePose>& start,
                                         uint64_t seed,
                                         const Params& p,
                                         const std::vector<char>* active_mask,
                                         HHState* hh_state) const {
    RefineContext ctx(*this, base_poly_, base_tris_, radius_, start, seed, p, active_mask,
                      hh_state);
    return ctx.run();
}
