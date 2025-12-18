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
            p.hh_segment = opt.sa_hh_segment;
            p.hh_reaction = opt.sa_hh_reaction;
            p.overlap_weight = opt.sa_overlap_weight;
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
            p.squeeze_pushes = opt.sa_squeeze_pushes;

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
        p.hh_segment = opt.sa_hh_segment;
        p.hh_reaction = opt.sa_hh_reaction;
        p.overlap_weight = opt.sa_overlap_weight;
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
        p.squeeze_pushes = opt.sa_squeeze_pushes;

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
