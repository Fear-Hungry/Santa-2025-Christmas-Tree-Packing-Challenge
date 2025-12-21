#include "boundary_refine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "collision.hpp"
#include "submission_io.hpp"
#include "spatial_grid.hpp"
#include "wrap_utils.hpp"

namespace {
constexpr double kTol = 1e-9;

struct Extents {
    double min_x = 0.0;
    double max_x = 0.0;
    double min_y = 0.0;
    double max_y = 0.0;
};

struct BoundaryChoice {
    int axis = 0;  // 0 = x, 1 = y
    int side = 0;  // 0 = min, 1 = max
};

struct MoveDelta {
    double dx = 0.0;
    double dy = 0.0;
    double ddeg = 0.0;
};

struct MoveProposal {
    int idx = 0;
    MoveDelta delta;
    bool deterministic = false;
    double temperature = 0.0;
};

struct MovePlan {
    bool shrink_x = false;
    double step = 0.0;
    bool deterministic = false;
};

enum class MoveOutcome {
    kApplied,
    kRetry,
    kStop,
};

struct CandidatePlacement {
    int idx = 0;
    TreePose pose;
    Polygon poly;
    BoundingBox bb;
};

struct RandomSource {
    std::mt19937_64& rng;
    std::uniform_real_distribution<double>& uni;
    std::normal_distribution<double>& normal;
};

struct ScaledCandidate {
    const MoveProposal& proposal;
    const TreePose& base;
    double scale = 1.0;
};

struct NeighborDelta {
    int idx = 0;
    double dx = 0.0;
    double dy = 0.0;
};

bool aabb_overlap(const BoundingBox& a, const BoundingBox& b) {
    if (a.max_x < b.min_x || b.max_x < a.min_x) {
        return false;
    }
    if (a.max_y < b.min_y || b.max_y < a.min_y) {
        return false;
    }
    return true;
}

std::vector<BoundingBox> build_bbs(const std::vector<Polygon>& polys) {
    std::vector<BoundingBox> out;
    out.reserve(polys.size());
    for (const auto& poly : polys) {
        out.push_back(bounding_box(poly));
    }
    return out;
}

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

double bb_axis_value(const BoundingBox& bb, int axis, int side) {
    const double values[2][2] = {
        {bb.min_x, bb.max_x},
        {bb.min_y, bb.max_y},
    };
    return values[axis][side];
}

double ext_axis_limit(const Extents& ext, int axis, int side) {
    const double limits[2][2] = {
        {ext.min_x + kTol, ext.max_x - kTol},
        {ext.min_y + kTol, ext.max_y - kTol},
    };
    return limits[axis][side];
}

bool on_boundary(const BoundingBox& bb, const Extents& ext, const BoundaryChoice& choice) {
    const double value = bb_axis_value(bb, choice.axis, choice.side);
    const double limit = ext_axis_limit(ext, choice.axis, choice.side);
    const double sign = (choice.side == 0) ? 1.0 : -1.0;
    return sign * (limit - value) >= 0.0;
}

BoundaryChoice pick_boundary_choice(bool shrink_x,
                                    RandomSource& rnd) {
    BoundaryChoice choice;
    choice.axis = shrink_x ? 0 : 1;
    choice.side = (rnd.uni(rnd.rng) < 0.5) ? 0 : 1;
    return choice;
}

std::vector<int> collect_boundary_candidates(const std::vector<BoundingBox>& bbs,
                                             const Extents& ext,
                                             const BoundaryChoice& choice) {
    std::vector<int> boundary;
    boundary.reserve(bbs.size());
    for (size_t i = 0; i < bbs.size(); ++i) {
        if (on_boundary(bbs[i], ext, choice)) {
            boundary.push_back(static_cast<int>(i));
        }
    }
    return boundary;
}

int pick_index_from_pool(const std::vector<int>& pool,
                         size_t total,
                         RandomSource& rnd) {
    if (!pool.empty()) {
        std::uniform_int_distribution<int> pick(0, static_cast<int>(pool.size()) - 1);
        return pool[static_cast<size_t>(pick(rnd.rng))];
    }
    std::uniform_int_distribution<int> pick(0, static_cast<int>(total) - 1);
    return pick(rnd.rng);
}

struct CandidateExtents {
    Extents extents;
    double side = 0.0;
};

CandidateExtents candidate_extents(std::vector<BoundingBox>& bbs,
                                   int idx,
                                   const BoundingBox& cand_bb) {
    const BoundingBox old_bb = bbs[static_cast<size_t>(idx)];
    bbs[static_cast<size_t>(idx)] = cand_bb;
    CandidateExtents out;
    out.extents = compute_extents(bbs);
    out.side = side_from_extents(out.extents);
    bbs[static_cast<size_t>(idx)] = old_bb;
    return out;
}

int pick_boundary_index(const std::vector<BoundingBox>& bbs,
                        bool shrink_x,
                        const Extents& ext,
                        RandomSource& rnd) {
    const BoundaryChoice choice = pick_boundary_choice(shrink_x, rnd);
    std::vector<int> boundary = collect_boundary_candidates(bbs, ext, choice);
    return pick_index_from_pool(boundary, bbs.size(), rnd);
}

double inward_dir_component(const BoundingBox& bb,
                            const Extents& ext,
                            int axis,
                            RandomSource& rnd) {
    const BoundaryChoice min_side{axis, 0};
    const BoundaryChoice max_side{axis, 1};
    if (on_boundary(bb, ext, min_side)) {
        return 1.0;
    }
    if (on_boundary(bb, ext, max_side)) {
        return -1.0;
    }
    return (rnd.uni(rnd.rng) < 0.5) ? 1.0 : -1.0;
}

Point pick_direction(const BoundingBox& bb,
                     bool shrink_x,
                     const Extents& ext,
                     RandomSource& rnd) {
    const int axis = shrink_x ? 0 : 1;
    const double dir = inward_dir_component(bb, ext, axis, rnd);
    return Point{dir * static_cast<double>(axis == 0),
                 dir * static_cast<double>(axis == 1)};
}

struct BoundaryRefineState {
    const Polygon& base_poly;
    std::vector<TreePose>& poses;
    std::vector<Polygon> polys;
    std::vector<BoundingBox> bbs;
    UniformGridIndex grid;
    std::vector<int> neigh;
    double limit_sq = 0.0;
    Extents extents;
    double current_s = 0.0;

    BoundaryRefineState(const Polygon& base_poly_in,
                        double radius,
                        std::vector<TreePose>& poses_in)
        : base_poly(base_poly_in),
          poses(poses_in),
          polys(transformed_polygons(base_poly_in, poses_in)),
          bbs(build_bbs(polys)),
          grid(static_cast<int>(poses_in.size()), 2.0 * radius + kTol) {
        grid.rebuild(poses);
        neigh.reserve(64);
        const double limit = 2.0 * radius + kTol;
        limit_sq = limit * limit;
        extents = compute_extents(bbs);
        current_s = side_from_extents(extents);
    }
};

bool should_check_collision(const BoundaryRefineState& st,
                            const CandidatePlacement& cand,
                            const NeighborDelta& delta) {
    if (delta.idx == cand.idx) {
        return false;
    }
    if (delta.dx * delta.dx + delta.dy * delta.dy > st.limit_sq) {
        return false;
    }
    if (!aabb_overlap(cand.bb, st.bbs[static_cast<size_t>(delta.idx)])) {
        return false;
    }
    return true;
}

bool collision_free(BoundaryRefineState& st, const CandidatePlacement& cand) {
    st.grid.gather(cand.pose.x, cand.pose.y, st.neigh);
    for (int j : st.neigh) {
        NeighborDelta delta;
        delta.idx = j;
        delta.dx = cand.pose.x - st.poses[static_cast<size_t>(j)].x;
        delta.dy = cand.pose.y - st.poses[static_cast<size_t>(j)].y;
        if (should_check_collision(st, cand, delta) &&
            polygons_intersect(cand.poly, st.polys[static_cast<size_t>(j)])) {
            return false;
        }
    }
    return true;
}

MoveProposal build_move_proposal(const BoundaryRefineState& st,
                                 const MovePlan& plan,
                                 RandomSource& rnd) {
    MoveProposal proposal;
    proposal.idx = pick_boundary_index(st.bbs, plan.shrink_x, st.extents, rnd);
    proposal.deterministic = plan.deterministic;

    const Point dir = pick_direction(st.bbs[static_cast<size_t>(proposal.idx)],
                                     plan.shrink_x,
                                     st.extents,
                                     rnd);

    proposal.delta.dx = dir.x * plan.step;
    proposal.delta.dy = dir.y * plan.step;
    proposal.delta.ddeg = 0.0;
    if (!plan.deterministic) {
        proposal.delta.dx += rnd.normal(rnd.rng) * plan.step * 0.15;
        proposal.delta.dy += rnd.normal(rnd.rng) * plan.step * 0.15;
        proposal.delta.ddeg = rnd.normal(rnd.rng) * 5.0;
    }
    return proposal;
}

TreePose build_scaled_candidate(const TreePose& base,
                                const MoveProposal& proposal,
                                double scale) {
    TreePose cand = base;
    cand.x += proposal.delta.dx * scale;
    cand.y += proposal.delta.dy * scale;
    cand.deg = wrap_deg(cand.deg + proposal.delta.ddeg * scale);
    return quantize_pose_wrap_deg(cand);
}

bool in_bounds(const TreePose& pose) {
    return pose.x >= -100.0 && pose.x <= 100.0 && pose.y >= -100.0 && pose.y <= 100.0;
}

CandidatePlacement build_placement(int idx, const TreePose& pose, const Polygon& base_poly) {
    CandidatePlacement placement;
    placement.idx = idx;
    placement.pose = pose;
    placement.poly = transform_polygon(base_poly, pose);
    placement.bb = bounding_box(placement.poly);
    return placement;
}

bool accept_candidate(const BoundaryRefineState& st,
                      const MoveProposal& proposal,
                      const CandidateExtents& ext,
                      RandomSource& rnd) {
    if (ext.side + 1e-12 < st.current_s) {
        return true;
    }
    if (!proposal.deterministic && proposal.temperature > 0.0) {
        const double prob = std::exp((st.current_s - ext.side) / proposal.temperature);
        return rnd.uni(rnd.rng) < prob;
    }
    return false;
}

void apply_candidate(BoundaryRefineState& st,
                     const CandidatePlacement& placement,
                     const CandidateExtents& ext) {
    const int idx = placement.idx;
    st.poses[static_cast<size_t>(idx)] = placement.pose;
    st.polys[static_cast<size_t>(idx)] = placement.poly;
    st.bbs[static_cast<size_t>(idx)] = placement.bb;
    st.grid.update_position(idx, placement.pose.x, placement.pose.y);
    st.extents = ext.extents;
    st.current_s = ext.side;
}

MoveOutcome try_scaled_candidate(BoundaryRefineState& st,
                                 const ScaledCandidate& cand,
                                 RandomSource& rnd) {
    TreePose pose = build_scaled_candidate(cand.base, cand.proposal, cand.scale);
    if (!in_bounds(pose)) {
        return MoveOutcome::kRetry;
    }

    CandidatePlacement placement =
        build_placement(cand.proposal.idx, pose, st.base_poly);
    if (!collision_free(st, placement)) {
        return MoveOutcome::kRetry;
    }

    CandidateExtents ext = candidate_extents(st.bbs, cand.proposal.idx, placement.bb);
    if (!accept_candidate(st, cand.proposal, ext, rnd)) {
        return cand.proposal.deterministic ? MoveOutcome::kRetry : MoveOutcome::kStop;
    }

    apply_candidate(st, placement, ext);
    return MoveOutcome::kApplied;
}

bool try_refine_move(BoundaryRefineState& st,
                     const MoveProposal& proposal,
                     RandomSource& rnd) {
    const int idx = proposal.idx;
    const TreePose base = st.poses[static_cast<size_t>(idx)];
    double scale = 1.0;
    for (int bt = 0; bt < 6; ++bt) {
        ScaledCandidate cand{proposal, base, scale};
        MoveOutcome outcome = try_scaled_candidate(st, cand, rnd);
        if (outcome == MoveOutcome::kApplied) {
            return true;
        }
        if (outcome == MoveOutcome::kStop) {
            return false;
        }
        scale *= 0.5;
    }
    return false;
}
}  // namespace

void refine_boundary(const Polygon& base_poly,
                     std::vector<TreePose>& poses,
                     const BoundaryRefineParams& params) {
    if (params.iters <= 0 || poses.empty()) {
        return;
    }

    for (auto& p : poses) {
        p = quantize_pose_wrap_deg(p);
    }

    std::mt19937_64 rng(params.seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::normal_distribution<double> normal(0.0, 1.0);
    RandomSource rnd{rng, uni, normal};

    BoundaryRefineState st(base_poly, params.radius, poses);

    const double initial_step =
        std::max(params.step_hint * 0.35, params.radius * 0.20);
    const double final_step =
        std::max(params.step_hint * 0.03, params.radius * 0.02);
    const double initial_T = std::max(0.01 * st.current_s, 1e-6);
    const double final_T = initial_T * 1e-3;
    const int compact_iters = params.iters / 2;

    for (int it = 0; it < params.iters; ++it) {
        double t = static_cast<double>(it) / std::max(1, params.iters - 1);
        double step = initial_step * std::pow(final_step / initial_step, t);
        double T = initial_T * std::pow(final_T / initial_T, t);

        double width = st.extents.max_x - st.extents.min_x;
        double height = st.extents.max_y - st.extents.min_y;
        bool shrink_x = (width >= height);
        MovePlan plan;
        plan.shrink_x = shrink_x;
        plan.step = step;
        plan.deterministic = (it < compact_iters);
        MoveProposal proposal = build_move_proposal(st, plan, rnd);
        proposal.temperature = T;
        try_refine_move(st, proposal, rnd);
    }
}
