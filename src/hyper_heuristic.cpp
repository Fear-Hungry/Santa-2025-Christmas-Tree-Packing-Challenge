#include "santa2025/hyper_heuristic.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "santa2025/logging.hpp"

namespace santa2025 {
namespace {

struct Candidate {
    std::vector<Pose> poses;
    double s = 0.0;
    double score = 0.0;
    double obj = 0.0;
};

int default_lahc_length(int n) {
    const int v = 10 * n;
    return std::clamp(v, 50, 500);
}

double objective_value(HHObjective obj, double s, double score) {
    switch (obj) {
    case HHObjective::kS:
        return s;
    case HHObjective::kPrefixScore:
        return score;
    }
    return s;
}

struct Operator {
    std::string name;
    // Apply returns a feasible candidate (operators are expected to keep feasibility).
    // Empty poses means "no candidate" (treated as infeasible).
    Candidate (*apply)(const Polygon&, const std::vector<Pose>&, const HHOptions&, std::mt19937_64&, double eps);
};

Candidate op_sa_intensify(
    const Polygon& poly,
    const std::vector<Pose>& cur,
    const HHOptions& opt,
    std::mt19937_64& rng,
    double eps
) {
    SAOptions sa = opt.sa_base;
    sa.n = opt.n;
    sa.objective = (opt.objective == HHObjective::kPrefixScore) ? SAObjective::kPrefixScore : SAObjective::kS200;
    sa.nmax_score = opt.nmax_score;

    sa.iters = std::max(50, sa.iters);
    sa.t0 = std::min(sa.t0, 0.10);
    sa.t1 = std::min(sa.t1, 1e-4);
    sa.trans_sigma0 = std::min(sa.trans_sigma0, 0.08);
    sa.trans_sigma1 = std::min(sa.trans_sigma1, 0.01);
    sa.boundary_prob = std::max(sa.boundary_prob, 0.6);
    sa.touch_prob = std::max(sa.touch_prob, 0.30);
    sa.cluster_prob = std::min(sa.cluster_prob, 0.15);

    sa.seed = rng();
    sa.log_every = 0;

    const auto res = simulated_annealing(poly, cur, sa, eps);
    Candidate out;
    out.poses = res.best_poses;
    out.s = packing_s200(poly, out.poses);
    out.score = packing_prefix_score(poly, out.poses, opt.nmax_score);
    out.obj = objective_value(opt.objective, out.s, out.score);
    return out;
}

Candidate op_sa_diversify(
    const Polygon& poly,
    const std::vector<Pose>& cur,
    const HHOptions& opt,
    std::mt19937_64& rng,
    double eps
) {
    SAOptions sa = opt.sa_base;
    sa.n = opt.n;
    sa.objective = (opt.objective == HHObjective::kPrefixScore) ? SAObjective::kPrefixScore : SAObjective::kS200;
    sa.nmax_score = opt.nmax_score;

    sa.iters = std::max(50, sa.iters);
    sa.t0 = std::max(sa.t0, 0.35);
    sa.t1 = std::max(sa.t1, 1e-4);
    sa.trans_sigma0 = std::max(sa.trans_sigma0, 0.25);
    sa.trans_sigma1 = std::max(sa.trans_sigma1, 0.02);
    sa.boundary_prob = std::max(sa.boundary_prob, 0.4);
    sa.touch_prob = std::max(sa.touch_prob, 0.20);
    sa.cluster_prob = std::max(sa.cluster_prob, 0.15);

    sa.seed = rng();
    sa.log_every = 0;

    const auto res = simulated_annealing(poly, cur, sa, eps);
    Candidate out;
    out.poses = res.best_poses;
    out.s = packing_s200(poly, out.poses);
    out.score = packing_prefix_score(poly, out.poses, opt.nmax_score);
    out.obj = objective_value(opt.objective, out.s, out.score);
    return out;
}

Candidate op_lns_boundary_small(
    const Polygon& poly,
    const std::vector<Pose>& cur,
    const HHOptions& opt,
    std::mt19937_64& rng,
    double eps
) {
    LNSOptions lns = opt.lns_base;
    lns.n = opt.n;
    lns.stages = std::max(1, lns.stages);
    lns.stage_attempts = std::max(1, lns.stage_attempts);
    lns.remove_frac = std::clamp(lns.remove_frac, 0.03, 0.10);
    lns.boundary_prob = 1.0;
    lns.destroy_mode = LNSDestroyMode::kBoundary;
    lns.seed = rng();
    lns.log_every = 0;

    const auto r = lns_shrink_wrap(poly, cur, lns, eps);
    Candidate out;
    out.poses = r.best_poses;
    out.s = packing_s200(poly, out.poses);
    out.score = packing_prefix_score(poly, out.poses, opt.nmax_score);
    out.obj = objective_value(opt.objective, out.s, out.score);
    return out;
}

Candidate op_lns_boundary_large(
    const Polygon& poly,
    const std::vector<Pose>& cur,
    const HHOptions& opt,
    std::mt19937_64& rng,
    double eps
) {
    LNSOptions lns = opt.lns_base;
    lns.n = opt.n;
    lns.stages = std::max(1, lns.stages);
    lns.stage_attempts = std::max(1, lns.stage_attempts);
    lns.remove_frac = std::clamp(lns.remove_frac, 0.10, 0.25);
    lns.boundary_prob = 1.0;
    lns.destroy_mode = LNSDestroyMode::kBoundary;
    lns.seed = rng();
    lns.log_every = 0;

    const auto r = lns_shrink_wrap(poly, cur, lns, eps);
    Candidate out;
    out.poses = r.best_poses;
    out.s = packing_s200(poly, out.poses);
    out.score = packing_prefix_score(poly, out.poses, opt.nmax_score);
    out.obj = objective_value(opt.objective, out.s, out.score);
    return out;
}

Candidate op_lns_random(
    const Polygon& poly,
    const std::vector<Pose>& cur,
    const HHOptions& opt,
    std::mt19937_64& rng,
    double eps
) {
    LNSOptions lns = opt.lns_base;
    lns.n = opt.n;
    lns.stages = std::max(1, lns.stages);
    lns.stage_attempts = std::max(1, lns.stage_attempts);
    lns.boundary_prob = 0.0;
    lns.destroy_mode = LNSDestroyMode::kRandom;
    lns.seed = rng();
    lns.log_every = 0;

    const auto r = lns_shrink_wrap(poly, cur, lns, eps);
    Candidate out;
    out.poses = r.best_poses;
    out.s = packing_s200(poly, out.poses);
    out.score = packing_prefix_score(poly, out.poses, opt.nmax_score);
    out.obj = objective_value(opt.objective, out.s, out.score);
    return out;
}

Candidate op_lns_cluster(
    const Polygon& poly,
    const std::vector<Pose>& cur,
    const HHOptions& opt,
    std::mt19937_64& rng,
    double eps
) {
    LNSOptions lns = opt.lns_base;
    lns.n = opt.n;
    lns.stages = std::max(1, lns.stages);
    lns.stage_attempts = std::max(1, lns.stage_attempts);
    lns.boundary_prob = std::clamp(lns.boundary_prob, 0.0, 1.0);
    lns.destroy_mode = LNSDestroyMode::kCluster;
    lns.seed = rng();
    lns.log_every = 0;

    const auto r = lns_shrink_wrap(poly, cur, lns, eps);
    Candidate out;
    out.poses = r.best_poses;
    out.s = packing_s200(poly, out.poses);
    out.score = packing_prefix_score(poly, out.poses, opt.nmax_score);
    out.obj = objective_value(opt.objective, out.s, out.score);
    return out;
}

int pick_operator_ucb(const std::vector<HHOperatorStats>& stats, double ucb_c, int t) {
    for (size_t i = 0; i < stats.size(); ++i) {
        if (stats[i].selected == 0) {
            return static_cast<int>(i);
        }
    }
    const double logt = std::log(static_cast<double>(t + 2));
    int best = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < stats.size(); ++i) {
        const double mean = stats[i].mean_reward;
        const double n = static_cast<double>(std::max(1, stats[i].selected));
        const double bonus = ucb_c * std::sqrt(logt / n);
        const double score = mean + bonus;
        if (score > best_score) {
            best_score = score;
            best = static_cast<int>(i);
        }
    }
    return best;
}

}  // namespace

HHResult hyper_heuristic_optimize(
    const Polygon& tree_poly,
    const std::vector<Pose>& initial,
    const HHOptions& opt,
    double eps
) {
    if (opt.n <= 0) {
        return HHResult{};
    }
    if (static_cast<int>(initial.size()) != opt.n) {
        throw std::invalid_argument("hyper_heuristic_optimize: initial.size() must equal opt.n");
    }
    if (opt.hh_iters <= 0) {
        throw std::invalid_argument("hyper_heuristic_optimize: hh_iters must be > 0");
    }
    if (!(opt.ucb_c >= 0.0)) {
        throw std::invalid_argument("hyper_heuristic_optimize: ucb_c must be >= 0");
    }

    const int L = (opt.lahc_length > 0) ? opt.lahc_length : default_lahc_length(opt.n);

    std::vector<Pose> cur = initial;
    const double init_s = packing_s200(tree_poly, cur);
    const double init_score = packing_prefix_score(tree_poly, cur, opt.nmax_score);
    double cur_s = init_s;
    double cur_score = init_score;
    double cur_obj = objective_value(opt.objective, cur_s, cur_score);

    std::vector<Pose> best = cur;
    double best_s = cur_s;
    double best_score = cur_score;
    double best_obj = cur_obj;

    std::vector<double> hist(static_cast<size_t>(L), cur_obj);

    std::mt19937_64 rng(opt.seed);

    std::vector<Operator> ops = {
        {"sa_intensify", &op_sa_intensify},
        {"sa_diversify", &op_sa_diversify},
        {"lns_boundary_small", &op_lns_boundary_small},
        {"lns_boundary_large", &op_lns_boundary_large},
        {"lns_cluster", &op_lns_cluster},
        {"lns_random", &op_lns_random},
    };

    HHResult out;
    out.best_poses = best;
    out.init_s = init_s;
    out.best_s = best_s;
    out.init_score = init_score;
    out.best_score = best_score;
    out.init_obj = cur_obj;
    out.best_obj = best_obj;
    out.ops.resize(ops.size());
    for (size_t i = 0; i < ops.size(); ++i) {
        out.ops[i].name = ops[i].name;
    }

    int accepted_log = 0;
    int feasible_log = 0;

    const std::string prefix = opt.log_prefix.empty() ? std::string("[hh]") : opt.log_prefix;
    if (opt.log_every > 0) {
        std::lock_guard<std::mutex> lk(log_mutex());
        std::cerr << prefix << " start n=" << opt.n << " hh_iters=" << opt.hh_iters << " L=" << L
                  << " objective=" << ((opt.objective == HHObjective::kS) ? "s" : "score") << " init_obj=" << cur_obj
                  << " init_s=" << init_s << " init_score=" << init_score << "\n";
    }

    for (int t = 0; t < opt.hh_iters; ++t) {
        const int k = pick_operator_ucb(out.ops, opt.ucb_c, t);
        out.ops[static_cast<size_t>(k)].selected++;
        out.attempted++;

        Candidate cand = ops[static_cast<size_t>(k)].apply(tree_poly, cur, opt, rng, eps);
        if (static_cast<int>(cand.poses.size()) != opt.n) {
            auto& st = out.ops[static_cast<size_t>(k)];
            const double reward = 0.0;
            st.mean_reward += (reward - st.mean_reward) / static_cast<double>(std::max(1, st.selected));
            continue;
        }

        out.feasible++;
        out.ops[static_cast<size_t>(k)].feasible++;
        feasible_log++;

        const double old_obj = cur_obj;
        const int pos = t % L;

        bool accept = false;
        if (cand.obj <= old_obj + 1e-15) {
            accept = true;
        } else if (cand.obj <= hist[static_cast<size_t>(pos)] + 1e-15) {
            accept = true;
        }

        hist[static_cast<size_t>(pos)] = old_obj;

        if (!accept) {
            // Reward on selection (including 0) keeps mean comparable across operators.
            auto& st = out.ops[static_cast<size_t>(k)];
            const double reward = 0.0;
            st.mean_reward += (reward - st.mean_reward) / static_cast<double>(std::max(1, st.selected));
            continue;
        }

        out.accepted++;
        out.ops[static_cast<size_t>(k)].accepted++;
        accepted_log++;

        // Reward on selection (normalized improvement; 0 for non-improvements).
        auto& st = out.ops[static_cast<size_t>(k)];
        const double reward = (cand.obj < old_obj - 1e-15)
                                  ? std::max(0.0, (old_obj - cand.obj) / std::max(1e-12, old_obj))
                                  : 0.0;
        st.mean_reward += (reward - st.mean_reward) / static_cast<double>(std::max(1, st.selected));

        cur = std::move(cand.poses);
        cur_s = cand.s;
        cur_score = cand.score;
        cur_obj = cand.obj;

        if (cur_obj < best_obj - 1e-15) {
            best = cur;
            best_s = cur_s;
            best_score = cur_score;
            best_obj = cur_obj;
            out.best_poses = best;
            out.best_s = best_s;
            out.best_score = best_score;
            out.best_obj = best_obj;
        }

        if (opt.log_every > 0 && (t % opt.log_every) == 0) {
            const double acc =
                static_cast<double>(accepted_log) / static_cast<double>(std::max(1, feasible_log));
            std::lock_guard<std::mutex> lk(log_mutex());
            std::cerr << prefix << " t=" << t << "/" << opt.hh_iters << " op=" << ops[static_cast<size_t>(k)].name
                      << " cur_obj=" << cur_obj << " best_obj=" << best_obj << " cur_s=" << cur_s << " best_s=" << best_s
                      << " acc=" << acc << " feasible=" << feasible_log << " accepted=" << accepted_log << "\n";
            accepted_log = 0;
            feasible_log = 0;
        }
    }

    if (opt.log_every > 0) {
        std::lock_guard<std::mutex> lk(log_mutex());
        std::cerr << prefix << " done best_obj=" << best_obj << " best_s=" << best_s << " best_score=" << best_score
                  << " attempted=" << out.attempted << " feasible=" << out.feasible << " accepted=" << out.accepted
                  << "\n";
    }

    return out;
}

}  // namespace santa2025
