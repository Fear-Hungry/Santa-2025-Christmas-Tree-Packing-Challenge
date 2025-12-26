#include "santa2025/order_hyper_heuristic.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "santa2025/logging.hpp"

namespace santa2025 {
namespace {

constexpr double kQuant = 1e6;

std::int64_t quant_deg(double deg) {
    return static_cast<std::int64_t>(std::llround(deg * kQuant));
}

struct RotatedBBoxCache {
    std::unordered_map<std::int64_t, BoundingBox> bboxes;
};

BoundingBox rotated_bbox_cached(const Polygon& base, double deg, RotatedBBoxCache& cache) {
    const std::int64_t k = quant_deg(deg);
    auto it = cache.bboxes.find(k);
    if (it != cache.bboxes.end()) {
        return it->second;
    }
    const BoundingBox bb = polygon_bbox(rotate_polygon(base, deg));
    cache.bboxes.emplace(k, bb);
    return bb;
}

BoundingBox bbox_for_pose(const BoundingBox& local, const Pose& p) {
    return BoundingBox{
        local.min_x + p.x,
        local.min_y + p.y,
        local.max_x + p.x,
        local.max_y + p.y,
    };
}

int default_lahc_length(int n) {
    const int v = 10 * n;
    return std::clamp(v, 50, 500);
}

double prefix_score_from_perm(const std::vector<BoundingBox>& world_bbs, const std::vector<int>& perm, int nmax) {
    if (nmax <= 0) {
        return 0.0;
    }
    if (world_bbs.empty() || perm.empty()) {
        return 0.0;
    }
    if (world_bbs.size() != perm.size()) {
        throw std::invalid_argument("prefix_score_from_perm: world_bbs.size() must equal perm.size()");
    }
    const int n = std::min(nmax, static_cast<int>(perm.size()));

    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();

    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        const int id = perm[static_cast<size_t>(i)];
        if (id < 0 || id >= static_cast<int>(world_bbs.size())) {
            throw std::invalid_argument("prefix_score_from_perm: perm contains out-of-range id");
        }
        const auto& bb = world_bbs[static_cast<size_t>(id)];
        min_x = std::min(min_x, bb.min_x);
        min_y = std::min(min_y, bb.min_y);
        max_x = std::max(max_x, bb.max_x);
        max_y = std::max(max_y, bb.max_y);
        const double s = std::max(max_x - min_x, max_y - min_y);
        total += (s * s) / static_cast<double>(i + 1);
    }
    return total;
}

std::vector<int> identity_perm(int n) {
    std::vector<int> perm(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        perm[static_cast<size_t>(i)] = i;
    }
    return perm;
}

struct Candidate {
    std::vector<int> perm;
    double score = 0.0;
};

struct Operator {
    std::string name;
    Candidate (*apply)(
        const std::vector<BoundingBox>& world_bbs,
        const std::vector<int>& cur_perm,
        int nmax_score,
        std::mt19937_64& rng
    );
};

Candidate op_swap_two(
    const std::vector<BoundingBox>& /*world_bbs*/,
    const std::vector<int>& cur_perm,
    int /*nmax_score*/,
    std::mt19937_64& rng
) {
    Candidate out;
    out.perm = cur_perm;
    const int n = static_cast<int>(out.perm.size());
    if (n < 2) {
        return out;
    }
    std::uniform_int_distribution<int> pick(0, n - 1);
    int i = pick(rng);
    int j = pick(rng);
    if (i == j) {
        j = (j + 1) % n;
    }
    if (i > j) {
        std::swap(i, j);
    }
    std::swap(out.perm[static_cast<size_t>(i)], out.perm[static_cast<size_t>(j)]);
    return out;
}

Candidate op_reverse_segment(
    const std::vector<BoundingBox>& /*world_bbs*/,
    const std::vector<int>& cur_perm,
    int /*nmax_score*/,
    std::mt19937_64& rng
) {
    Candidate out;
    out.perm = cur_perm;
    const int n = static_cast<int>(out.perm.size());
    if (n < 2) {
        return out;
    }
    std::uniform_int_distribution<int> pick(0, n - 1);
    int l = pick(rng);
    int r = pick(rng);
    if (l > r) {
        std::swap(l, r);
    }
    if (l == r) {
        r = std::min(n - 1, l + 1);
    }
    std::reverse(out.perm.begin() + l, out.perm.begin() + r + 1);
    return out;
}

Candidate op_shuffle_segment(
    const std::vector<BoundingBox>& /*world_bbs*/,
    const std::vector<int>& cur_perm,
    int /*nmax_score*/,
    std::mt19937_64& rng
) {
    Candidate out;
    out.perm = cur_perm;
    const int n = static_cast<int>(out.perm.size());
    if (n < 2) {
        return out;
    }
    std::uniform_int_distribution<int> pick(0, n - 1);
    int l = pick(rng);
    int r = pick(rng);
    if (l > r) {
        std::swap(l, r);
    }
    if (l == r) {
        r = std::min(n - 1, l + 1);
    }
    std::shuffle(out.perm.begin() + l, out.perm.begin() + r + 1, rng);
    return out;
}

Candidate op_reinsert(
    const std::vector<BoundingBox>& /*world_bbs*/,
    const std::vector<int>& cur_perm,
    int nmax_score,
    std::mt19937_64& rng
) {
    Candidate out;
    out.perm = cur_perm;
    const int n = static_cast<int>(out.perm.size());
    if (n < 2) {
        return out;
    }

    const int focus = std::clamp(nmax_score, 1, n);
    std::uniform_int_distribution<int> pick_src(0, n - 1);
    std::uniform_int_distribution<int> pick_dst(0, focus - 1);
    int src = pick_src(rng);
    int dst = pick_dst(rng);
    if (src == dst) {
        dst = (dst + 1) % n;
    }

    const int v = out.perm[static_cast<size_t>(src)];
    out.perm.erase(out.perm.begin() + src);
    out.perm.insert(out.perm.begin() + dst, v);
    return out;
}

int pick_operator_ucb(const std::vector<OrderHHOperatorStats>& stats, double ucb_c, int t) {
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

OrderHHResult optimize_prefix_order_hh(
    const Polygon& tree_poly,
    const std::vector<Pose>& poses,
    const OrderHHOptions& opt,
    double eps
) {
    (void)eps;
    if (opt.n <= 0) {
        return OrderHHResult{};
    }
    if (static_cast<int>(poses.size()) != opt.n) {
        throw std::invalid_argument("optimize_prefix_order_hh: poses.size() must equal opt.n");
    }
    if (opt.iters <= 0) {
        throw std::invalid_argument("optimize_prefix_order_hh: iters must be > 0");
    }
    if (!(opt.ucb_c >= 0.0)) {
        throw std::invalid_argument("optimize_prefix_order_hh: ucb_c must be >= 0");
    }
    if (opt.nmax_score <= 0 || opt.nmax_score > opt.n) {
        throw std::invalid_argument("optimize_prefix_order_hh: nmax_score must be in [1,n]");
    }

    const int L = (opt.lahc_length > 0) ? opt.lahc_length : default_lahc_length(opt.n);

    RotatedBBoxCache bbox_cache;
    std::vector<BoundingBox> world_bbs;
    world_bbs.reserve(poses.size());
    for (const auto& p : poses) {
        const BoundingBox local = rotated_bbox_cached(tree_poly, p.deg, bbox_cache);
        const BoundingBox bb = bbox_for_pose(local, p);
        world_bbs.push_back(bb);
    }

    std::vector<int> cur_perm = identity_perm(opt.n);
    double cur_score = prefix_score_from_perm(world_bbs, cur_perm, opt.nmax_score);

    std::vector<int> best_perm = cur_perm;
    double best_score = cur_score;

    std::vector<double> hist(static_cast<size_t>(L), cur_score);

    std::mt19937_64 rng(opt.seed);

    const std::vector<Operator> ops = {
        {"swap_two", &op_swap_two},
        {"reverse_segment", &op_reverse_segment},
        {"shuffle_segment", &op_shuffle_segment},
        {"reinsert", &op_reinsert},
    };

    OrderHHResult out;
    out.init_score = cur_score;
    out.best_score = best_score;
    out.ops.resize(ops.size());
    for (size_t i = 0; i < ops.size(); ++i) {
        out.ops[i].name = ops[i].name;
    }

    const std::string prefix = opt.log_prefix.empty() ? std::string("[order_hh]") : opt.log_prefix;
    if (opt.log_every > 0) {
        std::lock_guard<std::mutex> lk(log_mutex());
        std::cerr << prefix << " start n=" << opt.n << " iters=" << opt.iters << " L=" << L
                  << " init_score=" << cur_score << "\n";
    }

    int accepted_log = 0;
    for (int t = 0; t < opt.iters; ++t) {
        const int k = pick_operator_ucb(out.ops, opt.ucb_c, t);
        out.ops[static_cast<size_t>(k)].selected++;
        out.attempted++;

        Candidate cand = ops[static_cast<size_t>(k)].apply(world_bbs, cur_perm, opt.nmax_score, rng);
        if (static_cast<int>(cand.perm.size()) != opt.n) {
            auto& st = out.ops[static_cast<size_t>(k)];
            const double reward = 0.0;
            st.mean_reward += (reward - st.mean_reward) / static_cast<double>(std::max(1, st.selected));
            continue;
        }
        cand.score = prefix_score_from_perm(world_bbs, cand.perm, opt.nmax_score);

        const double old_score = cur_score;
        const int pos = t % L;

        bool accept = false;
        if (cand.score <= old_score + 1e-15) {
            accept = true;
        } else if (cand.score <= hist[static_cast<size_t>(pos)] + 1e-15) {
            accept = true;
        }
        hist[static_cast<size_t>(pos)] = old_score;

        if (!accept) {
            auto& st = out.ops[static_cast<size_t>(k)];
            const double reward = 0.0;
            st.mean_reward += (reward - st.mean_reward) / static_cast<double>(std::max(1, st.selected));
            continue;
        }

        out.accepted++;
        out.ops[static_cast<size_t>(k)].accepted++;
        accepted_log++;

        auto& st = out.ops[static_cast<size_t>(k)];
        const double reward = (cand.score < old_score - 1e-15)
                                  ? std::max(0.0, (old_score - cand.score) / std::max(1e-12, old_score))
                                  : 0.0;
        st.mean_reward += (reward - st.mean_reward) / static_cast<double>(std::max(1, st.selected));

        cur_perm = std::move(cand.perm);
        cur_score = cand.score;

        if (cur_score < best_score - 1e-15) {
            best_perm = cur_perm;
            best_score = cur_score;
            out.best_score = best_score;
        }

        if (opt.log_every > 0 && (t % opt.log_every) == 0) {
            const double acc = static_cast<double>(accepted_log) / static_cast<double>(std::max(1, opt.log_every));
            std::lock_guard<std::mutex> lk(log_mutex());
            std::cerr << prefix << " t=" << t << "/" << opt.iters << " op=" << ops[static_cast<size_t>(k)].name
                      << " cur_score=" << cur_score << " best_score=" << best_score << " acc=" << acc << "\n";
            accepted_log = 0;
        }
    }

    std::vector<Pose> best;
    best.reserve(poses.size());
    for (int i = 0; i < opt.n; ++i) {
        best.push_back(poses[static_cast<size_t>(best_perm[static_cast<size_t>(i)])]);
    }
    out.best_poses = std::move(best);

    if (opt.log_every > 0) {
        std::lock_guard<std::mutex> lk(log_mutex());
        std::cerr << prefix << " done best_score=" << best_score << " attempted=" << out.attempted
                  << " accepted=" << out.accepted << "\n";
    }

    return out;
}

}  // namespace santa2025
