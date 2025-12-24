#pragma once

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

#include "geometry/geom.hpp"

namespace sa_refine {

struct PickedIndex {
    int idx = -1;
    bool extreme = false;
};

inline std::vector<int> build_extreme_pool(const std::vector<BoundingBox>& bbs, int topk) {
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

class IndexPicker {
public:
    IndexPicker(int n,
                const std::vector<BoundingBox>& bbs,
                const std::vector<int>& active_indices,
                const std::vector<int>& active_pos,
                const std::vector<char>* active_mask,
                bool use_mask,
                int extreme_topk,
                std::mt19937_64& rng,
                std::uniform_real_distribution<double>& uni)
        : n_(n),
          bbs_(bbs),
          active_indices_(active_indices),
          active_pos_(active_pos),
          active_mask_(active_mask),
          use_mask_(use_mask),
          extreme_topk_(extreme_topk),
          rng_(rng),
          uni_(uni) {}

    void rebuild(double gmnx, double gmxx, double gmny, double gmxy) {
        boundary_pool_ = build_boundary_pool(gmnx, gmxx, gmny, gmxy);
    }

    const std::vector<int>& boundary_pool() const {
        return boundary_pool_;
    }

    PickedIndex pick_index(double prob_extreme) {
        if (!boundary_pool_.empty() && uni_(rng_) < prob_extreme) {
            std::uniform_int_distribution<int> pick(
                0, static_cast<int>(boundary_pool_.size()) - 1);
            return PickedIndex{boundary_pool_[static_cast<size_t>(pick(rng_))], true};
        }
        if (use_mask_) {
            std::uniform_int_distribution<int> pick(
                0, static_cast<int>(active_indices_.size()) - 1);
            return PickedIndex{active_indices_[static_cast<size_t>(pick(rng_))], false};
        }
        std::uniform_int_distribution<int> pick(0, n_ - 1);
        return PickedIndex{pick(rng_), false};
    }

    int pick_other_index(int i) const {
        if (n_ <= 1) {
            return 0;
        }
        if (use_mask_) {
            if (active_indices_.size() <= 1) {
                return i;
            }
            const int pos_i = active_pos_[static_cast<size_t>(i)];
            if (pos_i < 0) {
                std::uniform_int_distribution<int> pick(
                    0, static_cast<int>(active_indices_.size()) - 1);
                return active_indices_[static_cast<size_t>(pick(rng_))];
            }
            std::uniform_int_distribution<int> pick(
                0, static_cast<int>(active_indices_.size()) - 2);
            int jpos = pick(rng_);
            if (jpos >= pos_i) {
                ++jpos;
            }
            return active_indices_[static_cast<size_t>(jpos)];
        }
        std::uniform_int_distribution<int> pick(0, n_ - 2);
        int j = pick(rng_);
        if (j >= i) {
            ++j;
        }
        return j;
    }

private:
    bool is_active(int idx) const {
        if (!use_mask_) {
            return true;
        }
        if (!active_mask_) {
            return false;
        }
        return (*active_mask_)[static_cast<size_t>(idx)] != 0;
    }

    std::vector<int> build_boundary_pool(double gmnx, double gmxx, double gmny, double gmxy) const {
        const double boundary_tol = 1e-9;
        std::vector<int> boundary;
        boundary.reserve(static_cast<size_t>(n_));
        for (int i = 0; i < n_; ++i) {
            if (!is_active(i)) {
                continue;
            }
            const auto& bb = bbs_[static_cast<size_t>(i)];
            if (bb.min_x <= gmnx + boundary_tol || bb.max_x >= gmxx - boundary_tol ||
                bb.min_y <= gmny + boundary_tol || bb.max_y >= gmxy - boundary_tol) {
                boundary.push_back(i);
            }
        }
        if (boundary.empty()) {
            boundary = build_extreme_pool(bbs_, extreme_topk_);
            if (use_mask_) {
                std::vector<int> filtered;
                filtered.reserve(boundary.size());
                for (int idx : boundary) {
                    if (is_active(idx)) {
                        filtered.push_back(idx);
                    }
                }
                boundary = std::move(filtered);
            }
        }
        return boundary;
    }

    int n_ = 0;
    const std::vector<BoundingBox>& bbs_;
    const std::vector<int>& active_indices_;
    const std::vector<int>& active_pos_;
    const std::vector<char>* active_mask_ = nullptr;
    bool use_mask_ = false;
    int extreme_topk_ = 1;
    std::mt19937_64& rng_;
    std::uniform_real_distribution<double>& uni_;
    std::vector<int> boundary_pool_;
};

}  // namespace sa_refine
