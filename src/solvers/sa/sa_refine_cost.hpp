#pragma once

#include <algorithm>
#include <cmath>

#include "sa.hpp"

namespace sa_refine {

struct CostModel {
    const SARefiner::Params& p;
    double overlap_w0 = 0.0;
    double overlap_w1 = 0.0;
    bool soft_overlap = false;
    bool use_mtv_metric = false;
    double overlap_weight = 0.0;

    explicit CostModel(const SARefiner::Params& params)
        : p(params),
          overlap_w0((p.overlap_weight_start >= 0.0) ? p.overlap_weight_start : p.overlap_weight),
          overlap_w1((p.overlap_weight_end >= 0.0) ? p.overlap_weight_end : p.overlap_weight),
          soft_overlap(std::max(overlap_w0, overlap_w1) > 0.0),
          use_mtv_metric(p.overlap_metric == SARefiner::OverlapMetric::kMtv2) {
        overlap_weight = soft_overlap ? overlap_weight_at(0) : 0.0;
    }

    double overlap_weight_at(int t) const {
        const double w0 = overlap_w0;
        const double w1 = overlap_w1;
        if (w0 == w1) {
            return w1;
        }
        double frac = (p.iters > 1) ? static_cast<double>(t) /
                                          static_cast<double>(p.iters - 1)
                                    : 1.0;
        if (p.overlap_weight_power != 1.0) {
            frac = std::pow(frac, p.overlap_weight_power);
        }
        if (p.overlap_weight_geometric && w0 > 0.0 && w1 > 0.0) {
            return w0 * std::pow(w1 / w0, frac);
        }
        return w0 + (w1 - w0) * frac;
    }

    void update_weight(int t) {
        overlap_weight = soft_overlap ? overlap_weight_at(t) : 0.0;
    }

    double clamp_overlap(double metric) const {
        return (metric > p.overlap_eps_area) ? metric : 0.0;
    }

    double plateau_term(double width, double height) const {
        if (!(p.plateau_eps > 0.0)) {
            return 0.0;
        }
        return p.plateau_eps * std::min(width, height);
    }

    double cost_from(double width, double height, double overlap_value) const {
        const double side = std::max(width, height);
        double cost = side + plateau_term(width, height);
        if (soft_overlap && overlap_weight > 0.0) {
            cost += overlap_weight * overlap_value;
        }
        return cost;
    }

    template <typename OverlapMtvFn>
    double overlap_metric(const TreePose& a, const TreePose& b, OverlapMtvFn overlap_mtv) const {
        Point mtv{0.0, 0.0};
        double area = 0.0;
        if (!overlap_mtv(a, b, mtv, area)) {
            return 0.0;
        }
        if (!use_mtv_metric) {
            return area;
        }
        return mtv.x * mtv.x + mtv.y * mtv.y;
    }
};

}  // namespace sa_refine
