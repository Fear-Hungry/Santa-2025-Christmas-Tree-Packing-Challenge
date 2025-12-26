#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "santa2025/geometry.hpp"

namespace santa2025 {

struct NFP {
    double delta_deg = 0.0;
    std::vector<Polygon> pieces;
    std::vector<BoundingBox> piece_bboxes;
};

class NFPCache {
public:
    explicit NFPCache(Polygon base_poly, double eps = 1e-12);

    const NFP& get(double delta_deg);

    double eps() const { return eps_; }

private:
    double eps_ = 1e-12;
    Polygon base_poly_;
    std::vector<Polygon> base_tris_;
    std::unordered_map<std::int64_t, NFP> cache_;
};

bool nfp_contains(const NFP& nfp, const Point& t, double eps = 1e-12);

struct Pose {
    double x = 0.0;
    double y = 0.0;
    double deg = 0.0;
};

bool trees_overlap_nfp(NFPCache& cache, const Pose& a, const Pose& b);

}  // namespace santa2025

