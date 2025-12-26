#include "santa2025/collision_index.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace santa2025 {

CollisionIndex::CollisionIndex(const Polygon& tree_poly, double eps)
    : eps_(eps),
      radius_(polygon_max_radius(tree_poly)),
      nfp_(tree_poly, eps_),
      grid_(2.0 * radius_, radius_),
      poses_(),
      present_() {
    if (!(eps_ > 0.0)) {
        throw std::invalid_argument("CollisionIndex: eps must be > 0");
    }
}

void CollisionIndex::resize(int n) {
    if (n < 0) {
        throw std::invalid_argument("CollisionIndex::resize: n must be >= 0");
    }
    poses_.assign(static_cast<size_t>(n), Pose{});
    present_.assign(static_cast<size_t>(n), false);
    grid_.clear();
    grid_.reserve_ids(static_cast<size_t>(n));
}

void CollisionIndex::set_pose(int id, const Pose& p) {
    if (id < 0 || id >= size()) {
        throw std::out_of_range("CollisionIndex::set_pose: id out of range");
    }
    poses_[static_cast<size_t>(id)] = p;
    if (present_[static_cast<size_t>(id)]) {
        grid_.update(id, p.x, p.y);
    } else {
        grid_.insert(id, p.x, p.y);
        present_[static_cast<size_t>(id)] = true;
    }
}

void CollisionIndex::remove(int id) {
    if (id < 0 || id >= size()) {
        return;
    }
    if (!present_[static_cast<size_t>(id)]) {
        return;
    }
    grid_.remove(id);
    present_[static_cast<size_t>(id)] = false;
}

bool CollisionIndex::collides_with_any(const Pose& candidate, int ignore_id) const {
    return collides_with_any(candidate, ignore_id, 0.0);
}

bool CollisionIndex::collides_with_any(const Pose& candidate, int ignore_id, double extra_eps) const {
    extra_eps = std::max(0.0, extra_eps);
    // Broad phase: query candidate neighborhood by circle of radius ~ 2R.
    // Using R_query = 2R ensures we don't miss near-boundary cases when using a coarse grid.
    const double r_query = 2.0 * radius_ + eps_ + extra_eps;
    const double r2 = r_query * r_query;

    std::vector<int> ids;
    grid_.query_into(ids, candidate.x, candidate.y, r_query);

    for (const int id : ids) {
        if (id == ignore_id) {
            continue;
        }
        if (id < 0 || id >= size()) {
            continue;
        }
        if (!present_[static_cast<size_t>(id)]) {
            continue;
        }
        const Pose& other = poses_[static_cast<size_t>(id)];
        const double dx = other.x - candidate.x;
        const double dy = other.y - candidate.y;
        if (dx * dx + dy * dy > r2) {
            continue;
        }
        const double delta = candidate.deg - other.deg;
        const Point t_world{candidate.x - other.x, candidate.y - other.y};
        const Point t_other = rotate_point(t_world, -other.deg);
        const NFP& nfp = nfp_.get(delta);
        if (nfp_contains(nfp, t_other, eps_ + extra_eps)) {
            return true;
        }
    }
    return false;
}

const NFP& CollisionIndex::nfp(double delta_deg) const {
    return nfp_.get(delta_deg);
}

void CollisionIndex::translate_all(double dx, double dy) {
    if (std::abs(dx) == 0.0 && std::abs(dy) == 0.0) {
        return;
    }

    grid_.clear();
    grid_.reserve_ids(poses_.size());

    for (int id = 0; id < size(); ++id) {
        if (!present_[static_cast<size_t>(id)]) {
            continue;
        }
        Pose& p = poses_[static_cast<size_t>(id)];
        p.x += dx;
        p.y += dy;
        grid_.insert(id, p.x, p.y);
    }
}

}  // namespace santa2025
