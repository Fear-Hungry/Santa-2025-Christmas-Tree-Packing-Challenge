#pragma once

#include <vector>

#include "santa2025/geometry.hpp"
#include "santa2025/nfp.hpp"
#include "santa2025/spatial_hash_grid.hpp"

namespace santa2025 {

class CollisionIndex {
public:
    explicit CollisionIndex(const Polygon& tree_poly, double eps = 1e-12);

    void resize(int n);
    int size() const { return static_cast<int>(poses_.size()); }

    double tree_radius() const { return radius_; }
    double eps() const { return eps_; }

    // Sets/updates a tree pose and updates the spatial index.
    void set_pose(int id, const Pose& p);

    // Removes a tree from the index (keeps its last pose stored).
    void remove(int id);

    // Checks whether `candidate` overlaps any currently-present tree, optionally ignoring one id.
    bool collides_with_any(const Pose& candidate, int ignore_id = -1) const;

    // Same as collides_with_any, but treats contacts within `extra_eps` as collision (safety margin).
    bool collides_with_any(const Pose& candidate, int ignore_id, double extra_eps) const;

    // Access to cached NFP geometry for candidate generation.
    const NFP& nfp(double delta_deg) const;

    // Translates all currently-present poses and rebuilds the spatial index.
    void translate_all(double dx, double dy);

private:
    double eps_ = 1e-12;
    double radius_ = 0.0;

    mutable NFPCache nfp_;
    SpatialHashGrid grid_;

    std::vector<Pose> poses_;
    std::vector<bool> present_;
};

}  // namespace santa2025
